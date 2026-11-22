"""
Thread-safe, non-blocking Text-to-Speech engine using pyttsx3 + Windows SAPI5.

Architecture contract (§ E):
  • A single background daemon thread owns the pyttsx3 engine for COM stability.
  • speak(text) enqueues a string without blocking the caller.
  • clear_queue() drains pending items.
  • stop() sends a sentinel to the worker and joins the thread on exit.
"""

import queue
import threading

import pyttsx3

# ---------------------------------------------------------------------------
# TTS Configuration
# ---------------------------------------------------------------------------
_SPEECH_RATE: int = 160     # words per minute
_SPEECH_VOLUME: float = 1.0  # 0.0 → 1.0
_SENTINEL = None             # sentinel value signalling worker shutdown


class TTSEngine:
    """
    Non-blocking TTS engine backed by a pyttsx3 worker thread.

    The pyttsx3 engine is instantiated *inside* the worker thread so that
    Windows SAPI5 COM objects are created and used on the same thread,
    avoiding cross-thread COM access errors.

    Usage
    -----
    tts = TTSEngine()
    tts.speak("Hello")
    tts.stop()          # call on application exit
    """

    def __init__(self) -> None:
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._thread = threading.Thread(
            target=self._worker,
            name="TTSWorker",
            daemon=True,  # exits automatically when the main process exits
        )
        self._thread.start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def speak(self, text: str) -> None:
        """
        Enqueue *text* for speech synthesis.

        Returns immediately — never blocks the caller.
        Silently ignores empty or whitespace-only strings.
        """
        if not isinstance(text, str) or not text.strip():
            return
        self._queue.put(text)

    def clear_queue(self) -> None:
        """
        Drain all pending speech items from the queue.

        Useful when the user changes sign mid-word and stale utterances
        should be discarded rather than spoken out of context.
        """
        cleared = 0
        while True:
            try:
                self._queue.get_nowait()
                cleared += 1
            except queue.Empty:
                break
        if cleared:
            print(f"[tts_engine] Cleared {cleared} pending speech item(s).")

    def stop(self) -> None:
        """
        Signal the worker thread to shut down cleanly and wait for it to join.

        Call this once on application exit (e.g. inside a finally block).
        """
        self._queue.put(_SENTINEL)
        self._thread.join(timeout=3.0)
        if self._thread.is_alive():
            print("[tts_engine] Warning: TTS worker thread did not stop within timeout.")
        else:
            print("[tts_engine] Worker thread stopped cleanly.")

    # ------------------------------------------------------------------
    # Private worker
    # ------------------------------------------------------------------

    def _worker(self) -> None:
        """
        Background thread: owns the pyttsx3 engine for SAPI5 COM stability.

        pyttsx3.init() is called here so that all COM objects are created and
        consumed on the same OS thread — the Windows SAPI5 driver requires this.
        Loops until the sentinel (None) is dequeued.
        """
        engine = pyttsx3.init()
        engine.setProperty("rate", _SPEECH_RATE)
        engine.setProperty("volume", _SPEECH_VOLUME)

        print(
            f"[tts_engine] Worker started — "
            f"Rate={_SPEECH_RATE} WPM, Volume={_SPEECH_VOLUME}"
        )

        while True:
            text = self._queue.get()   # blocks until an item is available
            if text is _SENTINEL:      # shutdown sentinel received
                print("[tts_engine] Worker received stop signal. Exiting.")
                break
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception as exc:
                print(f"[tts_engine] Error during speech synthesis: {exc}")

        # Cleanly release the SAPI5 engine before the thread exits
        try:
            engine.stop()
        except Exception:
            pass
