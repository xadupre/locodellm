"""Writes a flushed, human-readable execution transcript while a run is active."""

from __future__ import annotations

import tempfile
from contextlib import AbstractContextManager, ExitStack
from pathlib import Path
from typing import IO, Any


class LiveLog(AbstractContextManager["LiveLog"]):
    """Keeps a live transcript available after its writer has closed."""

    def __init__(self, filename: str = "") -> None:
        self.stack = ExitStack()
        self.stream: IO[str]
        if filename:
            self.path = Path(filename).resolve()
            self.stream = self.stack.enter_context(self.path.open("a", encoding="utf-8"))
        else:
            temporary = self.stack.enter_context(
                tempfile.NamedTemporaryFile(  # noqa: SIM115 - managed by ExitStack
                    mode="w",
                    encoding="utf-8",
                    prefix="locodellm-live-",
                    suffix=".log",
                    delete=False,
                )
            )
            self.stream = temporary.file
            self.path = Path(temporary.name).resolve()
        self.model_streamed = False

    def __exit__(self, *exception: object) -> None:
        self.stack.close()

    def write(self, text: str) -> None:
        """Appends text and flushes it immediately for external readers."""
        self.stream.write(text)
        self.stream.flush()

    def section(self, name: str, text: str = "") -> None:
        """Appends a labelled section to the transcript."""
        self.write(f"\n[{name}]\n{text}\n")

    def model_text(self, text: str) -> None:
        """Appends a decoded model fragment without adding formatting."""
        if text:
            self.model_streamed = True
            self.write(text)

    def model_streamer(self, tokenizer: Any) -> Any:
        """Creates a synchronous Transformers streamer writing completed text fragments."""
        from transformers import TextStreamer

        log = self

        class LogStreamer(TextStreamer):
            def on_finalized_text(self, text: str, stream_end: bool = False) -> None:
                """Flushes decoded text without waiting for generation to finish."""
                log.model_text(text)

        return LogStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False)
