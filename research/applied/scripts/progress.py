"""
Progress tracking for long-running CLI jobs.

Two surfaces:
  1. Interactive — tqdm progress bars when stdout is a TTY.
  2. Unattended — a heartbeat log file (`_progress.log`) in the job's output
     directory, updated on every milestone and every N iterations.

The heartbeat file means you can `tail -f research/applied/results/.../_progress.log`
from another shell, or just `cat` it the next morning, without keeping the
launching terminal open.

Usage in a script:

    from progress import ProgressTracker

    tracker = ProgressTracker(out_dir, job_name="morris_sa", total=130)
    tracker.start("Loading climate slice…")
    # … load data …
    tracker.milestone("climate loaded", n_lsoas=185)

    for i in tracker.iter(range(130), desc="Morris trajectories"):
        # … one expensive run …
        pass

    tracker.finish("Wrote 130 elementary effects.")
"""

from __future__ import annotations

import json
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, TypeVar

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

T = TypeVar("T")


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %z")


class ProgressTracker:
    """File-backed progress tracker.

    Writes two artifacts to `out_dir`:
      _progress.log   — human-readable, append-only, one line per event
      _progress.json  — machine-readable snapshot, overwritten on each update
                        (counts, ETA, elapsed, last_msg)
    """

    def __init__(
        self,
        out_dir: Path,
        *,
        job_name: str,
        total: int | None = None,
        heartbeat_every: int = 1,
        print_every_seconds: float = 5.0,
    ):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.job_name = job_name
        self.total = total
        self.heartbeat_every = max(1, int(heartbeat_every))
        # Floor for in-terminal carriage-return updates from tick().
        # Independent from heartbeat_every (which controls log-file writes).
        self.print_every_seconds = float(print_every_seconds)

        self.log_path = self.out_dir / "_progress.log"
        self.json_path = self.out_dir / "_progress.json"

        self._t0 = time.monotonic()
        self._count = 0
        self._last_msg = ""
        self._last_inline_print = 0.0
        self._inline_active = False
        # In-terminal updates use carriage-return ("\r") to overwrite the
        # previous line. Only do this if stderr is a TTY; otherwise we'd
        # flood log files / nohup output with \r characters.
        self._inline_enabled = sys.stderr.isatty()

    # ── public API ────────────────────────────────────────────────────────

    def start(self, msg: str = "") -> None:
        line = f"START  {self.job_name}" + (f" — {msg}" if msg else "")
        self._emit(line, milestone=True)

    def milestone(self, msg: str, **extras) -> None:
        suffix = " " + " ".join(f"{k}={v}" for k, v in extras.items()) if extras else ""
        line = f"STEP   {msg}{suffix}"
        self._emit(line, milestone=True)

    def warn(self, msg: str) -> None:
        self._emit(f"WARN   {msg}", milestone=True)

    def tick(self, msg: str = "") -> None:
        """Increment counter by 1. Writes to log on `heartbeat_every`;
        overwrites in-terminal line every `print_every_seconds`.
        """
        self._count += 1
        # In-terminal carriage-return progress line (no second terminal needed).
        # Throttled to print_every_seconds so we don't slow down tight loops.
        now = time.monotonic()
        if self._inline_enabled and (now - self._last_inline_print) >= self.print_every_seconds:
            self._print_inline(msg)
            self._last_inline_print = now
        # Log-file heartbeat (separate cadence, append-only).
        if (self._count % self.heartbeat_every == 0) or (self._count == self.total):
            if not msg and self.total:
                msg = f"{self._count}/{self.total}"
            self._emit(f"TICK   {msg}", milestone=False)

    def _print_inline(self, msg: str = "") -> None:
        """Write a single carriage-return line to stderr — overwrites in place."""
        pct = (100.0 * self._count / self.total) if self.total else None
        eta = self._eta_str()
        bits = [f"{self.job_name}:"]
        if self.total:
            bits.append(f"{self._count}/{self.total}")
            if pct is not None:
                bits.append(f"({pct:5.1f}%)")
        else:
            bits.append(f"{self._count}")
        bits.append(f"elapsed={self._elapsed_str()}")
        if eta:
            bits.append(f"eta={eta}")
        if msg:
            bits.append(msg)
        line = " ".join(bits)
        # Pad to 100 chars so leftover characters from the previous line are wiped.
        sys.stderr.write("\r" + line.ljust(100)[:100])
        sys.stderr.flush()
        self._inline_active = True

    def finish(self, msg: str = "") -> None:
        elapsed = self._elapsed_str()
        line = f"DONE   {self.job_name} ({elapsed})" + (f" — {msg}" if msg else "")
        self._emit(line, milestone=True)

    def iter(self, it: Iterable[T], *, desc: str | None = None) -> Iterator[T]:
        """Wrap an iterable with progress reporting.

        Uses tqdm in interactive contexts (which handles its own in-terminal
        display); always writes heartbeats to the log file regardless.
        """
        using_tqdm = _HAS_TQDM and sys.stderr.isatty()
        if using_tqdm:
            it = tqdm(it, desc=desc or self.job_name, total=self.total)
        # Suppress our own carriage-return inline output while tqdm is driving
        # the terminal — would otherwise produce double rendering.
        prev_inline = self._inline_enabled
        if using_tqdm:
            self._inline_enabled = False
        try:
            for item in it:
                yield item
                self.tick(desc or "")
        finally:
            self._inline_enabled = prev_inline

    @contextmanager
    def section(self, name: str):
        """Time a block, log start + end."""
        t0 = time.monotonic()
        self.milestone(f"begin {name}")
        try:
            yield
        finally:
            dt = time.monotonic() - t0
            self.milestone(f"end   {name}", elapsed=f"{dt:.1f}s")

    # ── internals ─────────────────────────────────────────────────────────

    def _elapsed_str(self) -> str:
        dt = time.monotonic() - self._t0
        h, rem = divmod(int(dt), 3600)
        m, s = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def _eta_str(self) -> str | None:
        if not self.total or self._count == 0:
            return None
        rate = self._count / (time.monotonic() - self._t0)
        if rate <= 0:
            return None
        remaining = (self.total - self._count) / rate
        h, rem = divmod(int(remaining), 3600)
        m, s = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def _emit(self, line: str, *, milestone: bool) -> None:
        self._last_msg = line
        stamp = _now_iso()
        with open(self.log_path, "a") as f:
            f.write(f"[{stamp}] {line}\n")
        snapshot = {
            "job":         self.job_name,
            "started":     getattr(self, "_started_iso", None),
            "now":         stamp,
            "elapsed":     self._elapsed_str(),
            "count":       self._count,
            "total":       self.total,
            "pct":         (100.0 * self._count / self.total) if self.total else None,
            "eta":         self._eta_str(),
            "last_msg":    line,
        }
        if not hasattr(self, "_started_iso"):
            self._started_iso = stamp
            snapshot["started"] = stamp
        with open(self.json_path, "w") as f:
            json.dump(snapshot, f, indent=2)
        # Always echo milestones to stderr so launching shell sees them.
        if milestone:
            # If we're mid carriage-return progress line, drop a newline so
            # the milestone doesn't append onto the partial line.
            if self._inline_active:
                sys.stderr.write("\n")
                self._inline_active = False
            print(f"[{stamp}] {line}", file=sys.stderr, flush=True)


__all__ = ["ProgressTracker"]
