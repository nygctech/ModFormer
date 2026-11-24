"""The interactive session: ``modformer`` with nothing after it.

One loop reads a line, tells a command from an input, runs it through the same functions the
one-shot command uses, and returns to the prompt whatever happened. The input line is a small
prompt_toolkit box on a terminal, a plain reader when that is unavailable, and the same plain
reader without a prompt when the commands arrive on a pipe (``modformer --session < lines``).

Presentation lives in :mod:`modformer.cli` (``present``, ``present_line``, ``show``) and
:mod:`modformer.term`; this module decides what to run and what state to keep.
"""
from __future__ import annotations

import base64
import contextlib
import copy
import io
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path
from typing import Callable, Mapping, Optional

from . import CONTEXTS, __version__, cli, term

#: name -> (argument hint, one line of help)
COMMANDS: dict[str, tuple[str, str]] = {
    "site": ("N", "the 64-nt stretch that holds position N"),
    "region": ("A-B", "positions A to B"),
    "all": ("", "the whole transcript again"),
    "model": ("[base | ft CONTEXT]", "switch model; with no argument, list them"),
    "top": ("K", "list K sites in the card shown (default 5)"),
    "save": ("[FILE]", "write the table shown (TSV)"),
    "plot": ("[PNG]", "save the rate-track figure"),
    "copy": ("[table]", "copy the result as text, or the table as TSV"),
    "demo": ("", "score the bundled APP region"),
    "help": ("", "commands, inputs and keys"),
    "clear": ("", "clear the screen"),
    "quit": ("", "leave"),
}
_HIDDEN = ("helix",)
_WORDS = {"help": "help", "quit": "quit", "exit": "quit"}
_LINE_OPTIONS = ("--species", "--isoform", "--spliced", "--region", "--site")
_CHIP_AT = 200  # a paste longer than this, or of several lines, is shown as a chip
_OSC52_MAX = 74_994  # largest base64 payload sent to a terminal clipboard
_CUT_LENGTHS = (1023, 1024, 4095, 4096)  # line lengths at which a terminal cut a plain-mode line


class LineError(ValueError):
    """A line that cannot be run; the message says why."""


# Reading a line -----------------------------------------------------------------------
def resolve_command(word: str) -> str:
    """The command a typed word stands for: an exact name, or the one it is a prefix of."""
    word = word.lstrip("/").lower()
    if word in COMMANDS or word in _HIDDEN:
        return word
    hits = [c for c in COMMANDS if c.startswith(word)] if word else ["help"]
    if len(hits) == 1:
        return hits[0]
    if hits:
        raise LineError(f"/{word} could be " + " or ".join(f"/{h}" for h in hits))
    raise LineError(f"unknown command /{word}. /help lists the commands.")


def read_region(text: str, complaint: str) -> str:
    """``"A-B"`` from typed ``A-B`` (or ``A:B``), the lower position first, so 900-800 means
    800-900. Raises a LineError with ``complaint`` for anything else."""
    m = re.fullmatch(r"(\d+)\s*[-:]\s*(\d+)", text)
    if not m:
        raise LineError(complaint)
    first, last = sorted((int(m.group(1)), int(m.group(2))))
    return f"{first}-{last}"


def read_site(text: str, complaint: str) -> int:
    """The position in typed ``N``; a LineError with ``complaint`` for anything else."""
    if not re.fullmatch(r"-?\d+", text):
        raise LineError(complaint)
    return int(text)


def split_line(line: str, pastes: Optional[Mapping[str, str]] = None) -> tuple[str, str, object]:
    """Tell what a typed line is.

    Returns ``("empty", "", None)``, ``("command", name, argument text)``,
    ``("fasta", text, None)`` for pasted FASTA, or ``("input", text, options)``, where
    ``options`` holds the ``--species``, ``--isoform``, ``--spliced``, ``--region`` and ``--site``
    that followed it (a region as ``"A-B"`` with A not above B, a site as an int).
    ``pastes`` maps a paste chip's text to what was pasted; the chip is put back after the
    line is split, so pasted text is never read for options.
    """
    pastes = pastes or {}
    text = line.strip()
    if not text:
        return "empty", "", None
    if text in pastes:
        text = pastes[text].strip()
        return ("fasta", text, None) if text.startswith(">") else ("input", text, {})
    if text.startswith("/"):
        first, _, rest = text.partition(" ")
        try:
            return "command", resolve_command(first), rest.strip()
        except LineError:
            try:
                tokens = shlex.split(text)
            except ValueError:
                tokens = text.split()
            cut = next((i for i, t in enumerate(tokens) if t.startswith("--")), len(tokens))
            path = " ".join(tokens[:cut])  # an absolute path, with or without options after it
            if not os.path.exists(path):
                if tokens and "/" in tokens[0][1:]:
                    raise LineError(f"no such file: {path}") from None
                raise
    if text.lower() in _WORDS:
        return "command", _WORDS[text.lower()], ""
    if text.startswith(">"):
        return "fasta", text, None
    slots = {}
    for k, (chip, data) in enumerate(pastes.items()):
        if chip in text:
            slots[f"{k}"] = data
            text = text.replace(chip, f"{k}")
    try:
        tokens = shlex.split(text)
    except ValueError as exc:
        raise LineError(f"cannot read the line: {exc}") from None
    cut = next((i for i, t in enumerate(tokens) if t.startswith("--")), len(tokens))
    payload = " ".join(tokens[:cut])
    for mark, data in slots.items():
        payload = payload.replace(mark, data)
    payload = os.path.expanduser(payload) if payload.startswith("~") else payload
    if not payload:
        raise LineError("no input before the options")
    options: dict = {}
    rest_tokens = tokens[cut:]
    while rest_tokens:
        flag = rest_tokens.pop(0)
        flag, eq, value = flag.partition("=")
        if flag == "--spliced" and not eq:
            options["spliced"] = True
        elif flag in ("--species", "--isoform", "--region", "--site"):
            if not eq:
                if not rest_tokens:
                    raise LineError(f"{flag} needs a value")
                value = rest_tokens.pop(0)
            if flag == "--region":
                value = read_region(value, "--region takes two positions, for example "
                                           "--region 1200-1500")
            elif flag == "--site":
                value = read_site(value, "--site takes one position, for example --site 825")
            options[flag[2:]] = value
        else:
            raise LineError(f"after an input only {', '.join(_LINE_OPTIONS)} are accepted; "
                            "/help lists the commands for the rest")
    if "site" in options and "region" in options:
        raise LineError("give --site or --region, not both")
    return ("fasta" if payload.lstrip().startswith(">") else "input"), payload, options


def option_text(options: Optional[Mapping[str, object]]) -> str:
    """The line options as typed (`` --region 1200-1500``), in the order given; empty for none."""
    out = ""
    for key, value in (options or {}).items():
        out += f" --{key}" if value is True else f" --{key} {shlex.quote(str(value))}"
    return out


def echo(kind: str, text: str, width: int, unicode: bool = True,
         options: Optional[Mapping[str, object]] = None) -> str:
    """The one line that stands for an input in the scrollback: kind, size, both ends, then the
    options typed after it. A line too long for ``width`` is cut in its input, never its options."""
    sep = " · " if unicode else " | "
    n = len("".join(text.split()))
    if kind == "sequence":
        letters = "".join(text.split())
        body = letters if n <= 28 else letters[:12] + ("…" if unicode else "...") + letters[-12:]
        out = f"sequence{sep}{n:,} nt{sep}{body}"
    elif kind == "fasta":
        heads = sum(1 for ln in text.splitlines() if ln.strip().startswith(">"))
        body = sum(len("".join(ln.split())) for ln in text.splitlines()
                   if ln.strip() and not ln.strip().startswith(">"))
        out = f"fasta{sep}{heads} record{'s' * (heads != 1)}{sep}{body:,} nt"
    else:
        out = " ".join(text.split())
    tail = option_text(options)
    if len(out) + len(tail) <= width:
        return out + tail
    dots = "…" if unicode else "."
    room = width - len(tail) - 1
    if tail and room >= 8:  # the options stay whole; the input gives way
        return out[:room] + dots + tail
    out += tail
    return out[:max(1, width - 1)] + dots


def collect_fasta(first: str, more: Callable[[int], Optional[str]],
                  say: Callable[[str], None] = cli._err) -> str:
    """The text of a FASTA block that ``first`` (a ``>`` line) opened. ``more(n)`` reads the next
    line when ``n`` lines are held and returns None at the end of input; an empty line closes
    the block. Sequence never starts with ``/``, so a line that does is a command typed with the
    block open: the block is dropped, and said to be, and that command is returned to be run."""
    lines = [first]
    while True:
        nxt = more(len(lines))
        if nxt is None or not nxt.strip():
            return "\n".join(lines)
        if nxt.lstrip().startswith("/"):
            say(f"  FASTA block dropped ({len(lines)} line{'s' * (len(lines) != 1)} not scored)")
            return nxt
        lines.append(nxt)


def read_plain(stream, prompt: str = "", tty: bool = False) -> str:
    """One line from ``stream``; a line that starts with ``>`` opens a FASTA block that runs to
    the next empty line (or to a ``/`` command, which drops it). Raises EOFError at the end of
    input. With ``tty`` a line whose length is exactly what a terminal passes in one read is
    refused: it was probably cut."""
    def get(p: str) -> Optional[str]:
        if p:
            return cli.ask(p)
        line = stream.readline()
        return None if line == "" else line.rstrip("\r\n")

    first = get(prompt)
    if first is None:
        raise EOFError
    if first.lstrip().startswith(">"):
        return collect_fasta(first, lambda n: get("fasta> " if prompt else ""))
    if tty and len(first) in _CUT_LENGTHS:
        raise LineError(f"this line is exactly {len(first):,} characters, the most a terminal "
                        "passes in one line, so it was probably cut. Give a file path instead.")
    return first


# Narrowing ----------------------------------------------------------------------------
def narrow(table, seq: str, site: Optional[int] = None,
           region: Optional[tuple[int, int]] = None):
    """The rows of a whole-transcript table that a ``site`` or ``region`` run would report.

    The tile grid never moves, so a position has the same sequence context and partner as in a
    full run, and the answer is a slice of the full table. Scores can differ in the last digits
    from batching.
    ``seq`` is the normalized sequence, for the structure excerpt.
    Raises the ValueError ``predict`` would for a position outside the sequence.
    """
    from . import _top_fold
    from .tiling import ANCHOR, plan_tiles

    n = table.attrs["length"]
    tiles = plan_tiles(n, site=site, region=region)
    first, last = 1, n
    if site is not None:
        first = ((site - 1) // ANCHOR) * ANCHOR + 1
        last = min(n, first + ANCHOR - 1)
    elif region is not None:
        first, last = region
    out = table[(table["position"] >= first) & (table["position"] <= last)].reset_index(drop=True)
    out.attrs = dict(table.attrs, tiles=len(tiles), site=site, region=region)
    _top_fold(out, seq, table.attrs["folds"])
    return out


# Clipboard ----------------------------------------------------------------------------
def copy_text(text: str, what: str, fallback: Path, *, env: Optional[Mapping[str, str]] = None,
              tty=None, which: Callable = shutil.which, run: Callable = subprocess.run) -> str:
    """Put ``text`` where the person can paste it, and return the sentence that says what
    happened. The first clipboard program present is tried, then the terminal's own clipboard
    escape; when neither applies the text is saved to ``fallback``. ``tty`` is the terminal
    stream, or None when there is none (a pipe), which skips both clipboard routes. The word
    "copied" is used only after a program exited 0."""
    env = os.environ if env is None else env
    reason = "no terminal to copy to" if tty is None else "no clipboard program here"
    if tty is not None:
        programs = [("pbcopy", ["pbcopy"])]
        if env.get("WAYLAND_DISPLAY"):
            programs.append(("wl-copy", ["wl-copy"]))
        if env.get("DISPLAY"):
            programs.append(("xclip", ["xclip", "-selection", "clipboard"]))
        for name, argv in programs:
            if not which(argv[0]):
                continue
            try:
                done = run(argv, input=text.encode(), stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=5)
            except (OSError, subprocess.SubprocessError):
                continue
            if done.returncode == 0:
                return f"copied {what} with {name}"
        payload = base64.b64encode(text.encode()).decode()
        if len(payload) <= _OSC52_MAX:
            tty.write(f"\x1b]52;c;{payload}\x07")
            tty.flush()
            return (f"sent {what} to your terminal's clipboard (OSC 52). Terminals do not "
                    "confirm this: if paste comes up empty, use /save.")
        reason = "no clipboard program here, and it is too much for the terminal's clipboard"
    again = " (replaced)" if fallback.exists() else ""
    fallback.parent.mkdir(parents=True, exist_ok=True)
    fallback.write_text(text)
    return f"nothing was copied: {reason}. Saved instead: {fallback}{again}"


# The input box ------------------------------------------------------------------------
class Box:
    """The input line on a terminal: a rule, the line, a rule, and a status line.

    A small prompt_toolkit Application written to standard error. It reads in raw mode, so a
    pasted line is not limited in length; a long or multi-line bracketed paste enters as one
    chip and is put back when Enter is pressed.
    """

    def __init__(self) -> None:
        os.environ.setdefault("PROMPT_TOOLKIT_NO_CPR", "1")  # no cursor-position round trip
        from prompt_toolkit.history import InMemoryHistory
        from prompt_toolkit.output import create_output

        self.pastes: dict[str, str] = {}
        self.history = InMemoryHistory()
        self.output = create_output(stdout=sys.stderr)

    def read(self, status: Callable[[int], str]) -> tuple[str, dict[str, str]]:
        """One line and the chips in it. Raises EOFError (Ctrl-D) and KeyboardInterrupt."""
        from prompt_toolkit.application import Application
        from prompt_toolkit.buffer import Buffer
        from prompt_toolkit.completion import Completer, Completion
        from prompt_toolkit.filters import Condition
        from prompt_toolkit.key_binding import KeyBindings, merge_key_bindings
        from prompt_toolkit.key_binding.defaults import load_key_bindings
        from prompt_toolkit.keys import Keys
        from prompt_toolkit.layout import HSplit, Layout, VSplit, Window
        from prompt_toolkit.layout.containers import VerticalAlign
        from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl

        class Commands(Completer):
            def get_completions(self, document, event):
                text = document.text_before_cursor
                if text.startswith("/") and " " not in text:
                    for name in COMMANDS:
                        if name.startswith(text[1:]):
                            yield Completion(f"/{name} ", start_position=-len(text),
                                             display=f"/{name}")

        app: Application
        buf = Buffer(history=self.history, multiline=False, completer=Commands(),
                     complete_while_typing=False,
                     accept_handler=lambda b: app.exit(result=b.text))
        keys = KeyBindings()

        @keys.add(Keys.BracketedPaste)
        def _(event):
            data = event.data.replace("\r\n", "\n").replace("\r", "\n")
            if len(data) > _CHIP_AT or "\n" in data.strip():
                chip = f"[pasted #{len(self.pastes) + 1}: {len(''.join(data.split())):,} characters]"
                self.pastes[chip] = data
                data = chip
            buf.insert_text(data)

        @keys.add("c-c")
        def _(event):
            if buf.text:
                buf.reset()
            else:
                event.app.exit(exception=KeyboardInterrupt())

        @keys.add("c-d", filter=Condition(lambda: not buf.text))
        def _(event):
            event.app.exit(exception=EOFError())

        def width() -> int:
            return max(20, self.output.get_size().columns)

        def below() -> str:
            w, text = width(), buf.text
            if text.startswith("/") and " " not in text:
                hits = [(f"/{c}", h, d) for c, (h, d) in COMMANDS.items() if c.startswith(text[1:])]
                if hits:
                    return "\n".join(_cut(f"  {c:<8} {h:<12} {d}", w) for c, h, d in hits)
            if len(text) > 5000 and not any(c in text for c in self.pastes):
                return _cut("  this terminal pasted without bracketed paste: for long "
                            "sequences give a FASTA file", w)
            return status_line(status(w), w)

        def rule() -> str:
            return "─" * (width() - 1)

        root = HSplit([
            Window(FormattedTextControl(rule), height=1),
            VSplit([Window(FormattedTextControl("❯ "), width=2, dont_extend_height=True),
                    Window(BufferControl(buf), wrap_lines=False, height=1)]),
            Window(FormattedTextControl(rule), height=1),
            Window(FormattedTextControl(below), dont_extend_height=True),
        ], align=VerticalAlign.TOP)
        app = Application(layout=Layout(root, focused_element=buf), output=self.output,
                          key_bindings=merge_key_bindings([load_key_bindings(), keys]),
                          erase_when_done=True, full_screen=False)
        text = app.run()
        return text, dict(self.pastes)


def _fill(text: str, caps) -> str:
    """``text`` folded to the terminal width and indented two spaces. A long word, such as a file
    path, stays whole so it can be copied."""
    return textwrap.fill(text, max(30, caps.width - 1), initial_indent="  ",
                         subsequent_indent="    ", break_long_words=False, break_on_hyphens=False)


def _cut(text: str, width: int) -> str:
    return text if len(text) <= width - 1 else text[:max(1, width - 2)] + "…"


def _status_hint() -> str:
    return f"/ commands · ctrl-d · v{__version__}"


def status_line(left: str, width: int) -> str:
    """The line under the input box: what the session holds on the left and, when it fits with a
    gap, the commands hint and the version on the right. Never wider than ``width - 1``."""
    right = _status_hint()
    if width >= 60 and len(left) + len(right) + 2 < width:
        return left + right.rjust(width - 1 - len(left))
    return _cut(left, width)


_armed = False  # Ctrl-C interrupts only while a line is being handled (or a plain prompt waits)
_hit = -1e9  # when it last did
_redelivered = 0  # presses re-sent after Python swallowed the exception, since the last arming
_REDELIVER_MAX = 3


def _on_sigint(signum, frame) -> None:
    """Between two reads the terminal is in cooked mode and a Ctrl-C there is a real signal; it
    is dropped unless something is armed. A burst interrupts once: a press within a second of
    the last is dropped, a later one interrupts again."""
    global _hit
    if _armed and time.monotonic() - _hit > 1.0:
        _hit = time.monotonic()
        raise KeyboardInterrupt


def _unraisable(unraisable) -> None:
    """``sys.unraisablehook`` for the session. A Ctrl-C can land while Python runs a weakref
    callback or a ``__del__`` (a model load frees objects all the time); Python prints the
    KeyboardInterrupt raised there as "Exception ignored" and drops it, and the press is lost.
    Here the press is put back: it is sent again, outside the callback, and not counted as
    part of the burst. Everything else goes to the default hook."""
    global _hit, _redelivered
    if (issubclass(unraisable.exc_type or Exception, KeyboardInterrupt) and _armed
            and _redelivered < _REDELIVER_MAX):
        _redelivered += 1
        _hit = -1e9
        # Not from here: the handler would run inside this hook, which Python also swallows.
        threading.Timer(0.02, os.kill, (os.getpid(), signal.SIGINT)).start()
        return
    sys.__unraisablehook__(unraisable)


@contextlib.contextmanager
def _ctrl_c_handling():
    """The session's Ctrl-C: the signal handler, and the hook that puts back a press Python
    swallowed. Both are restored on the way out."""
    old_signal = signal.signal(signal.SIGINT, _on_sigint)
    old_hook = sys.unraisablehook
    sys.unraisablehook = _unraisable
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, old_signal)
        sys.unraisablehook = old_hook


@contextlib.contextmanager
def _interruptible():
    global _armed, _redelivered
    _armed = True
    _redelivered = 0
    try:
        yield
    finally:
        _armed = False


@contextlib.contextmanager
def _keys_quiet():
    """Switch the terminal's echo off while a run is in progress, so keys typed meanwhile do
    not land inside the progress line. They still reach the next prompt, and Ctrl-C works."""
    saved = None
    try:
        if sys.stdin.isatty():
            import termios

            fd = sys.stdin.fileno()
            old = termios.tcgetattr(fd)
            termios.tcsetattr(fd, termios.TCSANOW, old[:3] + [old[3] & ~termios.ECHO] + old[4:])
            saved = (termios, fd, old)
    except Exception:
        saved = None
    try:
        yield
    finally:
        if saved is not None:
            saved[0].tcsetattr(saved[1], saved[0].TCSANOW, saved[2])


def _warm() -> None:
    """Import the heavy modules in the background, so the first run starts at once. An import
    interrupted in the main thread would leave torch unusable; here Ctrl-C cannot reach it."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot  # noqa: F401
        import multimolecule  # noqa: F401
        import pandas  # noqa: F401
        import RNA  # noqa: F401
        import torch  # noqa: F401
        import transformers  # noqa: F401

        from . import model  # noqa: F401
        from .features import embed, fold, gather  # noqa: F401
    except BaseException:
        pass  # a missing piece shows up as its own error when a run needs it


# The session --------------------------------------------------------------------------
class Session:
    def __init__(self, args, caps) -> None:
        self.settings = args
        self.caps = caps
        self.scripted = not sys.stdin.isatty()
        self.records: list[tuple[str, str, dict]] = []  # the current input
        self.full: dict = {}  # record name -> (whole-transcript table, (model, context))
        self.shown: list[tuple[str, object]] = []  # (record name, table) on screen
        self.view: Optional[str] = None  # "site825", "1200-1500" or None for everything
        self.label = "modformer"  # names the table of a multi-record input
        self.failed = False
        self.done = False
        self.box: Optional[Box] = None
        self.warm: Optional[threading.Thread] = None

    # -- the loop
    def run(self) -> int:
        with contextlib.nullcontext() if self.scripted else _ctrl_c_handling():
            try:
                return self._loop()
            except KeyboardInterrupt:
                self._say("\n  stopped")
                return 130

    def _loop(self) -> int:
        if not self.scripted:
            self.warm = threading.Thread(target=_warm, daemon=True)
            self.warm.start()
            self._open_input()
            first = self._first_run()
            for sub in ("in", "out"):
                try:
                    (self._work() / sub).mkdir(parents=True, exist_ok=True)
                except OSError:
                    pass  # out/ is made again, with its own error, when a result is written
            self._say("\n" + _fill("Paste a sequence, give a FASTA file or type a gene name.   "
                                   "/demo   /help", self.caps) + "\n"
                      + _fill(f"Files: {self._work()} (put inputs in in/; results go to out/)",
                              self.caps) + "\n")
            if first:
                try:
                    with _interruptible():
                        self.guard(lambda: self.cmd_demo(""))
                except KeyboardInterrupt:
                    self._say("  stopped")
        reading = True
        while not self.done:
            try:  # one try around the whole turn, so a second Ctrl-C has nowhere to escape
                reading = True
                line, pastes = self._read()
                reading = False
                with _interruptible():
                    self.handle(line, pastes)
            except EOFError:
                break
            except KeyboardInterrupt:
                if self.scripted:
                    raise
                self._say("  Ctrl-D or /quit to leave" if reading else "  stopped")
            except LineError as exc:
                self.fail(str(exc))
        if not self.scripted:
            self._say("  bye")
        return 1 if self.scripted and self.failed else 0

    def _open_input(self) -> None:
        """Choose the box when a terminal and the library allow it, the plain reader otherwise."""
        if self.scripted:
            return
        if self.settings.plain or os.environ.get("TERM", "") == "dumb" or not sys.stderr.isatty():
            self._say(_fill("plain mode: some terminals cut a pasted line at 1,024 or "
                            "4,096 characters. For a long sequence give a "
                            "file path, or paste FASTA text (a \">\" line, then lines of "
                            "sequence, then an empty line).", self.caps))
            return
        try:
            self.box = Box()
        except Exception as exc:
            self._say(f"  the input box is not available here ({type(exc).__name__}: {exc}); "
                      "using plain mode")

    def _first_run(self) -> bool:
        """When the model is not on disk yet, ask to download it; True when it was fetched, so
        the demo is scored and the first thing to see is a result. A refusal or a failure is
        shown and is not an error. The wait for the background imports shows a stage line, and
        keys typed during it are dropped when the question is asked (``cli.ask``)."""
        model = cli.MODELS[self.settings.model]
        if not any(cli.missing_downloads(model)):
            return False
        try:
            with _interruptible():
                with term.Progress(sys.stderr, caps=self.caps) as progress, _keys_quiet():
                    self.join_warm(progress, "start up")
                    progress("start up", 2, 2)
                with term.Progress(sys.stderr, caps=self.caps) as progress:
                    cli.setup(model, self.settings, progress)
        except SystemExit as exc:
            self._say(f"  {exc.code}" if isinstance(exc.code, str) else "")
            return False
        except KeyboardInterrupt:
            self._say("  stopped")
            return False
        except Exception as exc:
            self._say("  " + cli.explain(exc, session=True))
            return False
        return True

    def _read(self) -> tuple[str, dict]:
        if self.box is not None:
            try:
                line, pastes = self.box.read(self.status)
                if line.lstrip().startswith(">"):  # FASTA typed or pasted line by line, to an empty line
                    held = {"n": 1}

                    def block_status(width: int) -> str:
                        return (f"  FASTA open, {held['n']} line{'s' * (held['n'] != 1)}: type or "
                                "paste sequence, then an empty line to score it (/ drops it)")

                    def more(n: int) -> Optional[str]:
                        held["n"] = n
                        try:
                            text, chips = self.box.read(block_status)
                        except EOFError:  # Ctrl-D scores the block, as the end of input does in plain mode
                            return None
                        except KeyboardInterrupt:
                            self._say(f"  FASTA block dropped ({n} line{'s' * (n != 1)} not scored)")
                            raise
                        for chip, data in chips.items():
                            text = text.replace(chip, data)
                        return text

                    line = collect_fasta(line, more)
                return line, pastes
            except (EOFError, KeyboardInterrupt):
                raise
            except Exception as exc:
                self.box = None
                self._say(f"  the input box stopped ({type(exc).__name__}: {exc}); "
                          "using plain mode")
        prompt = "" if self.scripted else "modformer> "
        with _interruptible():
            return read_plain(sys.stdin, prompt, tty=not self.scripted), {}

    def _model_label(self) -> str:
        s = self.settings
        return "base" if s.model == "base" else f"ft:{s.context}"

    def status(self, width: Optional[int] = None) -> str:
        """What the line under the box says. Given the box's ``width``, a record name too long
        for the line is shortened so the hint on the right keeps its room."""
        s = self.settings
        model = self._model_label()
        head = f"  {model} · device {s.device or 'auto'} · "
        if not self.records:
            return head + "nothing scored yet"
        if len(self.records) > 1:
            return head + f"{len(self.records)} records"
        view = re.sub(r"^site(\d)", r"site \1", self.view or "")  # the file stem says site825
        tail = f" · {view}" if view else ""
        name = self.records[0][0]
        if width and width >= 60:
            room = width - 1 - len(_status_hint()) - 3 - len(head) - len(tail)
            if 8 <= room < len(name):  # with less room than that the hint cannot show anyway
                name = name[: room - 1] + "…"
        return head + name + tail

    # -- output helpers
    def _say(self, text: str = "") -> None:
        print(text, file=sys.stderr, flush=True)

    def _glyph(self, name: str) -> str:
        return {"prompt": "❯" if self.caps.unicode else ">", "error": "✗" if self.caps.unicode else "x"}[name]

    def fail(self, message: str) -> None:
        self.failed = True
        message = re.sub(r"^modformer: ", "", message)
        lead = f"  {self._glyph('error')} "
        self._say(textwrap.fill(message, max(30, self.caps.width), initial_indent=lead,
                                subsequent_indent=" " * len(lead), break_long_words=False,
                                break_on_hyphens=False))

    def guard(self, fn: Callable[[], None]) -> None:
        """Run ``fn``; every way it can end returns to the prompt."""
        try:
            fn()
        except SystemExit as exc:
            if isinstance(exc.code, str):
                self.fail(exc.code)
            elif exc.code not in (0, None):
                self.failed = True  # argparse printed its own message
        except LineError as exc:
            self.fail(str(exc))
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            if os.environ.get("MODFORMER_DEBUG"):
                raise
            self.fail(cli.explain(exc, session=True))

    # -- one line
    def handle(self, line: str, pastes: Optional[Mapping[str, str]] = None) -> None:
        self.caps = term.detect(sys.stderr, plain=self.settings.plain)
        try:
            kind, payload, extra = split_line(line, pastes)
        except LineError as exc:
            self._echo("command", line.strip(), typed=line)
            self.fail(str(exc))
            return
        if kind == "empty":
            return
        if kind == "command":
            self._echo("command", f"/{payload} {extra}".strip(), typed=line)
            self.guard(lambda: getattr(self, f"cmd_{payload}")(extra))
            return
        shown = "fasta" if kind == "fasta" else cli.classify(payload)
        self._echo(shown, payload, extra if kind == "input" else None, typed=line)
        self.guard(lambda: self.open_input(kind, payload, extra or {}))

    def _echo(self, kind: str, text: str, options: Optional[Mapping[str, object]] = None,
              typed: str = "") -> None:
        """Write the line that stands for an input in the scrollback. With the plain reader on
        a terminal, a line identical to what the terminal already echoed is not written again."""
        lead = f"{'>' if self.scripted else self._glyph('prompt')} "
        shown = echo(kind, text, max(20, self.caps.width - len(lead)), self.caps.unicode, options)
        if not self.scripted and self.box is None and shown == typed.strip():
            return
        self._say(lead + shown)

    # -- inputs
    def open_input(self, kind: str, payload: str, options: dict) -> None:
        args = copy.copy(self.settings)
        for key, value in options.items():
            setattr(args, key, value)
        label = "modformer"
        if kind == "fasta":
            records = [(n, s, {"source": "paste"}) for n, s in cli.parse_fasta(payload, "pasted")]
            label = "pasted"
        else:
            payload = self._locate(payload)
            args.input = payload
            if os.path.isdir(payload):
                raise SystemExit(f"modformer: {payload} is a folder, not a FASTA file")
            records = cli.resolve_input(args, self.caps)
            if cli.classify(payload) == "file":
                label = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(payload).stem)
        # a typo, or a region the sequence does not hold, must not replace the transcript
        cli.check_request(records, args, cli.parse_region(args.region))
        self.records, self.full, self.shown, self.view = records, {}, [], None
        self.label = label
        self.score(records, args)

    def _locate(self, name: str) -> str:
        """A bare file name that is not here is looked for in the work folder's ``in/``."""
        if cli.classify(name) != "gene" or os.sep in name:
            return name  # a path that exists, a sequence, or a path that does not
        found = self._work() / "in" / name
        if found.is_file():
            return str(found)
        if name.lower().endswith(cli._FILE_SUFFIXES):
            for sub in ("in", "out"):
                (self._work() / sub).mkdir(parents=True, exist_ok=True)
            raise SystemExit(f"modformer: no file {name} here or in {self._work() / 'in'}")
        return name

    def _work(self) -> Path:
        return Path(os.environ.get("MODFORMER_WORK") or "modformer-work").expanduser()

    def _out(self, name: str, make: bool = True) -> Path:
        folder = self._work() / "out"
        if make:
            folder.mkdir(parents=True, exist_ok=True)
        return folder / name

    # -- scoring
    def join_warm(self, progress=None, stage: str = "load model") -> None:
        if self.warm is not None and self.warm.is_alive():
            if progress is not None:
                progress(stage, 0, 2)
            self.warm.join()

    def preload(self, model_name: str, args, progress) -> None:
        """Load the networks under a stage of their own, once per model and device."""
        import torch

        from . import _MODELS, load
        from .features import embed

        device = args.device or str(embed.resolve_device())  # a GPU too small for the model gives the CPU
        args.device = self.settings.device = device
        todo = []
        if (str(model_name), str(device)) not in _MODELS:
            todo.append(lambda: load(model_name, device))
        if str(torch.device(device)) not in embed._MODELS:
            todo.append(lambda: embed.load_language_model(device))
        for i, load_one in enumerate(todo):
            progress("load model", i, len(todo))
            load_one()
        if todo:
            progress("load model", len(todo), len(todo))

    def score(self, records, args) -> None:
        """Score ``records`` as the one-shot does, show the result and keep what the next
        command works on. Ctrl-C ends the run and leaves the transcript."""
        self.records = records
        results = []
        try:
            with term.Progress(sys.stderr, caps=self.caps) as progress, _keys_quiet():
                self.join_warm(progress)
                for name, table, _ in cli.score(records, args, self.caps, progress,
                                                      preload=self.preload):
                    progress.close()  # end the stage line, then let the next record animate again
                    progress.__enter__()
                    results.append((name, table))
                    if len(records) > 1:
                        cli.present_line(name, table, self.caps)
                    else:
                        self.card(name, table, args)
        except KeyboardInterrupt:
            if self.scripted:
                raise
            if len(records) > 1:  # the finished records stay on screen for /save, /plot and /copy
                self.shown = results
                self._say(f"  stopped after {len(results)} of {len(records)} records"
                          + (". /save, /plot and /copy use those." if results else "."))
                return
            self._say("  stopped. The transcript is kept: /site and /region work, /all scores "
                      "it in full.")
            return
        finally:
            self._free_gpu()
        self._keep(results, args)

    def _keep(self, results, args) -> None:
        self.shown = results
        key = (self.settings.model, self.settings.context)
        if len(results) == 1:
            name, table = results[0]
            if args.site is None and not args.region:
                self.full[name] = (table, key)
            self.view = (f"site{args.site}" if args.site is not None
                         else args.region.replace(" ", "") if args.region else None)
            return
        self.view = None
        path = self._write(self._tsv(results), self._out(f"{self.label}.modformer.tsv"), self.label)
        self._say(f"\n  table: {path}")

    def _free_gpu(self) -> None:
        torch = sys.modules.get("torch")
        try:
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _listed(self) -> bool:
        return len(self.records) > 1 or self.records[0][2].get("source") == "file"

    def _tsv(self, results):
        """The table to write: one table, or all records with a leading sequence_id column."""
        import pandas as pd

        tables = []
        for name, table in results:
            if self._listed():
                table = table.copy()
                table.insert(0, "sequence_id", name)
            tables.append(table)
        return pd.concat(tables, ignore_index=True) if len(tables) > 1 else tables[0]

    def _written(self, path: Path) -> str:
        return f"{path} (replaced)" if path.exists() else str(path)

    def _write(self, table, path: Path, name: str) -> str:
        shown = self._written(path)
        cli.write_table(table, str(path), name, piped=False)
        return shown

    def _stem(self, name: str) -> str:
        stem = cli.default_table_name(name)[: -len(".modformer.tsv")]
        return stem + (f".{self.view}" if self.view else "")

    def card(self, name: str, table, args, caps=None, out=None, safe: bool = True,
             hint: Optional[str] = cli.SESSION_HINT) -> None:
        """Show ``table`` as the result card, with the transcript for the letters and the
        pairing drawing, and the whole-transcript top rate (from the same model and context) as the
        brightest shade. A drawing that cannot be made is reported in one line and the session
        goes on with its state intact (``safe``); /copy lets the error through instead. ``hint``
        is the line of next commands under the card, or None for no line."""
        seq = next((s for n, s, _ in self.records if n == name), None)
        full = self.full.get(name)
        same = full is not None and len(full[0]) and all(
            full[0].attrs.get(k) == table.attrs.get(k) for k in ("model", "context"))
        scale = float(full[0]["predicted_rate"].max()) if same else None
        if safe:
            if not cli.present_safely(table, name, args, caps or self.caps, out=out, seq=seq,
                                      scale=scale, hint=hint, fail=self.fail):
                self.failed = True
            return
        cli.present(table, name, args, caps or self.caps, out=out, seq=seq, scale=scale,
                    hint=hint)

    # -- views
    def view_of(self, site: Optional[int] = None, region: Optional[str] = None) -> None:
        if not self.records:
            raise SystemExit("modformer: nothing is scored yet. Paste a sequence, give a FASTA "
                             "file or type a gene name first.")
        if len(self.records) > 1:
            raise SystemExit("modformer: narrowing works on one record; open that record alone")
        name, seq, meta = self.records[0]
        args = copy.copy(self.settings)
        args.site, args.region = site, region
        have = self.full.get(name)
        key = (self.settings.model, self.settings.context)
        if have is not None and have[1] == key:
            from .tiling import normalize

            parsed = cli.parse_region(region)
            try:
                table = have[0] if site is None and region is None else narrow(
                    have[0], normalize(seq), site=site, region=parsed)
            except ValueError as exc:
                raise SystemExit(f"modformer: {name}: {exc}")
            self.card(name, table, args)
            self.shown = [(name, table)]
            self.view = (f"site{site}" if site is not None else region.replace(" ", "")
                         if region else None)
            return
        if site is None and region is None and have is None:
            # /all after an input that scored a site or region only, or after a stop: the whole
            # transcript was never scored, so it is scored now, and the line says so
            from .tiling import normalize

            what = (self.view or "").replace("site", "site ") if (self.view or "").startswith(
                "site") else f"region {self.view}"
            part = f"only {what} was scored; " if self.view else ""
            self._say(f"  /all: {part}scoring the whole transcript ({len(normalize(seq)):,} nt).")
        self.score(self.records, args)

    def cmd_site(self, rest: str) -> None:
        self.view_of(site=read_site(rest, "/site takes one position, for example /site 825"))

    def cmd_region(self, rest: str) -> None:
        self.view_of(region=read_region(
            rest, "/region takes two positions, for example /region 1200-1500"))

    def cmd_all(self, rest: str) -> None:
        if len(self.records) > 1:
            self._lines()
        else:
            self.view_of()

    def _lines(self) -> None:
        self._need_table()
        for name, table in self.shown:
            cli.present_line(name, table, self.caps)

    def cmd_top(self, rest: str) -> None:
        """List K sites in the card on screen. It changes that card only: /copy of it copies the
        same K sites, and the next result lists the default again."""
        if not re.fullmatch(r"\d+", rest) or int(rest) < 1:
            raise LineError("/top takes a number of sites, for example /top 5")
        self._need_table()
        if len(self.shown) > 1:
            raise LineError("/top works on one record; open that record alone")
        if (self.view or "").startswith("site"):
            raise LineError("a site view lists no sites; /region or /all lists them")
        name, table = self.shown[0]
        args = copy.copy(self.settings)
        args.top = int(rest)
        self._top_of = (table, args.top)  # /copy of this card copies what is on screen
        self.card(name, table, args)

    def cmd_model(self, rest: str) -> None:
        words = rest.split()
        if not words:
            rows = (("base", "the base model, trained on human data (default)"),
                    ("ft CONTEXT", "the fine-tuned model, with a context (or ft:CONTEXT): "
                     + ", ".join(sorted(CONTEXTS))))
            lines = [textwrap.fill(text, max(40, self.caps.width - 1),
                                   initial_indent=f"  {name:<20} ", subsequent_indent=" " * 23,
                                   break_long_words=False, break_on_hyphens=False)
                     for name, text in rows]
            self._say("\n".join(lines + [f"  now: {self._model_label()}"]))
            return
        if len(words) == 1 and words[0].startswith("ft:"):
            words = ["ft", words[0][3:]]
        if words[0] == "base" and len(words) == 1:
            self.settings.model, self.settings.context = "base", None
        elif words[0] == "ft" and len(words) == 2 and words[1] in CONTEXTS:
            self.settings.model, self.settings.context = "ft", words[1]
        else:
            raise LineError("/model takes base, or ft and one of " + ", ".join(sorted(CONTEXTS)))
        self._say(f"  model: {self._model_label()}. /all, /site and /region use it from here on.")

    def cmd_demo(self, rest: str) -> None:
        records = [cli.demo_record()]
        self.records, self.full, self.shown, self.view = records, {}, [], None
        self.label = "demo"
        self.score(records, copy.copy(self.settings))

    def cmd_help(self, rest: str) -> None:
        width = min(80, max(40, self.caps.width)) - 1
        lines = ["  Commands"]
        for name, (hint, text) in COMMANDS.items():
            lead = f"    /{name:<8} {hint:<20} "
            if width - len(lead) >= 24:  # the text wraps under itself
                wrapped = textwrap.wrap(text, width - len(lead))
                lines.append(lead + wrapped[0])
                lines += [" " * len(lead) + w for w in wrapped[1:]]
            else:  # a narrow terminal: the text goes on the lines below
                lines.append(lead.rstrip())
                lines += ["        " + w for w in textwrap.wrap(text, width - 8)]
        fill = lambda text: textwrap.fill(text, width, initial_indent="  ", subsequent_indent="    ",
                                          break_long_words=False, break_on_hyphens=False)
        lines += ["", fill("Inputs: a sequence of 64 nt or more, a FASTA file, or a gene name. A bare "
                           f"file name not found here is read from {self._work() / 'in'}."),
                  fill("After an input: --species NAME, --isoform ID, --spliced, --region A-B, "
                       "--site N (score only that part)."),
                  fill("Keys: Tab completes a command, Up recalls a line, Ctrl-D leaves."
                       if self.box is not None else "Key: Ctrl-D leaves."), "",
                  fill("score  Resemblance to observed editing sites (0-1)."),
                  fill("rate   Predicted edited fraction (%). Both are predictions, not measurements."),
                  fill("/site draws the predicted fold between a site's anchor and its partner.")]
        self._say("\n".join(lines))

    def cmd_clear(self, rest: str) -> None:
        if sys.stderr.isatty():
            sys.stderr.write("\x1b[H\x1b[2J")
        self._say(term.banner(__version__, caps=self.caps))

    def cmd_quit(self, rest: str) -> None:
        self.done = True

    def cmd_helix(self, rest: str) -> None:
        term.play(term.easter_egg(caps=self.caps), sys.stderr, caps=self.caps)

    # -- files
    def _need_table(self) -> None:
        if not self.shown:
            raise SystemExit("modformer: nothing is scored yet")

    def _target(self, rest: str, default: str) -> Path:
        if rest:
            if rest.strip() == "-":
                raise LineError("give a file name, or nothing for the work folder: a dash is not a file")
            path = Path(rest.strip("'\"")).expanduser()
            work, here = self._work().resolve(), path.resolve()
            if default and here == work:  # the work folder, made when first used, not a file
                return self._out(default)
            if default and here in (work / "in", work / "out"):
                path.mkdir(parents=True, exist_ok=True)
                return path / default
            if not path.resolve().parent.is_dir():
                raise SystemExit(f"modformer: cannot write {rest}: its folder does not exist")
            return path
        return self._out(default)

    def cmd_save(self, rest: str) -> None:
        self._need_table()
        name = self.shown[0][0] if len(self.shown) == 1 else self.label
        stem = self._stem(name) if len(self.shown) == 1 else self.label
        path = self._target(rest, f"{stem}.modformer.tsv")
        self._say(f"  table: {self._write(self._tsv(self.shown), path, name)}")

    def cmd_plot(self, rest: str) -> None:
        self._need_table()
        for name, table in self.shown:
            tag = re.sub(r"[^A-Za-z0-9._-]+", "_", name)[:40]
            if len(self.shown) == 1:
                path = self._target(rest, f"{self._stem(name)}.modformer.png")
            elif rest:  # one figure per record: <name>.<record>.png
                path = self._target(str(Path(rest).expanduser().with_suffix("")) + f".{tag}.png", "")
            else:
                path = self._out(f"{self.label}.{tag}.modformer.png")
            shown = self._written(path)
            cli.plot(table, str(path), name)
            self._say(f"  plot:  {shown}")

    def cmd_copy(self, rest: str) -> None:
        self._need_table()
        tty = sys.stderr if sys.stderr.isatty() and not self.scripted else None
        if rest.strip() == "table":
            table = self._tsv(self.shown)
            text, what, ext = table.to_csv(sep="\t", index=False, float_format="%.6g"), \
                f"the table ({len(table):,} rows)", "tsv"
        elif rest.strip():
            raise LineError("/copy takes nothing, or the word table")
        else:
            buf = io.StringIO()
            caps = term.Caps("none", False, self.caps.unicode, self.caps.width)
            for name, table in self.shown:
                if len(self.shown) > 1:
                    cli.present_line(name, table, caps, out=buf)
                else:
                    args = copy.copy(self.settings)
                    held = getattr(self, "_top_of", None)
                    if held is not None and held[0] is table:
                        args.top = held[1]
                    self.card(name, table, args, caps=caps, out=buf, safe=False, hint=None)
            text = buf.getvalue()
            what, ext = f"the result ({text.count(chr(10))} lines)", "txt"
        name = self.shown[0][0] if len(self.shown) == 1 else self.label
        fallback = self._out(f"{self._stem(name)}.copy.{ext}", make=False)
        self._say("  " + copy_text(text, what, fallback, tty=tty))
