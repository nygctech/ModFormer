"""The ``modformer`` command.

    modformer                            open the interactive session
    modformer SEQUENCE                   score every adenosine of a pasted sequence
    modformer transcripts.fa             score every record of a FASTA file
    modformer APP --region 67248-69295   look a gene up, choose an isoform, score part of its pre-mRNA
    modformer APP --site 67993           score only the 64-nt anchor that holds one position
    modformer fetch                      download the model weights ahead of time

The input kind is detected: an existing file is read as FASTA, a string of at least 64
nucleotides is a sequence, anything else is a gene name.
"""
from __future__ import annotations

import argparse
import gzip
import os
import re
import sys
import textwrap
import time
import zlib
from pathlib import Path
from typing import Optional

from . import CONTEXTS, DEFAULT_MODEL, FINETUNED_MODEL, __version__

MODELS = {"base": DEFAULT_MODEL, "ft": FINETUNED_MODEL}
_SEQ_RE = re.compile(r"^[ACGUTNacgutn\s]+$")
_FILE_SUFFIXES = (".fa", ".fasta", ".fna", ".fq", ".fastq", ".gz", ".txt", ".tsv", ".csv")
_PLOT_SUFFIXES = (".png", ".pdf", ".svg")
#: A sequence has at least this many nucleotides; a shorter string is a gene name.
_MIN_SEQUENCE_NT = 64
#: An all-nucleotide input this long cannot be a gene symbol, so it is refused as a short
#: sequence without a lookup.
_SYMBOL_MAX_NT = 16
_ENSEMBL_ID = re.compile(r"^ENS[A-Z]*?[GT]\d{11}(?:\.\d+)?$", re.I)
_MOUSE_STYLE = re.compile(r"^[A-Z][a-z0-9]*[a-z][a-z0-9-]*$")  # App, Adarb1: a mouse symbol's capitals
_EXAMPLES = """examples:
  modformer                                   open the interactive session
  modformer APP --region 67248-69295          gene name: pick an isoform, score part of its pre-mRNA
  modformer my_transcripts.fa -o sites.tsv    every record of a FASTA file
  modformer APP --site 67993                  the 64-nt anchor that holds one position, in seconds
  modformer fetch                             download the weights now
"""


def _err(msg: str) -> None:
    print(msg, file=sys.stderr)


def ask(prompt: str, default: Optional[str] = None, flush: bool = False) -> Optional[str]:
    """Write ``prompt`` to standard error and read one line from standard input.

    Standard output stays clean for a table. Returns the line without its newline, or
    ``default`` at the end of input. On a terminal echo is switched on for the read, because a
    session turns it off while a run is in progress; with ``flush`` keys typed before the
    question was shown are dropped, because they were not an answer to it.
    """
    saved = None
    try:
        if sys.stdin.isatty():
            import termios

            fd = sys.stdin.fileno()
            if flush:  # before the prompt is shown, so an answer to it is never dropped
                termios.tcflush(fd, termios.TCIFLUSH)
            attrs = termios.tcgetattr(fd)
            if not attrs[3] & termios.ECHO:
                saved = (fd, attrs)
                termios.tcsetattr(fd, termios.TCSANOW, attrs[:3] + [attrs[3] | termios.ECHO] + attrs[4:])
    except Exception:
        saved = None
    sys.stderr.write(prompt)
    sys.stderr.flush()
    try:
        line = sys.stdin.readline()
    finally:
        if saved is not None:
            import termios

            termios.tcsetattr(saved[0], termios.TCSANOW, saved[1])
    if line == "":
        sys.stderr.write("\n")
        return default
    return line.rstrip("\r\n")


def _positive(text: str) -> int:
    try:
        n = int(text)
    except ValueError:
        n = 0
    if n < 1:
        raise argparse.ArgumentTypeError("takes a whole number, 1 or more")
    return n


def _device(text: str) -> str:
    if not re.fullmatch(r"cpu|mps|cuda(:\d+)?", text):
        raise argparse.ArgumentTypeError("takes cpu or cuda (cuda:1 for a second GPU)")
    return text


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="modformer", formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Predict A-to-I RNA editing sites and rates from sequence.",
        epilog=_EXAMPLES)
    p.add_argument("input", nargs="?", help="a sequence, a FASTA file, a gene name, "
                   "or one of the words: fetch, demo")
    sel = p.add_mutually_exclusive_group()
    sel.add_argument("--site", type=int, metavar="N",
                     help="score only the 64-nt anchor holding position N (1-based)")
    sel.add_argument("--region", metavar="A-B",
                     help="score only positions A to B (1-based, inclusive)")
    p.add_argument("-o", "--out", metavar="FILE", help="write the table here (TSV); '-' for "
                   "standard output; default: <name>.modformer.tsv in the current folder on a "
                   "terminal, standard output when piped")
    p.add_argument("--top", type=_positive, default=5, metavar="K",
                   help="sites to list (default 5)")
    p.add_argument("--plot", metavar="FILE",
                   help="also save a rate-track figure (.png, .pdf or .svg)")
    p.add_argument("--model", choices=sorted(MODELS),
                   help="base (default) or ft, the fine-tuned model (needs --context)")
    p.add_argument("--context", choices=sorted(CONTEXTS),
                   help="context for --model ft: observed (reference editing "
                   "sites) or a cell type or tissue")
    p.add_argument("--species", default=None, metavar="NAME",
                   help="species for a gene name, for example mouse (default human; an Ensembl "
                        "id carries its own)")
    p.add_argument("--isoform", metavar="ID",
                   help="row number, transcript id or name, 'canonical', 'mane', or 'longest' "
                        "(longest pre-mRNA; longest cDNA with --spliced)")
    p.add_argument("--spliced", action="store_true",
                   help="score the spliced transcript instead of the pre-mRNA")
    p.add_argument("--workers", type=_positive, metavar="N", help="processes used for folding")
    p.add_argument("--device", type=_device, help="cpu or cuda (default: cuda when available)")
    p.add_argument("--session", action="store_true",
                   help="open the interactive session; with input on a pipe it runs the "
                        "commands it reads, one per line")
    p.add_argument("--plain", action="store_true", help="no color, no animation")
    p.add_argument("-y", "--yes", action="store_true",
                   help="never ask: download when needed and take the default isoform")
    p.add_argument("--version", action="version", version=f"modformer {__version__}")
    return p


# Input ---------------------------------------------------------------------------------
def parse_fasta(text: str, default_name: str, source: Optional[str] = None
                ) -> list[tuple[str, str]]:
    """Records of FASTA text as ``(id, sequence)``. Text with no header is one record named
    ``default_name``."""
    records, name, chunks = [], None, []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if name is not None or chunks:
                records.append((name or default_name, "".join(chunks)))
            token = (line[1:].split() or [""])[0]
            # control characters (ESC, BEL, DEL, C1) would reach the terminal when the name is shown
            name = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", token) or f"record{len(records) + 1}"
            chunks = []
        else:
            chunks.append(line)
    if name is not None or chunks:
        records.append((name or default_name, "".join(chunks)))
    if not records:
        raise SystemExit(f"modformer: {source or default_name} holds no sequence")
    return _unique_ids(records)


def _unique_ids(records: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """``records`` with each repeated id renamed ``id_2``, ``id_3``, ... (one note on standard
    error), so every row of the table and every figure name belongs to one record."""
    taken, used, renamed, out = {n for n, _ in records}, set(), {}, []
    for name, seq in records:
        new, k = name, 2
        while new in used:
            while f"{name}_{k}" in taken or f"{name}_{k}" in used:
                k += 1
            new = f"{name}_{k}"
        used.add(new)
        if new != name:
            renamed.setdefault(name, []).append(new)
        out.append((new, seq))
    for name, new in renamed.items():
        _err(f"  id {name} appears more than once; the repeats are named {', '.join(new)}")
    return out


def read_fasta(path: Path) -> list[tuple[str, str]]:
    """Records of a FASTA file as ``(id, sequence)``. A file with no header is one record."""
    data = path.read_bytes()
    try:
        if data[:2] == b"\x1f\x8b":  # gzip magic bytes: a .fa.gz is read as it is
            data = gzip.decompress(data)
        text = data.decode()
    except UnicodeDecodeError:
        raise SystemExit(f"modformer: {path} is not a text file; give a FASTA file") from None
    except (EOFError, gzip.BadGzipFile, zlib.error):
        raise SystemExit(f"modformer: {path} is not a complete gzip file") from None
    return parse_fasta(text, path.stem, str(path))


def classify(text: str) -> str:
    """'file', 'sequence' or 'gene' for a command-line input.

    An existing path, or text with a folder separator in it, is a file. Other input that has 64
    or more non-space characters is a sequence, even with a stray letter in it, so the error
    names that letter instead of looking up a gene. A name ending in a file suffix is never a
    sequence; ``resolve_input`` reports it as a missing file.
    """
    if os.path.exists(text) or os.sep in text:
        return "file"
    if len("".join(text.split())) >= _MIN_SEQUENCE_NT and not text.lower().endswith(_FILE_SUFFIXES):
        return "sequence"
    return "gene"


#: Isoforms listed at most on a narrow terminal, where each takes two lines: with the gene line,
#: a note and the prompt the list stays within 24 rows, so its first line and the default show.
NARROW_ISOFORMS = 7


def isoform_lines(rows: list[dict], caps, limit: int = 12) -> list[str]:
    """The isoform chooser as text: the gene, what an alias resolved to, then one row per isoform
    (the main one first) with its mark, biotype and lengths. Below 79 columns an isoform takes two
    lines and fewer are listed, so the gene line and the default still fit a 24-row screen."""
    width = getattr(caps, "width", 80)
    wide = width >= 79
    limit = limit if wide else min(limit, NARROW_ISOFORMS)
    shown = rows[:limit]
    named = next((r for r in rows if r.get("requested")), None)
    extra = named is not None and named not in shown  # the transcript asked for is always listed
    if extra:
        shown = shown + [named]
    lines = [f"{rows[0].get('gene') or rows[0]['gene_id']}: {len(rows)} isoforms"
             + (f" (first {limit} shown{', and the one you named' if extra else ''})"
                if len(rows) > limit else "")]
    if rows[0].get("resolved_from"):
        lines += wrap(rows[0]["resolved_from"], caps).split("\n")
    if wide:
        lines.append(f"  {'#':>2}  {'transcript':<14} {'mark':<22} {'biotype':<15} "
                     f"{'spliced':>8} {'pre-mRNA':>9}")
    for r in shown:
        name = r["name"] or r["transcript_id"]
        mark = r["mark"]
        if r.get("requested") and not r.get("canonical"):
            mark = "named" + (", " + mark if mark else "")
        if wide:
            lines.append(f"  {r['number']:>2}  {name:<14} {_fit(mark, 23):<22} "
                         f"{_fit(r['biotype'], 16):<15} {r['length']:>8,} {r['genomic_length']:>9,}")
        else:
            lines.append(_fit(f"  {r['number']:>2}  {name}  {mark}", width))
            lines.append(_fit(f"      {r['biotype']}  spliced {r['length']:,}  "
                              f"pre-mRNA {r['genomic_length']:,}", width))
    return lines


def _fit(text: str, width: int) -> str:
    """``text`` cut with an ellipsis so it is never wider than ``width - 1`` columns."""
    return text if len(text) <= width - 1 else text[: max(1, width - 2)] + "…"


def _scored_length(nt: int, args) -> int:
    """How many nucleotides of a ``nt``-long sequence the run scores: the ``--region`` (clipped to
    the sequence), the tile of ``--site``, or all of it. The long-run warning judges this."""
    from .tiling import TILE

    site = getattr(args, "site", None)
    if site is not None:
        return min(nt, TILE)
    try:
        region = parse_region(getattr(args, "region", None))
    except SystemExit:  # a region that cannot be read is refused later, with its own message
        region = None
    if region is not None:
        return min(nt, abs(region[1] - region[0]) + 1)
    return nt


def _pick(rows: list[dict], choice, spliced: bool) -> dict:
    """:func:`pick_isoform`, except that ``longest`` means the longest spliced transcript when
    ``spliced`` is set."""
    from .genes import pick_isoform

    if spliced and isinstance(choice, str) and choice.strip().lower() == "longest":
        return max(rows, key=lambda r: r["length"])
    return pick_isoform(rows, choice)


def _choose_isoform(rows: list[dict], args, caps) -> dict:
    """The isoform to score: the one named by ``--isoform``, one chosen at the prompt, or the
    default (the transcript asked for, else the canonical one). Says what an alias resolved to
    and warns when the run will be very long."""
    from .genes import GeneError, long_warning, pick_isoform

    interactive = (not args.isoform and sys.stdin.isatty() and sys.stderr.isatty() and not args.yes
                   and len(rows) > 1)
    if interactive:
        for line in isoform_lines(rows, caps):
            _err(line)
        _err("  Folding takes a few seconds per tile; tiles fold in parallel, one per core.")
        default = pick_isoform(rows, None)
        label = ("the one you named" if default.get("requested")
                 else "canonical" if default.get("canonical") else "longest pre-mRNA")
        shown = f"{default['number']}: {default['name'] or default['transcript_id']}, {label}"
        prompt = f"  isoform number [{shown}]: "
        flush = True
        while True:  # a mistyped answer asks again; Enter or the end of input takes the default
            answer = (ask(prompt, "", flush=flush) or "").strip()
            flush = False
            try:
                row = _pick(rows, answer or None, args.spliced)
                break
            except GeneError as exc:
                _err(wrap(str(exc), caps))
    else:
        row = _pick(rows, args.isoform or None, args.spliced)
        if rows[0].get("resolved_from"):
            _err(wrap(rows[0]["resolved_from"], caps))
    part = getattr(args, "site", None) is not None or bool(getattr(args, "region", None))
    warning = long_warning(
        _scored_length(row["length"] if args.spliced else row["genomic_length"], args),
        None if args.spliced else row["length"], spliced=args.spliced, part=part)
    if warning:
        _err(wrap(warning, caps))
    return row


def resolve_input(args, caps) -> list[tuple[str, str, dict]]:
    """Turn the command-line input into ``(id, sequence, meta)`` records."""
    kind = classify(args.input)
    if kind == "file":
        if not os.path.exists(args.input):
            raise SystemExit(f"modformer: no such file: {args.input}")
        return [(name, seq, {"source": "file"}) for name, seq in read_fasta(Path(args.input))]
    if kind == "sequence":
        return [("sequence", args.input, {"source": "sequence"})]
    from .genes import GeneError, fetch_rna, list_isoforms

    if args.input.lower().endswith(_FILE_SUFFIXES):
        raise SystemExit(f"modformer: no such file: {args.input}")
    letters = len("".join(args.input.split()))
    shown = args.input if len(args.input) <= 24 else args.input[:21] + "..."
    too_short = f"{shown} is {letters} nt; a sequence needs at least {_MIN_SEQUENCE_NT}."
    if _SEQ_RE.match(args.input) and letters >= _SYMBOL_MAX_NT:
        raise SystemExit(f"modformer: {too_short} It is not a gene name either.")
    # Human is the default for a symbol; an Ensembl id says which species it belongs to.
    species = args.species or (None if _ENSEMBL_ID.match(args.input.strip()) else "human")
    from . import weights

    if not weights.offline():
        _err(f"  looking up {args.input} in Ensembl ...")
    try:
        rows = list_isoforms(args.input, species=species)
        row = _choose_isoform(rows, args, caps)
        seq, meta = fetch_rna(row.get("asked_accession", row["transcript_id"]),
                              spliced=args.spliced)
    except GeneError as exc:
        if _SEQ_RE.match(args.input) and str(exc).startswith("no gene named"):
            raise SystemExit(f"modformer: {too_short} {exc}")
        raise SystemExit(f"modformer: {exc}")
    meta = dict(meta, source="gene", gene=args.input)
    span = meta.get("span")
    where = f"chr{meta.get('chromosome', '?')}" + (f":{span[0]:,}-{span[1]:,}" if span else "")
    _err(wrap(f"{meta.get('accession', row['transcript_id'])}  {meta.get('substrate', '')}  "
              f"{len(seq):,} nt  ({meta.get('species', '').replace('_', ' ').capitalize()} "
              f"{meta.get('assembly', '')} {where} strand {meta.get('strand', '?')})", caps))
    if not args.species and _MOUSE_STYLE.match(args.input.strip()) and meta.get("species") == "homo_sapiens":
        _err(wrap(f"Human {args.input.strip().upper()}; for the mouse gene add --species mouse.", caps))
    if span and meta.get("strand") in ("+", "-"):
        if args.spliced:
            _err(wrap("Positions count along the spliced transcript from its 5' end.", caps))
        else:
            genomic = (f"{span[0]:,} + position - 1" if meta["strand"] == "+"
                       else f"{span[1]:,} - position + 1").replace(" ", "\u00a0")
            _err(wrap(f"Position 1 is the transcript's 5' end; genomic coordinate = {genomic}.",
                      caps))
    if meta.get("note") and meta["note"] not in (rows[0].get("resolved_from") or ""):
        _err(wrap(meta["note"], caps))  # not already said when the isoform was chosen
    return [(f"{args.input}|{meta.get('accession', row['transcript_id'])}", seq, meta)]


# Setup ---------------------------------------------------------------------------------
def _language_model_cached() -> bool:
    """Whether the language model is in the Hugging Face cache. Imports no torch."""
    try:
        from huggingface_hub import try_to_load_from_cache

        from .weights import LM_ID, LM_REVISION

        hit = try_to_load_from_cache(LM_ID, "model.safetensors", revision=LM_REVISION)
        return isinstance(hit, str)
    except Exception:
        return False


def _hf_cache_dir() -> str:
    """The Hugging Face cache folder (``HF_HUB_CACHE``, else under ``HF_HOME``), where the
    language model is kept."""
    from huggingface_hub import constants

    return str(constants.HF_HUB_CACHE)


def missing_downloads(model_name: str) -> tuple[bool, bool]:
    """``(weights missing, language model missing)``, without importing torch."""
    from . import weights

    bundle = weights.bundle_dir(model_name)
    have_weights = all((bundle / f).exists() for f in weights.REGISTRY[model_name]["files"])
    return not have_weights, not _language_model_cached()


def confirm_download() -> bool:
    """Ask whether to download. Only Enter (the default), ``y`` and ``yes`` say yes and only
    ``n`` and ``no`` say no; anything else is asked again with a one-line hint, because a 3 GB
    download must never start on an answer that was meant for something else. The end of input
    is a no."""
    flush = True
    while True:
        answer = (ask("  Download now? [Y/n] ", "n", flush=flush) or "").strip().lower()
        flush = False
        if answer in ("", "y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        _err("  Please answer y or n (Enter downloads).")


def setup(model_name: str, args, progress) -> None:
    """Make sure the weights are present, saying what will be downloaded first."""
    from . import weights

    no_weights, no_lm = missing_downloads(model_name)
    have_weights, have_lm = not no_weights, not no_lm
    if have_weights and have_lm:
        return
    if weights.offline():
        weights.ensure(model_name)  # raises with instructions when something is missing
        if not have_lm:
            raise weights.WeightsError(
                "the RiNALMo language model is not in the Hugging Face cache and "
                "MODFORMER_OFFLINE is set; run `modformer fetch` once with "
                "MODFORMER_OFFLINE unset")
        return
    _err("  ModFormer needs to download, once:")
    if not have_weights:
        _err("    ModFormer weights          0.5 GB")
        _err(f"      kept in {weights.home()}")
    if not have_lm:
        _err("    RiNALMo language model     2.6 GB")
        _err(f"      kept in {_hf_cache_dir()}")
    if sys.stdin.isatty() and not args.yes and not confirm_download():
        key = next((k for k, v in MODELS.items() if v == model_name), "base")
        raise SystemExit(f"nothing downloaded. Run `modformer fetch --model {key}` "
                         "when ready.")
    weights.ensure(model_name, progress=progress)
    if not have_lm:  # the yes covers both downloads, so the language model comes now as well
        if progress is not None:
            progress.close()
        fetch_language_model()


LM_FILES = ["config.json", "model.safetensors", "tokenizer_config.json", "vocab.txt"]


def fetch_language_model() -> None:
    """Download the language model into the Hugging Face cache unless it is already there."""
    from . import weights

    if _language_model_cached():
        return
    if weights.offline():
        raise weights.WeightsError("the RiNALMo language model is not in the Hugging Face "
                                   "cache and MODFORMER_OFFLINE is set; run "
                                   "`modformer fetch` once with MODFORMER_OFFLINE unset")
    from huggingface_hub import snapshot_download

    from .weights import LM_ID as MODEL_ID
    from .weights import LM_REVISION as REVISION

    _err("  Downloading the RiNALMo language model (2.6 GB, once).")
    # Only what loading reads: the repository also holds the same weights a second time as
    # pytorch_model.bin, another 2.6 GB that would never be used.
    snapshot_download(MODEL_ID, revision=REVISION, allow_patterns=LM_FILES)


# Output --------------------------------------------------------------------------------
def wrap(text: str, caps, indent: str = "  ") -> str:
    """Fold ``text`` into lines that fit the terminal, each starting with ``indent``. A short
    parenthetical stays on one line, so a break never falls inside "(Mus musculus GRCm39 ...)"."""
    width = max(20, getattr(caps, "width", 80) - len(indent))
    held = re.sub(r"\([^()]*\)", lambda m: m.group(0).replace(" ", "\u00a0")
                  if len(m.group(0)) <= width else m.group(0), text)
    return "\n".join(indent + line.replace("\u00a0", " ")
                     for line in textwrap.wrap(held, width, break_on_hyphens=False) or [""])


_PUBLIC_CONTEXT = {internal: public for public, internal in CONTEXTS.items()}
SESSION_HINT, ONESHOT_HINT = "/", "--"


def model_token(table) -> str:
    """Short name of the model and context that produced ``table``, read from the table itself:
    ``base``, or ``ft:`` and the context (a result stored under an internal context name shows
    its public one)."""
    model, context = table.attrs.get("model"), table.attrs.get("context")
    if model == DEFAULT_MODEL:
        return "base"
    if model == FINETUNED_MODEL:
        return "ft:" + _PUBLIC_CONTEXT.get(context, context or "?")
    return str(model or "?")


def _view(table, args, seq_len: int):
    """``(site, span)`` of what ``table`` shows: the site asked for (or None) and the 1-based
    inclusive span of a narrowed view (the site's 64-nt anchor, or the region; None for all)."""
    from .tiling import ANCHOR

    site = table.attrs.get("site")
    site = getattr(args, "site", None) if site is None else site
    region = table.attrs.get("region") or parse_region(getattr(args, "region", None))
    if site is not None:
        first = ((site - 1) // ANCHOR) * ANCHOR + 1
        return site, (first, min(seq_len, first + ANCHOR - 1))
    return None, (region[0], min(seq_len, region[1])) if region else None


def site_status(table, args, caps, out, listed: bool = True) -> None:
    """The line for ``--site`` / ``/site``: the asked position's score and rate, or a plain note
    that the position is not an adenosine. With ``listed`` the table that goes to a file or a pipe
    lists the adenosines of its anchor; without it (the card) the note names the best of them,
    the one the track marks and the pairing drawing shows."""
    site, _ = _view(table, args, table.attrs["length"])
    if site is None or table.empty:
        return
    hit = table[table["position"] == site]
    if hit.empty and listed:
        print(wrap(f"position {site} is not an adenosine; the table lists the adenosines of "
                   "its anchor", caps), file=out)
    elif hit.empty:
        best = table.sort_values(["edit_score", "predicted_rate", "position"],
                                 ascending=[False, False, True]).iloc[0]
        print("\n" + wrap(f"position {site} is not an adenosine. The best site of its anchor is "
                          f"{int(best.position)}: editing-site score {best.edit_score:.3f}, predicted rate "
                          f"{best.predicted_rate * 100:.2f}%", caps), file=out)
    else:
        r = hit.iloc[0]
        print("\n" + wrap(f"site {site}: editing-site score {r.edit_score:.3f}, "
                          f"predicted rate {r.predicted_rate * 100:.2f}%", caps), file=out)


def show(table, name: str, seq_len: int, args, caps, out=None, *, seq: Optional[str] = None,
         scale: Optional[float] = None) -> None:
    """The body of a result card: the rate track and the top sites. The site view (``--site``,
    ``/site``) trades the table for the predicted pairing of that site, so the card and the input
    box under it stay within a 24-row terminal; the whole transcript and a region show the table
    only. A narrowed view draws the track over its own span.

    ``seq`` is the transcript, for the letters under a zoomed track and the pairing drawing;
    ``scale`` is the rate that gets the brightest shade (default: the highest rate shown).
    """
    import pandas as pd

    from . import term
    from .tiling import ANCHOR, TILE, TILE_OFFSET, normalize, window_for

    out = sys.stderr if out is None else out
    if table.empty:
        print(f"  {name}: no adenosines scored", file=out)
        return
    # Sites are ranked by editing-site score; the rate is shown beside it, never instead.
    ranked = table.sort_values(["edit_score", "predicted_rate", "position"],
                               ascending=[False, False, True])
    site, span = _view(table, args, seq_len)
    hit = table[table["position"] == site] if site is not None else table.iloc[:0]
    rows = 0 if site is not None else args.top
    best = ranked.head(rows)
    focus = hit.iloc[0] if len(hit) else ranked.iloc[0]
    positions, rates = table["position"].tolist(), table["predicted_rate"].tolist()
    letters = normalize(seq) if seq else None
    marks = [int(focus.position)] if site is not None else ranked["position"].tolist()[:5]
    zoom = {"start": span[0], "length": span[1] - span[0] + 1, "sequence": letters} if span else \
        {"length": seq_len}
    print(term.rate_track(positions, rates, mark=marks, scale=scale, caps=caps, **zoom), file=out)
    if rows:
        print(file=out)
        print(term.top_sites([(int(r.position), r.context, float(r.edit_score),
                               float(r.predicted_rate)) for r in best.itertuples()],
                             n=rows, scale=scale, caps=caps), file=out)
    site_status(table, args, caps, out, listed=False)
    if site is None:
        return
    pos = int(focus.position)
    first = ((pos - 1) // ANCHOR) * ANCHOR + 1
    tile = ((pos - 1) // TILE) * TILE
    partner = None if pd.isna(focus.partner_start) or pd.isna(focus.partner_end) else (
        int(focus.partner_start), int(focus.partner_end))
    print(file=out)
    print(term.duplex(window_for(letters, tile) if letters else None,
                      (table.attrs.get("folds") or {}).get(tile), origin=tile - TILE_OFFSET + 1,
                      anchor=(first, min(seq_len, first + ANCHOR - 1)), partner=partner, focus=pos,
                      length=seq_len, caps=caps), file=out)


def hint_line(table, args, caps, style: str) -> str:
    """The dim line under a card that names what to do next; empty when there is nothing to say."""
    from . import term

    seq_len = table.attrs["length"]
    site, span = _view(table, args, seq_len)
    top = int(table.sort_values(["edit_score", "predicted_rate", "position"],
                                ascending=[False, False, True]).iloc[0]["position"])
    around = f"{max(1, top - 32)}-{min(seq_len, top + 32)}"
    if style == SESSION_HINT:
        text = ("/all   /save   /copy   /plot" if site is not None
                else f"/site {top}   /all   /save   /copy   /plot" if span
                else f"/site {top}   /region {around}   /save   /copy   /plot")
    else:
        out = getattr(args, "out", None)
        text = "" if span else f"--site {top}   --region {around}" + ("" if out else "   -o FILE")
    return term.hint_line(text, caps=caps) if text else ""


def default_table_name(name: str) -> str:
    """File name for a table when none was asked for: the id, shortened to its first fields."""
    stem = "_".join(name.split("|")[:2])
    return re.sub(r"[^A-Za-z0-9._-]+", "_", stem)[:60] + ".modformer.tsv"


def write_table(table, dest: Optional[str], name: str, piped: bool) -> Optional[Path]:
    if dest == "-" or (dest is None and piped):
        table.to_csv(sys.stdout, sep="\t", index=False, float_format="%.6g")
        return None
    path = Path(dest) if dest else Path(default_table_name(name))
    table.to_csv(path, sep="\t", index=False, float_format="%.6g")
    return path


def plot(table, path: str, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 2.4))
    ax.vlines(table["position"], 0, table["predicted_rate"] * 100, color="#e4572e", lw=0.8)
    ax.set_xlabel("position (nt)")
    ax.set_ylabel("predicted rate (%)")
    ax.set_title(title, loc="left", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# Runs ----------------------------------------------------------------------------------
def check_options(args) -> None:
    """Refuse options that cannot work together, and output paths that cannot be written. This
    needs no input, so ``main`` runs it before any gene lookup."""
    if args.model == "ft" and not args.context:
        raise SystemExit("modformer: the fine-tuned model needs --context, one of "
                         + ", ".join(sorted(CONTEXTS)))
    if args.model == "base" and args.context:
        raise SystemExit("modformer: --context goes with --model ft; the base model takes none")
    for opt in ("out", "plot"):  # a quoted or --opt=~/x path reaches us unexpanded
        value = getattr(args, opt)
        if value and value != "-":
            setattr(args, opt, str(Path(value).expanduser()))
    if args.plot and Path(args.plot).suffix.lower() not in _PLOT_SUFFIXES:
        raise SystemExit("modformer: --plot takes a file name ending in .png, .pdf or .svg")
    for target in (args.out if args.out != "-" else None, args.plot):
        if target and not Path(target).expanduser().resolve().parent.is_dir():
            raise SystemExit(f"modformer: cannot write {target}: its folder does not exist")


def check_request(records, args, region) -> None:
    """Refuse a request that cannot succeed before anything is downloaded or loaded."""
    from .tiling import normalize, plan_tiles

    check_options(args)
    for name, seq, _ in records:
        try:
            plan_tiles(len(normalize(seq)), site=args.site, region=region)
        except ValueError as exc:
            raise SystemExit(f"modformer: {name}: {exc}")


def parse_region(text: Optional[str]) -> Optional[tuple[int, int]]:
    """``(first, last)`` from ``A-B``, smaller position first; None for no region."""
    if not text:
        return None
    m = re.match(r"^(\d+)\s*[-:]\s*(\d+)$", text)
    if not m:
        raise SystemExit("modformer: --region takes two positions, for example --region 1200-1500")
    first, last = sorted((int(m.group(1)), int(m.group(2))))  # 900-800 means 800-900
    return first, last


def _warn_long(records, args, caps, progress=None) -> None:
    """The long-run warning for a FASTA or pasted record, as a gene gets it when its isoform is
    chosen. Gene records were warned then, with advice that fits an isoform."""
    from .genes import long_warning
    from .tiling import normalize

    part = getattr(args, "site", None) is not None or bool(getattr(args, "region", None))
    for name, seq, meta in records:
        if meta.get("source") == "gene":
            continue
        warning = long_warning(_scored_length(len(normalize(seq)), args), spliced=True, part=part)
        if warning:
            _say_above(progress, wrap(f"{name}: {warning}", caps))


def _say_above(progress, text: str) -> None:
    """Print a line to standard error without landing on a live status line."""
    lock = getattr(progress, "_lock", None)
    if lock is None or not getattr(progress, "_live", False):
        _err(text)
        return
    with lock:
        print(f"\r\x1b[K{text}", file=sys.stderr)


def score(records, args, caps, progress, preload=None):
    """Score each record in turn. A generator of ``(name, table, seconds)``.

    The request is checked and the weights are set up before the first record is scored.
    ``preload(model_name, args, progress)``, when given, runs after that and before the
    first prediction, so a caller can load the networks under a stage of its own.
    """
    from . import predict

    model_name = MODELS[args.model]
    region = parse_region(args.region)
    check_request(records, args, region)
    _warn_long(records, args, caps, progress)
    try:
        setup(model_name, args, progress)
    except SystemExit as exc:  # the session prints this text as it is; here it gets the prefix
        raise SystemExit(f"modformer: {exc.code}") if isinstance(exc.code, str) else exc
    from transformers.utils import logging as hf_logging

    hf_logging.disable_progress_bar()  # the loader's own bar would fight the status line
    hf_logging.set_verbosity_error()  # and its warnings (deprecations from inside the model) would split it
    if preload is not None:
        preload(model_name, args, progress)
    for name, seq, meta in records:
        started = time.time()
        try:
            table = predict(seq, model=model_name, site=args.site, region=region,
                            context=args.context, device=args.device, workers=args.workers,
                            progress=progress)
        except ValueError as exc:
            raise SystemExit(f"modformer: {name}: {exc}")
        yield name, table, time.time() - started


def present(table, name: str, args, caps, *, card: bool = True, out=None,
            seq: Optional[str] = None, scale: Optional[float] = None,
            hint: Optional[str] = None) -> None:
    """Show one record's result: a header rule that names the model and the view, then, with
    ``card``, the track and the top sites (the pairing drawing in a site view), and a dim line of
    next steps.

    ``seq`` and ``scale`` go to :func:`show`; ``hint`` is ``SESSION_HINT`` or ``ONESHOT_HINT``
    for the commands to name, or None for no line.
    """
    from . import term

    out = sys.stderr if out is None else out
    n = table.attrs["length"]
    site, span = _view(table, args, n)
    where = (f"site {site}" if site is not None
             else f"region {span[0]}-{span[1]}" if span else f"{len(table):,} A")
    print(term.card_header(name, model_token(table), where, caps=caps), file=out)
    if not card:
        site_status(table, args, caps, out)
        return
    show(table, name, n, args, caps, out=out, seq=seq, scale=scale)
    line = hint_line(table, args, caps, hint) if hint and not table.empty else ""
    if line:
        print(file=out)
        print(line, file=out)


def present_safely(table, name: str, args, caps, *, out=None, fail=None, **kw) -> bool:
    """:func:`present`, except that a drawing that cannot be made never costs the result: the
    failure is reported in one plain line (through ``fail`` when given) and False comes back, so
    the caller still writes the table. ``MODFORMER_DEBUG`` lets the error through."""
    try:
        present(table, name, args, caps, out=out, **kw)
        return True
    except Exception as exc:
        if os.environ.get("MODFORMER_DEBUG"):
            raise
        message = f"could not draw the result of {name} ({explain(exc)}); the table is not affected"
        if fail is not None:
            fail(message)
        else:
            _err(wrap(("✗ " if getattr(caps, "unicode", False) else "x ") + message, caps))
        return False


def present_line(name: str, table, caps, out=None) -> None:
    """One short line for a record of a many-record run: name, length, adenosines, best scores."""
    out = sys.stderr if out is None else out
    stats = ("no adenosines" if table.empty else
             f"{len(table):,} adenosines  top score {table['edit_score'].max():.3f}  "
             f"top rate {table['predicted_rate'].max() * 100:.1f}%")
    print(wrap(f"{name}  {table.attrs['length']:,} nt  {stats}", caps), file=out)


def deliver(tables, records, args, piped: bool) -> None:
    """Write the table of every record (and the figures) the command line asked for."""
    import pandas as pd

    full = pd.concat(tables, ignore_index=True) if len(tables) > 1 else tables[0]
    path = write_table(full, args.out, records[0][0] if len(records) == 1 else "modformer", piped)
    if path is not None:
        _err(f"\n  table: {path}")
    if args.plot:
        for k, (name, _, _) in enumerate(records):
            target = args.plot
            if len(records) > 1:  # one figure per record: <plot stem>.<record>.<suffix>
                target = str(Path(args.plot).with_suffix("")) + "." + re.sub(
                    r"[^A-Za-z0-9._-]+", "_", name)[:40] + Path(args.plot).suffix
            plot(tables[k], target, name)
            _err(f"  plot:  {target}")


def run(records, args, caps, progress) -> int:
    piped = not sys.stdout.isatty()
    listed = len(records) > 1 or records[0][2].get("source") == "file"
    tables = []
    for (name, table, _), (_, seq, _) in zip(score(records, args, caps, progress), records):
        if progress is not None and hasattr(progress, "close"):
            progress.close()
        present_safely(table, name, args, caps, card=not (piped and args.out in (None, "-")),
                       seq=seq, hint=ONESHOT_HINT if len(records) == 1 else None)
        if listed:
            table.insert(0, "sequence_id", name)
        tables.append(table)
    deliver(tables, records, args, piped)
    return 0


def explain(exc: BaseException, session: bool = False) -> str:
    """One plain sentence for an error that ends a command; the caller adds any prefix."""
    from .genes import GeneError
    from .weights import WeightsError

    if isinstance(exc, OSError) and exc.strerror:  # no "[Errno 21] ..." for a person
        what = exc.strerror[:1].lower() + exc.strerror[1:]
        return f"{exc.filename}: {what}" if exc.filename else what
    if isinstance(exc, (WeightsError, GeneError, RuntimeError, ValueError, OSError)):
        text = str(exc)
        if "out of memory" in text.lower():
            return ("the GPU has too little free memory. "
                    + ("Start again with `modformer --device cpu` (slower), or wait until it is free."
                       if session else "Run again when it is free, or add --device cpu (slower)."))
        return str(exc)
    return f"{type(exc).__name__}: {exc}"


def demo_record() -> tuple[str, str, dict]:
    from importlib import resources

    text = resources.files("modformer.data").joinpath("demo.fa").read_text()
    lines = text.strip().splitlines()
    return lines[0][1:].split()[0], "".join(lines[1:]), {"source": "demo"}


def guided(args, caps, progress) -> int:
    if args.input is None:
        _err("  No input given, so this is the guided run: set up, then score a demo sequence.\n")
    code = run([demo_record()], args, caps, progress)
    _err("\n  Try next:\n"
         "    modformer APP --region 67248-69295   a gene name (choose an isoform)\n"
         "    modformer my.fa -o sites.tsv         your own sequences\n"
         "    modformer APP --site 67993           one position only\n")
    return code


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    fetch_models = [args.model] if args.model else sorted(MODELS)  # `fetch` takes every model
    args.model = args.model or "base"
    if args.session and (args.input is not None or args.site is not None or args.region
                         or args.out or args.plot):
        parser.error("--session takes no input, --site, --region, -o or --plot: "
                     "give those at its prompt")
    from . import term
    from .genes import GeneError
    from .weights import WeightsError

    caps = term.detect(sys.stderr, plain=args.plain)
    if sys.stderr.isatty():
        _err(term.banner(__version__, caps=caps))
    if args.input == "helix":
        term.play(term.easter_egg(caps=caps), sys.stderr, caps=caps)
        return 0
    # A bare run opens the session only with a person at all three ends of it; anything else
    # (a pipe, cron, a notebook cell) gets the guided demo, which never reads standard input.
    streams = [getattr(f, "isatty", lambda: False)() for f in (sys.stdin, sys.stdout, sys.stderr)]
    if args.session or (args.input is None and all(streams)):
        from .session import Session

        return Session(args, caps).run()
    progress = term.Progress(sys.stderr, caps=caps)
    try:
        with progress:
            if args.input == "fetch":
                from . import weights

                for key in fetch_models:
                    weights.ensure(MODELS[key], progress=progress)
                progress.close()
                fetch_language_model()
                _err(weights.info())
                return 0
            check_options(args)
            if args.input is None or args.input == "demo":
                return guided(args, caps, progress)
            return run(resolve_input(args, caps), args, caps, progress)
    except KeyboardInterrupt:
        _err("\n  stopped")
        return 130
    except BrokenPipeError:  # the reader (head, for one) closed the pipe: stop quietly
        try:
            os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        except (OSError, ValueError, AttributeError):
            pass
        return 0
    except (WeightsError, GeneError, RuntimeError, ValueError, OSError) as exc:
        # One clean line for the user; a traceback helps nobody at a prompt.
        raise SystemExit(f"modformer: {explain(exc)}")
    except Exception as exc:  # anything unforeseen: still one line (MODFORMER_DEBUG=1 shows the trace)
        if os.environ.get("MODFORMER_DEBUG"):
            raise
        raise SystemExit(f"modformer: {explain(exc)}")


if __name__ == "__main__":
    sys.exit(main())
