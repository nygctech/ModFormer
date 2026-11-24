"""Gene name to RNA sequence, through Ensembl.

``list_isoforms`` turns a gene symbol, alias or Ensembl id (gene or transcript) into a table of
transcripts, ``pick_isoform`` chooses one, and ``fetch_rna`` returns its RNA sequence together
with the provenance needed to interpret coordinates (accession with version, assembly, strand,
span).

Two substrates are available for a transcript:

* the intron-inclusive genomic span in transcript orientation (pre-mRNA), the default and
  the substrate the model was trained on;
* the spliced cDNA (``spliced=True``), a different substrate.

Sequences come from the public Ensembl REST API. When `gget <https://github.com/pachterlab/gget>`_
is installed, the pre-mRNA is fetched through ``gget.seq`` and, if that fails or stalls, through
the REST API directly (``MODFORMER_GENE_ROUTE=direct`` or ``gget`` forces one route). Both are
the same Ensembl record. Requests have timeouts and are retried with backoff on rate limits,
server errors and dropped connections.

Sequences are 5' to 3' on the transcript strand, with T written as U. Ensembl already
reverse-complements minus-strand transcripts, so no further reverse complement is applied.

Set ``MODFORMER_OFFLINE=1`` (or pass ``fixture=``) to serve a small recorded response
instead of calling the network. ``MODFORMER_ENSEMBL_URL`` overrides the REST server.
"""

from __future__ import annotations

import http.client
import json
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from importlib import resources
from pathlib import Path
from typing import Any

ENSEMBL_REST = "https://rest.ensembl.org"
FIXTURE_NAME = "gene_fixture.json"
LONG_NT = 100_000  # about 800 tiles, roughly 15 minutes on a 16-core GPU machine; callers warn beyond it
_TIMEOUT = 60.0  # base timeout of a sequence download (see _seq_timeout)
_LOOKUP_TIMEOUT = 30.0  # a lookup or cross-reference answers quickly, so an unreachable server fails sooner
_RETRY = (429, 500, 502, 503, 504)
_sleep, _now = time.sleep, time.monotonic  # module-level so a caller can replace them

# common name -> Ensembl species name; the aliases of each are in the lines below
SPECIES: dict[str, str] = {}
for _line in ("homo_sapiens human hs h_sapiens", "mus_musculus mouse mm house_mouse",
              "rattus_norvegicus rat rn", "danio_rerio zebrafish zf",
              "drosophila_melanogaster fly fruit_fly fruitfly drosophila dm",
              "caenorhabditis_elegans worm nematode c_elegans celegans",
              "gallus_gallus chicken", "sus_scrofa pig", "macaca_mulatta macaque rhesus rhesus_macaque"):
    _latin, *_names = _line.split()
    SPECIES.update({n: _latin for n in (_latin, *_names)})
_EXAMPLES = "human, mouse, rat, zebrafish, fly, worm, chicken, pig, macaque, or an Ensembl name such as bos_taurus"
_ENS_ID = re.compile(r"^(ENS[A-Z]*?)([GTPE])(\d{11})(?:\.(\d+))?$", re.I)
_OTHER_ID = re.compile(r"^(FB(?:gn|tr)\d{7}|WBGene\d{8})$")  # fly and worm gene ids


class GeneError(RuntimeError):
    """A gene or transcript could not be looked up or fetched. The message says what to do."""


class AmbiguousGene(GeneError):
    """A name matches several genes; ``candidates`` is a list of ``(symbol, gene_id)``."""

    def __init__(self, message: str, candidates: list[tuple[str, str]]):
        super().__init__(message)
        self.candidates = candidates


class _NotFound(GeneError):
    """Ensembl answered 400 or 404: the thing asked about does not exist."""


# --------------------------------------------------------------------------- REST


def _server() -> str:
    return os.environ.get("MODFORMER_ENSEMBL_URL", ENSEMBL_REST).rstrip("/")


def _offline() -> bool:
    return os.environ.get("MODFORMER_OFFLINE", "").strip().lower() in {"1", "true", "yes"}


def _retry_after(headers) -> float | None:
    try:
        return min(float(headers.get("Retry-After")), 60.0)
    except (TypeError, ValueError):
        return None


def _rest(path: str, *, text: bool = False, body: Any = None, timeout: float = _LOOKUP_TIMEOUT,
          attempts: int = 3, budget: float = 300.0) -> Any:
    """Ensembl REST call (GET, or POST of ``body``). Retries rate limits, 5xx, timeouts, drops."""
    url, ctype, start = _server() + path, "text/plain" if text else "application/json", _now()
    data = None if body is None else json.dumps(body).encode()
    problem = ""
    for attempt in range(attempts):
        req = urllib.request.Request(url, data=data, headers={
            "Content-Type": "application/json" if data else ctype, "Accept": ctype, "User-Agent": "modformer"})
        wait = None
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode()
            return raw if text else json.loads(raw)
        except urllib.error.HTTPError as e:
            if e.code not in _RETRY:
                try:
                    detail = json.loads(e.read().decode()).get("error", "")
                except Exception:
                    detail = ""
                cls = _NotFound if e.code in (400, 404) else GeneError
                raise cls(detail or f"Ensembl returned HTTP {e.code} for {path.split('?')[0]}") from None
            problem, wait = f"HTTP {e.code}", _retry_after(e.headers)
        except json.JSONDecodeError:
            raise GeneError(f"Ensembl sent an unreadable reply for {path.split('?')[0]}") from None
        except (urllib.error.URLError, http.client.HTTPException, OSError) as e:
            reason = getattr(e, "reason", e)
            problem = "timed out" if isinstance(reason, TimeoutError) else str(reason)
        wait = min(2.0 ** attempt, 20.0) if wait is None else wait
        if attempt + 1 == attempts or _now() - start + wait > budget:
            break
        _sleep(wait)
    raise GeneError(f"Ensembl did not answer ({problem}) after {attempt + 1} tries; retry in a minute. "
                    "MODFORMER_OFFLINE=1 serves only the bundled example gene, PFN1.")


def _call_with_timeout(fn, timeout: float):
    """Run ``fn()`` in a daemon thread so a stalled third-party call cannot hang the caller."""
    box: dict[str, Any] = {}

    def run():
        try:
            box["value"] = fn()
        except BaseException as e:  # reported to the caller below
            box["error"] = e

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        raise GeneError(f"gget did not answer within {timeout:.0f} s")
    if "error" in box:
        raise box["error"]
    return box["value"]


# ------------------------------------------------------------------------ fixture


def _load_fixture(fixture: str | Path | dict | None) -> dict | None:
    """The recorded responses to serve, or ``None`` when live calls should be made."""
    if isinstance(fixture, dict):
        return fixture
    if fixture is not None:
        return json.loads(Path(fixture).read_text())
    if _offline():
        return json.loads(resources.files("modformer.data").joinpath(FIXTURE_NAME).read_text())
    return None


def long_warning(nt: int, spliced_nt: int | None = None, *, spliced: bool = False,
                 part: bool = False) -> str:
    """A sentence for the user when ``nt`` nucleotides make a very long run, else empty.
    ``spliced_nt`` is the spliced transcript's length, named in the advice when known. Set
    ``spliced`` when ``nt`` already counts the spliced transcript and ``part`` when it counts a
    ``--site`` or ``--region`` span; the advice the user has followed is left out. A part of the
    pre-mRNA comes first because it keeps the introns; the spliced transcript does not, and the
    pairing partner of an editing site often lies in one."""
    if nt <= LONG_NT:
        return ""
    advice = []
    if not part:
        advice.append("--site or --region scores one part")
    if not spliced and not part:
        mature = f"the {spliced_nt:,}-nt spliced transcript" if spliced_nt else "the shorter spliced transcript"
        advice.append(f"--spliced scores {mature} instead, without the introns and any pairing "
                      "partners inside them")
    head = f"{nt:,} nt is very long and the run will take a long time"
    return f"{head}; {', and '.join(advice)}" if advice else head


# ---------------------------------------------------------------------- isoforms


def _species_name(species: str) -> str:
    s = re.sub(r"[\s.\-]+", "_", (species or "human").strip().lower()).strip("_")
    return SPECIES.get(s, s)


def _isoform_row(t: dict, gene: dict, canonical_id: str | None) -> dict:
    tid = t["id"].split(".")[0] if t["id"].startswith("ENS") else t["id"]
    mane = next((m.get("type", "").replace("_", " ") for m in t.get("MANE") or []
                 if m.get("type", "").startswith("MANE_Select")), "")
    mane = mane or next((m.get("type", "").replace("_", " ") for m in t.get("MANE") or []), "")
    canonical = bool(t.get("is_canonical")) or tid == canonical_id
    return {
        "transcript_id": tid,
        "accession": f"{tid}.{t['version']}" if t.get("version") else tid,
        "name": t.get("display_name") or tid,
        "biotype": t.get("biotype", ""),
        "length": int(t.get("length", 0)),  # spliced (cDNA) length
        "genomic_length": int(t["end"]) - int(t["start"]) + 1,  # intron-inclusive span
        "canonical": canonical,
        "mane": mane,  # "MANE Select" / "MANE Plus Clinical" (human), else ""
        "mark": ", ".join(m for m in ("canonical" if canonical else "", mane) if m),
        "gene_id": gene["id"],
        "gene": gene.get("display_name") or gene["id"],
        "species": gene.get("species", ""),
        "assembly": gene.get("assembly_name", t.get("assembly_name", "")),
        "chromosome": str(t.get("seq_region_name", gene.get("seq_region_name", ""))),
        "strand": "-" if int(t["strand"]) < 0 else "+",
        "start": int(t["start"]),
        "end": int(t["end"]),
    }


def _rows_from_gene(gene: dict, requested: str | None = None, note: str = "") -> list[dict]:
    """Rows in chooser order: canonical, MANE, protein coding, then longest. Numbers are 0-based."""
    canon = (gene.get("canonical_transcript") or "").split(".")[0] or None
    rows = [_isoform_row(t, gene, canon) for t in gene.get("Transcript", [])]
    rows.sort(key=lambda r: (not r["canonical"], not r["mane"], r["biotype"] != "protein_coding",
                             -r["genomic_length"], r["transcript_id"]))
    for i, r in enumerate(rows):
        r.update(number=i, requested=r["transcript_id"] == requested, resolved_from=note)
    return rows


def _strip_version(ident: str) -> tuple[str, str | None]:
    base, _, ver = ident.partition(".")
    return (base, ver) if ver.isdigit() and (_ENS_ID.match(ident) or _OTHER_ID.match(base)) else (ident, None)


def _gene_json(gene_id: str) -> dict:
    return _rest(f"/lookup/id/{gene_id}?expand=1&mane=1")


def _from_id(ident: str, sp: str) -> list[dict]:
    base, ver = _strip_version(ident.strip())
    base = base.upper() if base.upper().startswith("ENS") else base
    m = _ENS_ID.match(base)
    if m and m.group(2) not in "GT":
        raise GeneError(f"{ident} is {'a protein' if m.group(2) == 'P' else 'an exon'} id; "
                        "give the gene or transcript id, or a gene name")
    try:
        data = _rest(f"/lookup/id/{base}?expand=1&mane=1")
    except _NotFound:
        raise GeneError(f"Ensembl has no record of {ident}. Check that it is a gene or transcript id.") from None
    notes = []
    if ver and data.get("version") and int(ver) != int(data["version"]):
        notes.append(f"{ident} is not the current version; using the current {base}.{data['version']}")
    if data.get("species") and data["species"] != sp and sp:
        notes.append(f"{base} belongs to {data['species'].replace('_', ' ').capitalize()}, so that species is used")
    requested = None
    if data.get("object_type") == "Transcript":
        requested, parent = base, data["Parent"]
        data = _gene_json(parent)
        symbol = data.get("display_name")
        notes.append(f"{base} is a transcript of {symbol} ({parent})" if symbol and symbol != parent
                     else f"{base} is a transcript of {parent}")
    elif data.get("object_type") != "Gene":
        raise GeneError(f"{ident} is a {data.get('object_type')}, not a gene or transcript")
    return _rows_from_gene(data, requested, "; ".join(notes))


def _typos(name: str) -> list[str]:
    """Names one slip away from ``name``: a dropped, swapped or changed character, a hyphen added or removed."""
    n, out = name.upper(), set()
    out |= {n[:i] + n[i + 1:] for i in range(len(n))}
    out |= {n[:i] + n[i + 1] + n[i] + n[i + 2:] for i in range(len(n) - 1)}
    out |= {n[:i] + d + n[i + 1:] for i, c in enumerate(n) if c.isdigit() for d in "0123456789"}
    out |= {n[:i] + "-" + n[i:] for i in range(2, len(n))} | {n.replace("-", "")}
    return sorted(v for v in out if len(v) > 1 and v != n)


def _suggest(name: str, sp: str) -> str:
    """Official symbols among the likely typos of ``name`` (one batch lookup); empty when none or on any failure."""
    try:
        found = _rest(f"/lookup/symbol/{sp}", body={"symbols": _typos(name)}, timeout=15, attempts=1)
    except GeneError:
        return ""
    symbols = sorted({str(g.get("display_name", "")) for g in found.values() if isinstance(g, dict)} - {""})
    return " or ".join(symbols[:3])


def _by_alias(name: str, sp: str) -> list[dict]:
    """Official symbols failed: look the name up as a synonym or previous name."""
    try:
        refs = _rest(f"/xrefs/symbol/{sp}/{urllib.parse.quote(name, safe='')}?object_type=gene")
    except _NotFound:
        refs = []
    ids = list(dict.fromkeys(r["id"] for r in refs if not r["id"].startswith("LRG")))
    genes = []
    if ids:
        found = _rest("/lookup/id?expand=1&mane=1", body={"ids": ids[:20]})
        genes = [found[i] for i in ids[:20] if found.get(i)]
    by_symbol: dict[str, dict] = {}
    for g in sorted(genes, key=lambda g: len(str(g.get("seq_region_name", "")))):  # primary before alt loci
        by_symbol.setdefault(str(g.get("display_name", g["id"])).upper(), g)
    if not by_symbol:
        near = _suggest(name, sp)
        raise GeneError(f"no gene named {name!r} in {sp.replace('_', ' ').capitalize()} (symbols, synonyms and "
                        "previous names were checked). "
                        + (f"Did you mean {near}? Otherwise check" if near else "Check")
                        + " the spelling, or give an Ensembl gene id.")
    if len(by_symbol) > 1:
        cands = [(g.get("display_name", g["id"]), g["id"]) for g in by_symbol.values()]
        raise AmbiguousGene(f"{name!r} is a synonym of several genes: "
                            + ", ".join(f"{s} ({i})" for s, i in cands) + ". Use one of these names or ids.", cands)
    gene = next(iter(by_symbol.values()))
    note = f"{name!r} is not an official symbol; it is a synonym or previous name of {gene.get('display_name')} ({gene['id']})"
    return _rows_from_gene(gene, None, note)


def _by_symbol(name: str, sp: str) -> list[dict]:
    try:
        data = _rest(f"/lookup/symbol/{sp}/{urllib.parse.quote(name, safe='')}?expand=1&mane=1")
    except _NotFound as e:
        if "species" in str(e).lower():
            raise GeneError(f"Ensembl does not know the species {sp!r}. Use one of: {_EXAMPLES}.") from None
        return _by_alias(name, sp)
    return _rows_from_gene(data)


def _offline_rows(gene: str, fx: dict, species: str = "human") -> list[dict]:
    base, _ = _strip_version(gene)
    ids = {fx["gene_id"].upper(), *(t.upper() for t in fx["transcripts"])}
    known = {fx["symbol"].upper(), *ids}
    if base.upper() not in known:
        raise GeneError(f"offline mode only knows the recorded example gene {fx['symbol']} "
                        f"({fx['gene_id']}), not {gene!r}. Unset MODFORMER_OFFLINE to query Ensembl.")
    have = str((fx.get("gene") or {}).get("species") or "")
    if base.upper() not in ids and have and _species_name(species) != have:
        raise GeneError(f"offline mode has only the bundled example, {fx['symbol']} in "
                        f"{have.replace('_', ' ').capitalize()}; it cannot look up {gene!r} in "
                        f"{_species_name(species).replace('_', ' ').capitalize()}. "
                        "Unset MODFORMER_OFFLINE to query Ensembl.")
    requested = base.upper() if base.upper() in {t.upper() for t in fx["transcripts"]} else None
    return _rows_from_gene(fx["gene"], requested)


def list_isoforms(gene: str, species: str = "human", *,
                  fixture: str | Path | dict | None = None) -> list[dict]:
    """Isoform table for a gene symbol, alias, or Ensembl gene or transcript id.

    Each row has ``number`` (0-based position, what ``pick_isoform`` takes), ``transcript_id``,
    ``accession`` (with version), ``name``, ``biotype``, ``length`` (spliced), ``genomic_length``
    (with introns), ``canonical``, ``mane`` and ``mark`` (a display string such as
    ``"canonical, MANE Select"``), plus the gene, assembly, chromosome, strand and span.
    Order: canonical first, then MANE, protein coding, longest. When the input was not an official
    symbol of the species, every row's ``resolved_from`` says how it was resolved (an alias, an old
    id version, an id from another species); ``requested`` marks the transcript when a transcript
    id was given, and ``pick_isoform(rows)`` then returns it.

    Raises ``GeneError`` (``AmbiguousGene`` when an alias names several genes).
    """
    gene = (gene or "").strip()
    if not gene:
        raise GeneError("empty gene name")
    if re.search(r"\s", gene):
        raise GeneError(f"{gene!r} is not a gene name (it contains spaces); give one gene at a time")
    fx = _load_fixture(fixture)
    if fx is not None:
        return _offline_rows(gene, fx, species)
    sp = _species_name(species)
    try:
        rows = _from_id(gene, sp) if (_ENS_ID.match(_strip_version(gene)[0]) or _OTHER_ID.match(gene)) \
            else _by_symbol(gene, sp)
    except (KeyError, TypeError, ValueError) as e:
        raise GeneError(f"unexpected reply from Ensembl for {gene!r} ({type(e).__name__}: {e})") from None
    if not rows:
        raise GeneError(f"{gene!r} ({sp.replace('_', ' ').capitalize()}) has no transcripts in Ensembl")
    return rows


def pick_isoform(isoforms: list[dict], choice: str | int | None = None) -> dict:
    """Choose one isoform row.

    ``None`` gives the requested transcript if the input named one, else the canonical isoform
    (the longest if none is flagged). ``"canonical"``, ``"mane"`` and ``"longest"`` (longest
    genomic span) are words. Otherwise a 0-based row number (as shown by the table), a transcript
    id with or without version, or a transcript name such as ``"ADAR-201"``. When the id names an
    older version of the transcript, the row comes back with ``asked_accession`` set to it, which
    ``fetch_rna`` accepts to report that the current version was used.
    """
    if not isoforms:
        raise GeneError("no isoforms to choose from")
    longest = max(isoforms, key=lambda r: r["genomic_length"])
    canon = next((r for r in isoforms if r["canonical"]), longest)
    if choice is None:
        return next((r for r in isoforms if r.get("requested")), canon)
    if isinstance(choice, int) or str(choice).strip().isdigit():
        i = int(choice)
        if not 0 <= i < len(isoforms):
            raise GeneError(f"isoform number {i} is out of range (0-{len(isoforms) - 1})")
        return isoforms[i]
    key = str(choice).strip()
    if key.lower() == "canonical":
        return canon
    if key.lower() == "longest":
        return longest
    if key.lower() == "mane":
        for r in isoforms:
            if r.get("mane"):
                return r
        raise GeneError("this gene has no MANE transcript; use a number, id or name")
    base, ver = _strip_version(key.upper() if key.upper().startswith("ENS") else key)
    for r in isoforms:
        if base in (r["transcript_id"], r["transcript_id"].upper()) or key.upper() == r["name"].upper():
            current = r["accession"].partition(".")[2]
            if ver and current and ver.isdigit() and current.isdigit() and int(ver) < int(current) \
                    and base == r["transcript_id"]:
                return {**r, "asked_accession": f"{base}.{ver}"}  # pass to fetch_rna for its old-version note
            return r
    shown = ", ".join(f"{r['number']}: {r['name']} ({r['transcript_id']})" for r in isoforms[:12])
    raise GeneError(f"no isoform matches {choice!r}; choices: {shown}" + (" ..." if len(isoforms) > 12 else ""))


# --------------------------------------------------------------------- sequences


def _to_rna(seq: str) -> tuple[str, int]:
    """Uppercase, T to U, any ambiguity code to N. Returns the sequence and how many were mapped."""
    seq = seq.strip().upper().replace("T", "U")
    clean = re.sub(r"[^ACGUN]", "N", seq)
    return clean, sum(a != b for a, b in zip(seq, clean))


def _parse_desc(desc: str) -> dict:
    """``chromosome:GRCh38:17:4848947:4852309:-1`` -> assembly, chromosome, span, strand.
    The assembly name may hold a hyphen (``ARS-UCD2.0``), a dot or an underscore."""
    m = re.search(r"([^:\s]+):([^:\s]+):([^:\s]+):(\d+):(\d+):(-?1)", desc or "")
    if not m:
        raise GeneError(f"unexpected sequence description {desc!r}")
    _, asm, chrom, a, b, strand = m.groups()
    return {"assembly": asm, "chromosome": chrom, "start": int(a), "end": int(b),
            "strand": "-" if strand == "-1" else "+"}


def _seq_timeout(nt: int) -> float:
    return _TIMEOUT + nt / 25_000  # a 2 Mb span takes the service a while


def _direct(tid: str, info: dict) -> tuple[str, str]:
    """Genomic span through the REST API, in transcript orientation."""
    nt = int(info["end"]) - int(info["start"]) + 1
    data = _rest(f"/sequence/id/{tid}?type=genomic", timeout=_seq_timeout(nt), attempts=4)
    return data["seq"], data.get("desc", "")


def _gget(tid: str, info: dict) -> tuple[str, str]:
    import gget
    import gget.gget_seq as gget_seq

    gget_seq.ENSEMBL_REST_API = _server() + "/"  # follow the override; gget defaults to plain http
    nt = int(info["end"]) - int(info["start"]) + 1
    out = _call_with_timeout(lambda: gget.seq(tid, verbose=False), _seq_timeout(nt) * 2)
    if not out or len(out) < 2:
        raise GeneError(f"gget returned no sequence for {tid}")
    return out[1], out[0].lstrip(">").partition(" ")[2]


def _routes() -> list:
    """Genomic-span fetchers in the order to try them."""
    want = os.environ.get("MODFORMER_GENE_ROUTE", "auto").strip().lower()
    if want == "direct":
        return [_direct]
    try:
        import gget  # noqa: F401
        import gget.gget_seq  # noqa: F401
    except ImportError:
        if want == "gget":
            raise GeneError("MODFORMER_GENE_ROUTE=gget but gget is not installed") from None
        return [_direct]
    return [_gget] if want == "gget" else [_gget, _direct]


def _genomic(tid: str, info: dict) -> tuple[str, str, str]:
    """(sequence, description, route) of the intron-inclusive span, in transcript orientation."""
    problems = []
    for route in _routes():
        try:
            return (*route(tid, info), "gget" if route is _gget else "Ensembl REST")
        except GeneError as e:
            problems.append(str(e))
        except Exception as e:  # gget raises whatever it likes
            problems.append(f"{route.__name__[1:]}: {type(e).__name__}: {e}")
    raise GeneError(f"could not fetch the sequence of {tid}: " + "; ".join(problems))


def _transcript_record(tid: str, fx: dict | None) -> dict:
    if fx is not None:
        rec = fx["transcripts"].get(tid)
        if rec is None:
            raise GeneError(f"offline mode has no recorded data for {tid}; recorded: "
                            + ", ".join(fx["transcripts"]))
        return rec
    try:
        return _rest(f"/lookup/id/{tid}")
    except _NotFound:
        raise GeneError(f"Ensembl has no transcript {tid}") from None


def fetch_rna(transcript_id: str, spliced: bool = False, *,
              fixture: str | Path | dict | None = None) -> tuple[str, dict]:
    """RNA sequence of one transcript, 5' to 3' on its own strand, with T as U.

    By default this is the intron-inclusive genomic span (pre-mRNA); ``spliced=True`` returns
    the cDNA instead. Returns ``(rna, meta)``. ``meta`` holds ``transcript_id``, ``accession``
    (with version), ``gene_id``, ``name``, ``species``, ``assembly``, ``chromosome``, ``strand``,
    ``span`` (1-based genomic start and end), ``introns_included``, ``substrate``, ``length``
    (use it to warn about very long runs; see ``LONG_NT``), ``ambiguous_bases_set_to_N`` (how
    many IUPAC ambiguity codes were written as N), ``source`` (``"Ensembl"`` or
    ``"recorded fixture"``), ``route`` and ``note`` (set when a version other than the current one
    was given and the current one was used).
    """
    raw = (transcript_id or "").strip()
    tid, asked_version = _strip_version(raw.upper() if raw.upper().startswith("ENS") else raw)
    m = _ENS_ID.match(tid)
    if not tid or (m and m.group(2) != "T"):
        raise GeneError(f"{transcript_id!r} is not an Ensembl transcript id (expected ENST...)")
    fx = _load_fixture(fixture)
    info = _transcript_record(tid, fx)
    if info.get("object_type", "Transcript") != "Transcript":
        raise GeneError(f"{transcript_id!r} is a {info['object_type']}, not a transcript")
    version = info.get("version")
    note = ""
    if asked_version and version and int(asked_version) != int(version):
        note = f"{raw} is not the current version; using the current {tid}.{version}"

    route = "recorded fixture"
    if fx is not None:
        rec = fx["transcripts"][tid]
        seq = rec["cdna" if spliced else "genomic"]
        desc = None if spliced else rec["genomic_desc"]
    elif spliced:
        seq = _rest(f"/sequence/id/{tid}?type=cdna", text=True, attempts=4,
                    timeout=_seq_timeout(int(info.get("length", 0)))).strip()
        desc, route = None, "Ensembl REST"  # the cDNA has no genomic description; the span comes from the lookup
    else:
        seq, desc, route = _genomic(tid, info)

    span = _parse_desc(desc) if desc else {
        "assembly": info.get("assembly_name", ""), "chromosome": str(info.get("seq_region_name", "")),
        "start": int(info["start"]), "end": int(info["end"]),
        "strand": "-" if int(info["strand"]) < 0 else "+"}
    expect = int(info["length"]) if spliced and info.get("length") else span["end"] - span["start"] + 1
    if (not spliced or info.get("length")) and len(seq) != expect:
        raise GeneError(f"{tid}: Ensembl returned {len(seq):,} nt but records {expect:,}; try again")

    rna, ambiguous = _to_rna(seq)
    meta = {
        "transcript_id": tid,
        "accession": f"{tid}.{version}" if version else tid,
        "gene_id": info.get("Parent", ""),
        "name": info.get("display_name", tid),
        "species": info.get("species", ""),
        "assembly": span["assembly"],
        "chromosome": span["chromosome"],
        "strand": span["strand"],
        "span": (span["start"], span["end"]),
        "introns_included": not spliced,
        "substrate": "spliced cDNA" if spliced else "pre-mRNA (genomic span, introns included)",
        "length": len(rna),
        "ambiguous_bases_set_to_N": ambiguous,
        "source": "recorded fixture" if fx is not None else "Ensembl",
        "route": route,
        "note": note,
    }
    return rna, meta


def record_fixture(gene: str, species: str = "human", path: str | Path | None = None) -> dict:
    """Record a live gene's responses (every isoform) as a fixture for offline use. With ``path``
    the fixture is written there as JSON; ``data/gene_fixture.json`` is a recording of PFN1."""
    isoforms = list_isoforms(gene, species)
    gene_json = _gene_json(isoforms[0]["gene_id"])
    transcripts = {}
    for row in isoforms:
        tid = row["transcript_id"]
        info = _rest(f"/lookup/id/{tid}?expand=1")
        seq, desc, _ = _genomic(tid, info)
        cdna = _rest(f"/sequence/id/{tid}?type=cdna", text=True, timeout=_TIMEOUT, attempts=4).strip()
        info.pop("Translation", None)
        info["Exon"] = [{k: e[k] for k in ("id", "start", "end", "strand")} for e in info.get("Exon", [])]
        transcripts[tid] = {**info, "genomic": seq, "genomic_desc": desc, "cdna": cdna}
    gene_json["Transcript"] = [{k: v for k, v in t.items() if k not in ("Exon", "Translation")}
                               for t in gene_json.get("Transcript", [])]
    fx = {"symbol": gene_json["display_name"], "gene_id": gene_json["id"], "gene": gene_json,
          "transcripts": transcripts}
    if path:
        Path(path).write_text(json.dumps(fx, separators=(",", ":")) + "\n")
    return fx
