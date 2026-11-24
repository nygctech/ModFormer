"""Model weight registry and first-run download.

Each released model is a pair of files, ``model.safetensors`` and ``config.json``, kept under
``<home>/weights/<name>/``. The weights are hosted on Zenodo and fetched on first use, resumed if
interrupted, and verified against a SHA-256 checksum before they are moved into place. The config
is small and ships inside the package (``modformer/configs/<name>.json``); it is written next to the
weights after its checksum is verified, with no download. Set ``MODFORMER_OFFLINE=1`` to forbid any
network access and use files you placed in the cache yourself.

Environment variables:

``MODFORMER_HOME``             cache root (default ``~/.cache/modformer``)
``MODFORMER_OFFLINE``          ``1`` disables downloads
``MODFORMER_WEIGHTS_URL``      base URL that replaces the Zenodo file location (for a mirror)
``MODFORMER_DATA_URL``         the same for the figure source archive of the data record
``MODFORMER_ZENODO_TOKEN``     optional access token for reviewer access to a model record that is not public
``MODFORMER_ZENODO_DATA_TOKEN``  the same for the data record

A token switches that record's downloads to its preview address on zenodo.org. It is sent only
there, never to a ``MODFORMER_WEIGHTS_URL`` or ``MODFORMER_DATA_URL`` mirror, and it is removed from
every message this module produces.

The small source tables behind the paper's main figures and Extended Data Figures 1-3 live in the data
record as one archive, ``figure_source_data.zip``. :func:`ensure_figure_data` downloads it once to
``<home>/figure_source_data.zip``, verifies it, and extracts the tables that are asked for into
``<home>/figure-data/``, each checked against its own SHA-256.
"""

from __future__ import annotations

import hashlib
import http.client
import importlib.resources
import os
import re
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterable

#: The public language model the released weights were trained on, at the pinned snapshot.
LM_ID = "multimolecule/rinalmo-giga"
LM_REVISION = "2a71f6f98fb41dd2e6542a5e131d3778111d1468"

MODEL_RECORD = "24300574"
DATA_RECORD = "24300820"
MODEL_DOI = "10.5281/zenodo.24300574"
DATA_DOI = "10.5281/zenodo.24300820"
ZENODO_API = "https://zenodo.org/api/records"
TOKEN_VARS = {MODEL_RECORD: "MODFORMER_ZENODO_TOKEN", DATA_RECORD: "MODFORMER_ZENODO_DATA_TOKEN"}

# Local file name -> published file suffix. Files of a bundle that are not listed here (the
# config) ship inside the package and are never downloaded.
_LOCAL_TO_REMOTE = {"model.safetensors": ".safetensors"}

REGISTRY: dict[str, dict] = {
    "ModFormer-24L512-12ep": {
        "record": MODEL_RECORD,
        "files": {
            "model.safetensors": "4369db6fbef7d6bab2573926f7cfed508c218871a59dfab5705e1af87550c2d2",
            "config.json": "178350f841581f86c6b66e9007caa0b22c9bf95fbe021d5744e3b156302780c7",
        },
    },
    "ModFormer-24L512-12ep-FT1ep": {
        "record": MODEL_RECORD,
        "files": {
            "model.safetensors": "139acc38b599d0ea4fc2a3e753effa9dc4648c3a1c7271a2156aaca2c341d4cb",
            "config.json": "2c56add9ca808fdd1679e58db6676d735a2a00c992c1d9b047cef24bc1a62f04",
        },
    },
}

#: Source tables of the paper's main figures and Extended Data Figures 1-3 (data record): file name -> SHA-256.
FIGURE_DATA: dict[str, str] = {
    "fig2_cells.csv": "733c3c6b11eea45d7b2e102391973b9083ec736f50e555f357a2c15bd6b8e78a",
    "fig3a_recall_curves.csv": "91057176171da8b0a45ca6625533d6c0f386b7b6e7d094695d450f665af8ce12",
    "fig3a_recall_points.csv": "6f75ba4d0905c3c1df4aae4d3b36371a4d12d84c5ccff8502d827aa05bece53a",
    "fig3b_logo_counts.csv": "18fe68ec9ddae34d33a5316c4ba43fed4a9d256641ba0f964dd64b26c632792d",
    "fig3b_logo_meta.csv": "42e344f342e32b7be897efd61af88ddb213edc417c951d0c6512ed51173b07cb",
    "fig3c_paired_edits.csv": "9ac66a0fecafe932399c6af2b59eb1718f09cd63070577ed20cdaf06467b788b",
    "fig4b_budget.csv": "3b1abdb34e2bb1273f2da2c0a8ec1235d661f7256b34122ccf78ffe18e1bd061",
    "fig4c_conditions.csv": "802c163fa0bfa2e33d82923983321f74b78e20e226369df6e0a53e331fe995e2",
    "fig4d_top_pick.csv": "7e3abe2a31a1bd081d65cfdcba4cacdde533883b697908cbda7ecd24eddfb1bb",
    "fig4e_topn.csv": "b4d878716bdde2e1c7e31ccfdfe2b6a027bd4354689384eb66647e6ac41e3321",
    "fig4f_endogenous.csv": "d8bef86b9d54eee7a7b92e3e867338daadc848ac148d2c6bfbdc6ab9d10e7a38",
    "ed1b_auc_by_benchmark.csv": "a7176b7f924df75fc7642caa785e6825a78634f9c5f67a4eba69df30e35e9695",
    "ed1c_auc_summary.csv": "386a7b972e30bb258daf9d180ba4f94bbd4741658b74f1521c49fc329da9bba8",
    "ed1d_rate_spearman.csv": "f980c990c67657e07f0cacf695b1b8003083aba85948f779980951abbfa4ddba",
    "ed2a_datasets.csv": "0da60757e10a5a570cf65df60321989d2ad7a2cd0c52896ff9aa29bfd8db26a0",
    "ed2b_finetune_mix.csv": "fc22e20335a69dd85efb715b14b5b527e452d8a37feeef0d749b894d451e9dc3",
    "ed2cd_pretraining_sets.csv": "5073a9ad10d6869a8a2f5dfc8f45e69377591f52ba8081af0adb749e9b370d94",
    "ed2e_rate_scale.csv": "c962704b0c42d8c92b03a8ce502599eeb55c408085ea6cf68dc810c2667f276b",
    "ed2f_alu_overlap.csv": "40480dec176308b1ef866baac49416656455f4f8ca3dbe425037f48ab208e87d",
    "ed3_top_pick_difference.csv": "397fc4ffc4c86fd465e60306fdbe5670373ddbd74e22cda1544b67442d1d8c69",
    "ed3a_pick_distance.csv": "2554b5873383d5e0a1a216c941b9ea891ca133b46b56ceae395912b4ad652476",
    "ed3b_conditions_pick.csv": "993a202a81968792629f81d6ecec782e4678863b57cd5878d9245883422e86dc",
    "ed3c_distance_bands.csv": "2789c609bd5a2c4c5947539b44414f820ca9069da58910ed54cc5cdbc221ea0a",
    "ed3d_topk.csv": "bfb21bcb6f18d23d62dbeee1885d9784953a3be2069e89f4b4c3e24174d6c392",
    "ed3e_topn_share.csv": "2bd4dd1ad38c07eb3d05e93f02bbf231675bda0993292a3f80a9f867d032c496",
    "ed3f_largest_gains.csv": "1c0e764f3ba5e0a8abbec148e51df3515342942da9863127f92a99c6932534ea",
}

#: The data record holds every figure table in this one archive, with the tables under
#: ``figure_source_data/main/`` (``fig*.csv``) and ``figure_source_data/extended_data/`` (``ed*.csv``).
FIGURE_ZIP = "figure_source_data.zip"
#: SHA-256 of the archive.
FIGURE_ZIP_SHA256 = "477398d20370c22b813e8df448cab15deb96b90648abeccc59e7e8e8af4bf228"
_FIGURE_ZIP_FOLDER = "figure_source_data"
_MAX_TABLE_BYTES = 1 << 30  # a table larger than this is not one of ours; stops a decompression bomb

Progress = Callable[[str, int, int], None]


class WeightsError(RuntimeError):
    """A weight file could not be found, downloaded or verified."""


class NotPublishedError(WeightsError):
    """The remote file does not exist (HTTP 404)."""


def home() -> Path:
    """Cache root: ``$MODFORMER_HOME`` or ``~/.cache/modformer``."""
    env = os.environ.get("MODFORMER_HOME")
    return Path(env).expanduser() if env else Path.home() / ".cache" / "modformer"


def offline() -> bool:
    return os.environ.get("MODFORMER_OFFLINE", "").strip().lower() in {"1", "true", "yes"}


def bundle_dir(name: str) -> Path:
    return home() / "weights" / name


def _token(record: str) -> str:
    """The preview token for ``record`` from the environment, or an empty string."""
    return os.environ.get(TOKEN_VARS[record], "").strip()


def _is_zenodo(url: str) -> bool:
    return urllib.parse.urlsplit(url).netloc == "zenodo.org"


def _is_preview(url: str) -> bool:
    return _is_zenodo(url) and "/draft/files/" in urllib.parse.urlsplit(url).path


def _zenodo_url(record: str, remote_name: str) -> str:
    """Address of one file of a Zenodo record: public, or the preview address when a token is set."""
    quoted = urllib.parse.quote(remote_name)
    token = _token(record)
    if token:
        return f"{ZENODO_API}/{record}/draft/files/{quoted}/content?token={urllib.parse.quote(token, safe='')}"
    return f"{ZENODO_API}/{record}/files/{quoted}/content"


def redact(text) -> str:
    """``text`` with every access token removed: ``token=...`` values and the configured tokens."""
    out = re.sub(r"(?i)(token=)[^&\s'\"<>]+", r"\1***", str(text))
    for record in TOKEN_VARS:
        token = _token(record)
        for form in {token, urllib.parse.quote(token, safe="")} - {""}:
            out = out.replace(form, "***")
    return out


def _token_var(url: str) -> str:
    """Name of the token variable that belongs to the record a Zenodo URL points at."""
    return TOKEN_VARS[DATA_RECORD if f"/records/{DATA_RECORD}/" in url else MODEL_RECORD]


def remote_url(name: str, local_file: str) -> str:
    """Download URL of one file of model ``name``."""
    remote_name = name + _LOCAL_TO_REMOTE[local_file]
    base = os.environ.get("MODFORMER_WEIGHTS_URL")
    if base:
        return f"{base.rstrip('/')}/{remote_name}"
    return _zenodo_url(MODEL_RECORD, remote_name)


def figure_url(name: str) -> str:
    """Download URL of one file of the data record (the figure table archive, ``FIGURE_ZIP``)."""
    base = os.environ.get("MODFORMER_DATA_URL")
    if base:
        return f"{base.rstrip('/')}/{name}"
    return _zenodo_url(DATA_RECORD, name)


def packaged_config(name: str) -> bytes:
    """The ``config.json`` bytes shipped inside the package for model ``name``."""
    resource = importlib.resources.files("modformer") / "configs" / f"{name}.json"
    try:
        return resource.read_bytes()
    except (FileNotFoundError, OSError):
        raise WeightsError(f"this installation has no packaged config for {name}; reinstall modformer") from None


def write_packaged_config(name: str, dest: Path | str) -> Path:
    """Write the packaged config of ``name`` to ``dest`` if it matches its pinned checksum."""
    dest = Path(dest)
    blob = packaged_config(name)
    want = REGISTRY[name]["files"]["config.json"]
    if hashlib.sha256(blob).hexdigest() != want:
        raise WeightsError(f"the packaged config of {name} does not match its checksum; reinstall modformer")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = _private_part(dest)
    try:
        tmp.write_bytes(blob)
        os.replace(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)
    return dest


def _private_part(dest: Path) -> Path:
    """A temporary path next to ``dest`` that no other process or thread shares."""
    return dest.with_name(f"{dest.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.part")


@contextmanager
def _locked(dest: Path):
    """Hold an exclusive lock on ``<dest>.lock`` so concurrent first runs download once.

    The holder deletes the lock file before it lets go, so nothing is left in the cache. A
    process that waited on a lock file that is gone by then opens the new one and waits again.
    """
    try:
        import fcntl
    except ImportError:  # no flock on this platform
        yield
        return
    lock = dest.with_name(dest.name + ".lock")
    while True:
        fh = open(lock, "a")
        try:
            fcntl.flock(fh, fcntl.LOCK_EX)
        except OSError:  # a file system without locks: run unlocked
            fh.close()
            yield
            return
        try:
            current = os.stat(lock).st_ino
        except FileNotFoundError:
            current = None
        if current == os.fstat(fh.fileno()).st_ino:
            break
        fh.close()  # the previous holder removed this file; take the lock on the new one
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)
        fh.close()


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def fetch(url: str, dest: Path | str, sha256: str, progress: Progress | None = None,
          *, timeout: float = 60.0) -> Path:
    """Download ``url`` to ``dest``, resuming a ``.part`` file, and verify the checksum.

    The bytes are written to ``<dest>.part`` and renamed to ``dest`` only after the
    SHA-256 matches. On a mismatch the partial file is deleted and an error is raised.
    ``progress(stage, done_bytes, total_bytes)`` is called as data arrives. Processes that
    fetch the same ``dest`` at once take turns: the first downloads and the others find the
    verified file.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with _locked(dest):
        if dest.exists() and sha256_file(dest) == sha256:
            return dest
        return _download(url, dest, sha256, progress, timeout)


def _download(url: str, dest: Path, sha256: str, progress: Progress | None, timeout: float) -> Path:
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0

    def _open(offset: int):
        req = urllib.request.Request(url, headers={"User-Agent": "modformer"})
        if offset:
            req.add_header("Range", f"bytes={offset}-")
        return urllib.request.urlopen(req, timeout=timeout)

    shown = redact(url)
    try:
        resp = _open(have)
    except urllib.error.HTTPError as e:
        e.close()
        if e.code == 416 and have:  # partial already holds the whole file
            resp = None
        elif e.code == 404:
            hint = (f"If you already have the files, place them under {dest.parent} "
                    f"(as {dest.name}) and set MODFORMER_OFFLINE=1.")
            if _is_preview(url):
                raise NotPublishedError(
                    f"{shown} was not found (HTTP 404). The preview record does not hold this file; "
                    f"check that the token in {_token_var(url)} belongs to this record. {hint}") from None
            if _is_zenodo(url):
                raise NotPublishedError(f"{shown} was not found (HTTP 404) on Zenodo. {hint}") from None
            raise NotPublishedError(f"{shown} was not found (HTTP 404). {hint}") from None
        elif e.code == 403 and _is_preview(url):
            raise WeightsError(
                f"the token in {_token_var(url)} was refused (HTTP 403) for {shown}; it may have "
                "expired or belong to another record.") from None
        else:
            raise WeightsError(f"download of {shown} failed: HTTP {e.code}") from None
    except OSError as e:  # URLError, timeouts and refused connections
        reason = e.reason if isinstance(e, urllib.error.URLError) else e
        raise WeightsError(f"could not reach {shown}: {redact(reason)}") from None

    if resp is not None:
        with resp:
            status = getattr(resp, "status", 200)
            mode = "ab"
            if have and status != 206:  # server ignored Range: start over
                mode, have = "wb", 0
            length = int(resp.headers.get("Content-Length") or 0)
            total = have + length if length else 0
            done = have
            try:
                with open(part, mode) as out:
                    while block := resp.read(1 << 20):
                        out.write(block)
                        done += len(block)
                        if progress:
                            progress(f"download {dest.name}", done, total or done)
            except (OSError, http.client.HTTPException) as e:
                raise WeightsError(
                    f"download of {dest.name} was interrupted ({redact(e)}); the partial file is kept, "
                    "so running again resumes it."
                ) from None
            if total and done < total:  # the server closed the connection early
                raise WeightsError(
                    f"download of {dest.name} stopped at {done} of {total} bytes; the partial "
                    "file is kept, so running again resumes it."
                )

    got = sha256_file(part)
    if got != sha256:
        part.unlink(missing_ok=True)
        raise WeightsError(
            f"checksum mismatch for {dest.name}: expected {sha256}, got {got}. "
            "The partial download was deleted; try again."
        )
    os.replace(part, dest)
    return dest


def _verified_marker(path: Path) -> Path:
    return path.with_name(path.name + ".sha256ok")


def _is_verified(path: Path, sha256: str) -> bool:
    """Checksum a file once; later calls trust a marker keyed to size and mtime."""
    marker = _verified_marker(path)
    st = path.stat()
    stamp = f"{sha256} {st.st_size} {st.st_mtime_ns}"
    try:
        if marker.read_text().strip() == stamp:
            return True
    except OSError:
        pass
    if sha256_file(path) != sha256:
        return False
    try:
        marker.write_text(stamp + "\n")
    except OSError:
        pass
    return True


def ensure(name: str, progress: Progress | None = None) -> Path:
    """Return the directory holding ``model.safetensors`` and ``config.json`` for ``name``.

    Present files are checksum-verified the first time they are seen. A missing weight file is
    downloaded, unless ``MODFORMER_OFFLINE=1``, in which case an error says where to put it. A
    missing config is written from the copy inside the package, which needs no network.
    """
    if name not in REGISTRY:
        raise WeightsError(f"unknown model {name!r}; known models: {', '.join(REGISTRY)}")
    out = bundle_dir(name)
    for local_file, sha in REGISTRY[name]["files"].items():
        path = out / local_file
        if path.exists():
            if _is_verified(path, sha):
                continue
            raise WeightsError(
                f"{path} does not match the published checksum (expected {sha}). "
                "Delete it to download a fresh copy, or replace it with the released file."
            )
        if local_file not in _LOCAL_TO_REMOTE:
            write_packaged_config(name, path)
            _is_verified(path, sha)
            continue
        if offline():
            raise WeightsError(
                f"{local_file} for {name} is not available offline. MODFORMER_OFFLINE is set, so "
                f"nothing is downloaded. Place the released file at {path} "
                f"(files are on Zenodo, doi {MODEL_DOI}, as {name}{_LOCAL_TO_REMOTE[local_file]}), "
                "or unset MODFORMER_OFFLINE."
            )
        fetch(remote_url(name, local_file), path, sha, progress)
        _is_verified(path, sha)
    return out


def figure_data_dir() -> Path:
    return home() / "figure-data"


def figure_member(name: str) -> str:
    """Path inside the archive of figure table ``name``: ``main/`` for ``fig*`` tables, ``extended_data/`` for ``ed*``."""
    if name != Path(name).name or name in {"", ".", ".."}:
        raise WeightsError(f"{name!r} is not a plain table file name")
    if name.startswith("fig"):
        sub = "main"
    elif name.startswith("ed"):
        sub = "extended_data"
    else:
        raise WeightsError(f"cannot tell where figure table {name!r} sits in {FIGURE_ZIP}: "
                           "tables are named fig* (main figures) or ed* (Extended Data)")
    return f"{_FIGURE_ZIP_FOLDER}/{sub}/{name}"


def _extract_tables(archive: Path, names: list[str], out: Path) -> None:
    """Write exactly the tables ``names`` from ``archive`` into ``out``, each checked against FIGURE_DATA.

    Only the member names computed from ``names`` are read: nothing is extracted by a name taken from the
    archive, so a member such as ``../x`` is never written. Each table goes to a private ``.part`` file and is
    renamed once its SHA-256 matches, so two processes extracting the same table do not collide.
    """
    out.mkdir(parents=True, exist_ok=True)
    try:
        zf = zipfile.ZipFile(archive)
    except (zipfile.BadZipFile, OSError) as e:
        raise WeightsError(f"{archive} cannot be read as a zip archive ({e}); delete it to download a "
                           "fresh copy") from None
    with zf:
        for name in names:
            member = figure_member(name)
            try:
                info = zf.getinfo(member)
            except KeyError:
                raise WeightsError(f"{FIGURE_ZIP} holds no {member}; the archive does not match this version "
                                   "of modformer. Delete it and update modformer.") from None
            if info.is_dir() or info.file_size > _MAX_TABLE_BYTES:
                raise WeightsError(f"{member} in {FIGURE_ZIP} is not a table file")
            part = _private_part(out / name)
            h = hashlib.sha256()
            try:
                try:
                    with zf.open(info) as src, open(part, "wb") as dst:
                        while block := src.read(1 << 20):
                            h.update(block)
                            dst.write(block)
                except (zipfile.BadZipFile, OSError, EOFError, RuntimeError) as e:
                    raise WeightsError(f"could not read {member} from {archive} ({e}); delete the archive to "
                                       "download a fresh copy") from None
                if h.hexdigest() != FIGURE_DATA[name]:
                    raise WeightsError(f"{member} in {FIGURE_ZIP} does not match the published checksum of "
                                       f"{name} (expected {FIGURE_DATA[name]}, got {h.hexdigest()}); delete "
                                       "the archive to download a fresh copy")
                os.replace(part, out / name)
            finally:
                part.unlink(missing_ok=True)


def ensure_figure_data(progress: Progress | None = None, names: Iterable[str] | None = None) -> Path:
    """Return the folder holding the figure source tables, fetching any that are missing.

    ``names`` limits the work to those tables (file names such as ``fig3a_recall_curves.csv``);
    by default every table is wanted. A table already in the folder is only checked against its
    published SHA-256; one that fails the check counts as missing. If any wanted table is missing,
    ``figure_source_data.zip`` of the data record is downloaded once to
    ``<home>/figure_source_data.zip`` (and kept, so later tables need no download),
    checked against its pinned SHA-256, and the missing tables are extracted from it, each checked
    again. A cached archive that fails its checksum is downloaded again. With ``MODFORMER_OFFLINE=1``
    nothing is downloaded: a zip already in the cache is used, and otherwise (or when the cached zip
    is damaged) the error says where to put the zip or the tables.
    """
    if isinstance(names, str):
        names = [names]
    wanted = list(FIGURE_DATA) if names is None else list(dict.fromkeys(names))
    unknown = [n for n in wanted if n not in FIGURE_DATA]
    if unknown:
        raise WeightsError(f"unknown figure table {', '.join(map(repr, unknown))}; "
                           f"known tables: {', '.join(FIGURE_DATA)}")
    out = figure_data_dir()
    missing, damaged = [], []
    for name in wanted:
        path = out / name
        # A table that is absent or fails its checksum is rebuilt from the verified archive.
        if not path.is_file():
            missing.append(name)
        elif sha256_file(path) != FIGURE_DATA[name]:
            missing.append(name)
            damaged.append(name)
    if not missing:
        return out
    archive = home() / FIGURE_ZIP
    if archive.is_file() and sha256_file(archive) != FIGURE_ZIP_SHA256:
        if offline():
            raise WeightsError(
                f"{archive} does not match the published checksum (expected {FIGURE_ZIP_SHA256}) and "
                "MODFORMER_OFFLINE is set, so it cannot be downloaded again. Replace it with the "
                f"released {FIGURE_ZIP} of the data record (doi {DATA_DOI}), or unset MODFORMER_OFFLINE.")
        fetch(figure_url(FIGURE_ZIP), archive, FIGURE_ZIP_SHA256, progress)  # replaces the damaged copy
    elif not archive.is_file():
        if offline():
            lead = (f"{out / damaged[0]} does not match its published checksum"
                    if damaged else f"{missing[0]} is not available offline")
            raise WeightsError(
                f"{lead}, and MODFORMER_OFFLINE is set, so nothing is downloaded. "
                f"Place {FIGURE_ZIP} of the data record (doi {DATA_DOI}) at {archive}, or replace the "
                f"figure tables in {out}, or unset MODFORMER_OFFLINE.")
        fetch(figure_url(FIGURE_ZIP), archive, FIGURE_ZIP_SHA256, progress)
    _extract_tables(archive, missing, out)
    return out


def info() -> str:
    """Human-readable summary of the registry and local cache state, for the CLI."""
    lines = [f"cache: {home()}", f"models: doi {MODEL_DOI}   data: doi {DATA_DOI}"]
    for name, entry in REGISTRY.items():
        present = all((bundle_dir(name) / f).exists() for f in entry["files"])
        lines.append(f"  {name}: {'downloaded' if present else 'not downloaded'}")
    return "\n".join(lines)
