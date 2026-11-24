#!/bin/sh
# ModFormer installer. One command to a working `modformer`:
#
#   curl -LsSf https://raw.githubusercontent.com/nygctech/ModFormer/main/install.sh | sh
#
# What it does, and nothing else:
#   1. finds a Python 3.10 to 3.14 on your machine, or fetches a private one with uv
#   2. creates a private environment in ~/.modformer (no conda, no sudo)
#   3. installs ModFormer with the right PyTorch build for your machine
#   4. links the `modformer` command into ~/.local/bin
#   5. opens `modformer`, a terminal session that downloads the model weights on first use
#
# Settings (all optional, as environment variables):
#   MODFORMER_PREFIX    where the private environment lives   (default ~/.modformer)
#   MODFORMER_BIN       where the command is linked            (default ~/.local/bin)
#   MODFORMER_SOURCE    what to install: a path or URL         (default the GitHub repo)
#   MODFORMER_TORCH     "cpu", "gpu" or "auto"                 (default auto)
#   MODFORMER_NO_START  set to 1 to install without starting
#
# To uninstall, remove these (use your own paths if you set MODFORMER_PREFIX, MODFORMER_BIN,
# MODFORMER_HOME or HF_HOME):
#   rm -rf ~/.modformer ~/.local/bin/modformer ~/.cache/modformer
#   rm -rf ~/.cache/huggingface/hub/models--multimolecule--rinalmo-giga
# The second line removes the RiNALMo download (2.6 GB).
#
# Running it again updates ModFormer and keeps the PyTorch build you have. To switch the
# build, run it again with MODFORMER_TORCH=cpu or MODFORMER_TORCH=gpu.
set -eu

PREFIX="${MODFORMER_PREFIX:-$HOME/.modformer}"
BIN="${MODFORMER_BIN:-$HOME/.local/bin}"
SOURCE="${MODFORMER_SOURCE:-https://github.com/nygctech/ModFormer/archive/refs/heads/main.tar.gz}"
TORCH="${MODFORMER_TORCH:-auto}"

say() { printf '%s\n' "$*"; }
die() { printf 'modformer install: %s\n' "$*" >&2; exit 1; }

# Settings: absolute paths (a relative one would break the link in $BIN), and a known PyTorch build.
case "$PREFIX" in /*) ;; \~/*) PREFIX="$HOME/${PREFIX#\~/}" ;; *) PREFIX="$(pwd)/$PREFIX" ;; esac
case "$BIN" in /*) ;; \~/*) BIN="$HOME/${BIN#\~/}" ;; *) BIN="$(pwd)/$BIN" ;; esac
TORCH="$(printf '%s' "$TORCH" | tr '[:upper:]' '[:lower:]')"
case "$TORCH" in
  auto|cpu|gpu) ;;
  cuda) TORCH="gpu" ;;
  *) die "MODFORMER_TORCH must be cpu, gpu or auto (got '$TORCH')" ;;
esac

# 1. A Python 3.10 to 3.14 with the venv module.
if [ "$(uname -s)-$(uname -m)" = "Darwin-x86_64" ]; then
  die "PyTorch publishes no recent builds for Intel Macs. ModFormer needs an Apple-silicon Mac (run this from a native arm64 shell, not Rosetta) or Linux."
fi
PY=""
NOVENV=""
for candidate in python3.12 python3.11 python3.13 python3.14 python3.10 python3 python; do
  if command -v "$candidate" >/dev/null 2>&1; then
    if "$candidate" -c 'import sys; raise SystemExit(0 if (3, 10) <= sys.version_info[:2] <= (3, 14) else 1)' 2>/dev/null; then
      if "$candidate" -c 'import venv, ensurepip' 2>/dev/null; then
        PY="$candidate"
        break
      fi
      [ -n "$NOVENV" ] || NOVENV="$candidate"
    fi
  fi
done
say "ModFormer installer"
mkdir -p "$PREFIX"
UV="$PREFIX/uv/uv"
if [ -z "$PY" ]; then
  # No usable Python (or one without venv, as on a bare Ubuntu): fetch uv, one program that brings
  # its own Python. Everything stays in $PREFIX, so no sudo, and uninstalling is still one folder.
  [ -n "$NOVENV" ] && say "  python       $NOVENV has no venv module; using a private Python instead"
  if [ ! -x "$UV" ]; then
    say "  installing   uv and Python 3.12 into $PREFIX"
    mkdir -p "$PREFIX/uv"
    url="https://astral.sh/uv/0.13.9/install.sh"
    if command -v curl >/dev/null 2>&1; then
      curl -LsSf "$url" -o "$PREFIX/uv/install.sh" || die "could not download uv from $url"
    elif command -v wget >/dev/null 2>&1; then
      wget -qO "$PREFIX/uv/install.sh" "$url" || die "could not download uv from $url"
    else
      die "needs Python 3.10 to 3.14 with venv, or curl or wget to fetch one"
    fi
    UV_UNMANAGED_INSTALL="$PREFIX/uv" sh "$PREFIX/uv/install.sh" >/dev/null || die "could not install uv"
    [ -x "$UV" ] || die "uv did not install to $UV"
  fi
else
  say "  python       $("$PY" -c 'import sys; print(sys.executable, sys.version.split()[0])')"
fi
say "  environment  $PREFIX/env"

# 2. A private environment.
# An interrupted first run can leave an environment with no pip; --clear rebuilds it. A working
# environment is left alone, so its PyTorch build stays.
if ! "$PREFIX/env/bin/python" -c 'import pip' >/dev/null 2>&1; then
  if [ -n "$PY" ]; then
    "$PY" -m venv --clear "$PREFIX/env"
  else
    UV_PYTHON_INSTALL_DIR="$PREFIX/python" UV_CACHE_DIR="$PREFIX/uv/cache" \
      "$UV" venv --quiet --clear --seed --managed-python --python 3.12 "$PREFIX/env" \
      || die "uv could not create the environment with its own Python 3.12"
  fi
fi
ENVPY="$PREFIX/env/bin/python"
"$ENVPY" -m pip install --quiet --upgrade pip

# 3. PyTorch for this machine, then ModFormer.
# With MODFORMER_TORCH unset (auto), an installed PyTorch stays as it is: a login node with no
# GPU must not replace the GPU build that the compute nodes use from the same home directory.
have="$("$ENVPY" -c 'import torch; print("gpu" if torch.version.cuda else "cpu")' 2>/dev/null || echo none)"
SWAP=0
if [ "$TORCH" = "auto" ]; then
  if [ "$have" != "none" ]; then
    TORCH="$have"
  elif command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    TORCH="gpu"
  else
    TORCH="cpu"
  fi
elif [ "$have" != "none" ] && [ "$have" != "$TORCH" ] && [ "$(uname -s)" = "Linux" ]; then
  # pip keeps an installed torch, so a changed MODFORMER_TORCH has to force the new build.
  # (Linux only: macOS has one build.)
  SWAP=1
fi
say "  pytorch      $TORCH build"
if [ "$TORCH" = "gpu" ] && [ "$(uname -s)" = "Linux" ]; then
  say "  installing   PyTorch (a few minutes the first time; the GPU build is several GB)"
else
  say "  installing   PyTorch (a few minutes the first time)"
fi
TORCH_INDEX=""
if [ "$TORCH" = "cpu" ] && [ "$(uname -s)" = "Linux" ]; then
  # The default Linux wheel pulls in CUDA libraries (about 4 GB) that a CPU machine never uses.
  TORCH_INDEX="--index-url https://download.pytorch.org/whl/cpu"
fi
# pip downloads before it removes anything, so a failed download leaves the old build in place.
# shellcheck disable=SC2086
if [ "$SWAP" = "1" ]; then
  "$ENVPY" -m pip install --quiet --force-reinstall --no-deps torch $TORCH_INDEX
fi
# shellcheck disable=SC2086
"$ENVPY" -m pip install --quiet torch $TORCH_INDEX
say "  installing   ModFormer"
# Dependencies first. An https archive at the same version counts as installed to pip, so the
# second command replaces the code. If a download fails, the previous install keeps working.
"$ENVPY" -m pip install --quiet "$SOURCE"
"$ENVPY" -m pip install --quiet --force-reinstall --no-deps "$SOURCE"

# 4. The command.
mkdir -p "$BIN"
ln -sf "$PREFIX/env/bin/modformer" "$BIN/modformer"
say "  command      $BIN/modformer"
case ":$PATH:" in
  *":$BIN:"*) ;;
  *) say ""
     say "  $BIN is not on your PATH. Add this line to your shell profile:"
     say "    export PATH=\"$BIN:\$PATH\"" ;;
esac

# 5. Start.
say ""
if [ "${MODFORMER_NO_START:-0}" = "1" ]; then
  say "Installed. Run:  modformer"
  exit 0
fi
if (exec </dev/tty) 2>/dev/null; then
  # Piped installs have no keyboard on stdin; take it from the terminal. (The file can
  # exist and still not open, for example under cron or in a container with no terminal.)
  exec "$BIN/modformer" </dev/tty
else
  say "Installed. Run:  modformer"
fi
