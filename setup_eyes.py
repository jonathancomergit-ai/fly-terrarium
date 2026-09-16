"""
setup_eyes.py - one-time setup for the Eyes page (webcam -> fly vision).

    python setup_eyes.py

flyvis needs Python 3.9-3.12, so this builds a separate environment in .venv-eye:
  1. installs uv (a fast Python/venv manager from PyPI) for the current user
  2. uv downloads a self-contained Python 3.12 (doesn't touch your system Python or PATH)
  3. creates .venv-eye with CUDA PyTorch + flyvis
  4. downloads the flyvis pretrained models (~10 MB, checksum-verified) into data/flyvis
  5. patches a Windows-only bug in datamate (flyvis's storage library): it deleted an HDF5 file
     that was still open, which Windows refuses ("file is being used by another process")
Safe to run again: every step skips work that's already done.
"""
import os
import subprocess
import sys
import sysconfig
from pathlib import Path

HERE = Path(__file__).parent
VENV = HERE / ".venv-eye"
PY = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
TORCH_INDEX = "https://download.pytorch.org/whl/cu128"   # CUDA 12.8 builds (RTX 50xx needs >= 12.8)


def run(*cmd, **kw):
    print(">", " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def find_uv():
    scheme = "nt_user" if os.name == "nt" else "posix_user"
    uv = Path(sysconfig.get_path("scripts", scheme)) / ("uv.exe" if os.name == "nt" else "uv")
    if not uv.exists():
        run(sys.executable, "-m", "pip", "install", "--user", "uv")
    return uv


def patch_datamate():
    io = VENV / "Lib/site-packages/datamate/io.py"
    if not io.exists():
        io = next(VENV.glob("lib/python3.12/site-packages/datamate/io.py"))
    s = io.read_text(encoding="utf-8")
    if "FlyTerrarium patch" in s:
        print("datamate already patched")
        return
    old = '''    val = np.asarray(val)
    try:
        f = h5.File(path, libver="latest", mode="w")
        if f["data"].dtype != val.dtype:
            raise ValueError()
        f["data"][...] = val
        f.swmr_mode = True
        assert f.swmr_mode
    except Exception:
'''
    new = '''    val = np.asarray(val)
    f = None
    try:
        f = h5.File(path, libver="latest", mode="w")
        if f["data"].dtype != val.dtype:
            raise ValueError()
        f["data"][...] = val
        f.swmr_mode = True
        assert f.swmr_mode
    except Exception:
        # FlyTerrarium patch: close before unlinking - Windows can't delete an open file
        if f is not None:
            try:
                f.close()
            except Exception:
                pass
'''
    if s.count(old) != 1:
        print("WARNING: datamate changed upstream, patch not applied - check datamate/io.py _write_h5")
        return
    io.write_text(s.replace(old, new), encoding="utf-8")
    print("patched datamate/io.py")


def main():
    uv = find_uv()
    if not PY.exists():
        run(uv, "python", "install", "3.12")
        run(uv, "venv", VENV, "--python", "3.12")
    run(uv, "pip", "install", "--python", PY, "torch", "--index-url", TORCH_INDEX)
    run(uv, "pip", "install", "--python", PY, "flyvis")
    patch_datamate()
    env = dict(os.environ, FLYVIS_ROOT_DIR=str(HERE / "data" / "flyvis"), PYTHONIOENCODING="utf-8")
    if not (HERE / "data/flyvis/results/flow/0000/000").exists():
        run(PY, "-m", "flyvis_cli.download_pretrained_models", "--skip_large_files", env=env)
    if not (HERE / "data/eyes.npz").exists():
        run(sys.executable, HERE / "prep_eyes.py")
    run(PY, "-c", "import torch, flyvis; print('eyes ready, CUDA:', torch.cuda.is_available())", env=env)


if __name__ == "__main__":
    main()
