"""Runs the standalone paper experiment."""

import importlib.util
import os
from pathlib import Path
import sys


def configure_cuda_libraries():
    """Restarts with the active environment's cuDNN libraries on the loader path."""
    if not sys.platform.startswith("linux") or importlib.util.find_spec("nvidia") is None:
        return
    spec = importlib.util.find_spec("nvidia.cudnn")
    if spec is None or spec.submodule_search_locations is None:
        return
    directories = [
        str(Path(location) / "lib")
        for location in spec.submodule_search_locations
        if (Path(location) / "lib").is_dir()
    ]
    if not directories:
        return
    prefix = os.pathsep.join(directories)
    current = os.environ.get("LD_LIBRARY_PATH", "")
    if current == prefix or current.startswith(prefix + os.pathsep):
        return
    environment = os.environ.copy()
    environment["LD_LIBRARY_PATH"] = prefix + (os.pathsep + current if current else "")
    print(f"[runtime] Using cuDNN libraries from {prefix}", flush=True)
    # The dynamic loader reads this path at process startup, before importing torch.
    os.execvpe(sys.executable, [sys.executable, *sys.argv], environment)


if __name__ == "__main__":
    configure_cuda_libraries()
    from slm_grpo.experiment import main

    main()
