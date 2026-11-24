"""Execute both clean notebooks using this interpreter, writing ignored outputs.

Run from the repository root: python scripts/check_notebooks.py
"""
import json
from pathlib import Path
import sys
import tempfile
import nbformat
from nbclient import NotebookClient
from jupyter_client import AsyncKernelManager
from jupyter_client.kernelspec import KernelSpecManager


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "outputs" / "notebook_checks"
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="qrt-kernel-") as tmp:
        directory = Path(tmp) / "qrt-local"
        directory.mkdir()
        (directory / "kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "QRT local test interpreter", "language": "python",
        }))
        for path in sorted((root / "notebooks").glob("*.ipynb")):
            notebook = nbformat.read(path, as_version=4)
            nbformat.validate(notebook)
            manager = AsyncKernelManager(kernel_name="qrt-local", kernel_spec_manager=KernelSpecManager(kernel_dirs=[tmp]))
            client = NotebookClient(notebook, km=manager, timeout=180, resources={"metadata": {"path": str(root / "notebooks")}})
            client.on_cell_start = lambda cell, cell_index: print(f"  cell {cell_index + 1}", flush=True)
            print(f"Executing {path.name}", flush=True)
            client.execute(cleanup_kc=True)
            nbformat.write(notebook, output / (path.stem + "_executed.ipynb"))
            print(f"PASS {path.name}", flush=True)


if __name__ == "__main__":
    main()
