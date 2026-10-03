"""Common source copying, replay entry points and reproducibility bundle hashes."""

from pathlib import Path
import shutil

from src.data.common import repo_path, sha256
from src.models.artifacts import source_files


def copy_sources(destination):
    """Copy maintained source and runtime files, excluding generated caches."""
    for name in source_files():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo_path(name), target)


def write_replay_scripts(directory, name, *, module, function):
    """Generate import-safe train/validation programs for one exact setting."""
    prefix = (
        "from pathlib import Path\nimport sys\n\n"
        "ROOT = Path(__file__).resolve().parents[1]\n"
        "sys.path.insert(0, str(ROOT))\n"
        f"from {module} import {function}\n\n"
        "if __name__ == '__main__':\n"
    )
    for suffix, output, arguments in (
        ("train", name, ""), ("test", f"{name}_inner1", ", inner_fold=1")
    ):
        code = prefix + (
            f"    {function}(Path(__file__).with_name('{name}.json'), "
            f"ROOT / 'replays' / '{output}'{arguments})\n"
        )
        (directory / f"{name}_{suffix}.py").write_text(code, encoding="utf-8")


def file_hashes(directory):
    """Manifest actual bundle content, excluding its own manifest and caches."""
    directory = Path(directory)
    return {path.relative_to(directory).as_posix(): sha256(path)
            for path in sorted(directory.rglob("*"))
            if path.is_file() and path != directory / "bundle_manifest.json"
            and "__pycache__" not in path.relative_to(directory).parts
            and path.suffix not in {".pyc", ".pyo"}}
