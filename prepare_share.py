"""Copy a complete project into a new folder suitable for GitHub browser upload."""

import argparse
from datetime import datetime
import os
from pathlib import Path
import shutil
import stat


SOURCE_FILES = (
    "main.py", "README.md", "README.txt", "requirements.txt",
    "requirements-training.txt", "environment-windows.txt", "verify_setup.py",
    "prepare_share.py", "PUBLISHING.md", ".gitignore", ".gitattributes",
)
ASSET_DIRS = ("saved_models", "datasets", "data")
LANGUAGES = ("en", "ar", "multi")
MODEL_FILES = ("tfidf.pkl", "svm.pkl", "nb.pkl", "tokenizer.pkl", "meta.pkl", "lstm.keras")
SKIP_DIRS = {
    ".git", "venv", "env", "node_modules", "__pycache__", ".cache",
    ".ipynb_checkpoints", ".pytest_cache", ".mypy_cache", ".ruff_cache",
}
MAX_BROWSER_BYTES = 25 * 1024 * 1024
LINK_TAGS = {0xA0000003, 0xA000000C}  # Windows junction and symbolic-link tags.


def _is_link(path):
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False
    return stat.S_ISLNK(info.st_mode) or getattr(info, "st_reparse_tag", 0) in LINK_TAGS


def _regular_info(path):
    try:
        info = path.lstat()
    except FileNotFoundError:
        return None
    if stat.S_ISREG(info.st_mode) and not _is_link(path):
        return info
    return None


def _asset_files(folder):
    if _is_link(folder) or not folder.is_dir():
        return
    def raise_walk_error(error):
        raise error

    for directory, dirs, files in os.walk(folder, followlinks=False, onerror=raise_walk_error):
        directory = Path(directory)
        dirs[:] = sorted(
            name for name in dirs
            if name.lower() not in SKIP_DIRS
            and not name.lower().startswith(".venv")
            and not _is_link(directory / name)
        )
        for name in sorted(files):
            path = directory / name
            if _regular_info(path) is not None:
                yield path


def build_export(project_root, output=None):
    """Return output Path, file_count, and total_bytes in a result dictionary."""
    root = Path(project_root).expanduser()
    if _is_link(root):
        raise ValueError("The project root must be an ordinary directory, not a link.")
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"Project folder is not a directory: {root}")

    target = Path(output).expanduser() if output is not None else root.parent / (
        "bot-comment-detector-github-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    if os.path.lexists(target):
        raise ValueError(f"Export destination already exists; choose a new path: {target}")
    target = target.resolve(strict=False)
    if target == root or root in target.parents:
        raise ValueError("Export destination must be outside the original project folder.")

    invalid_source = []
    for name in SOURCE_FILES:
        info = _regular_info(root / name)
        if info is None or info.st_size == 0:
            invalid_source.append(name)
    if invalid_source:
        raise ValueError(
            "Missing, empty, or linked package files:\n  " + "\n  ".join(invalid_source)
            + "\nMerge the complete source update into the original project first."
        )
    invalid_models = []
    for language in LANGUAGES:
        folder = root / "saved_models" / language
        linked_folder = _is_link(root / "saved_models") or _is_link(folder)
        for name in MODEL_FILES:
            path = folder / name
            info = None if linked_folder else _regular_info(path)
            if info is None or info.st_size == 0:
                invalid_models.append(str(path.relative_to(root)))
    if invalid_models:
        raise ValueError(
            "Missing, empty, or linked model artifacts:\n  "
            + "\n  ".join(invalid_models)
            + "\nMerge the source update into the original project with all model files first."
        )

    selected = [root / name for name in SOURCE_FILES]
    for name in ASSET_DIRS:
        selected.extend(_asset_files(root / name))
    sizes = {path: path.stat().st_size for path in selected}
    oversized = [path for path, size in sizes.items() if size > MAX_BROWSER_BYTES]
    if oversized:
        details = "\n  ".join(
            f"{path.relative_to(root)} ({sizes[path] / 1024**2:.2f} MiB)"
            for path in oversized
        )
        raise ValueError(
            "Files exceed the 25 MiB browser-upload limit:\n  " + details
            + "\nUse Git for larger files, or Git LFS where required, instead of this browser export."
        )

    target.mkdir(parents=True, exist_ok=False)
    try:
        for path in selected:
            if _regular_info(path) is None:
                raise ValueError(f"Source changed during export: {path}")
            destination = target / path.relative_to(root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination, follow_symlinks=False)
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            f"An incomplete export remains at {target}. Original files were not changed. {exc}"
        ) from exc
    return {"output": target, "file_count": len(selected), "total_bytes": sum(sizes.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="A new export folder outside the project")
    args = parser.parse_args()
    try:
        result = build_export(Path(__file__).resolve().parent, args.output)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"Export failed: {exc}\n")
    print(f"Export folder: {result['output']}")
    print(f"Copied {result['file_count']} files ({result['total_bytes'] / 1024**2:.2f} MiB).")
    print(f'Open in File Explorer: explorer.exe "{result["output"]}"')
    print("Upload the CONTENTS of this export folder so README.md appears at the repository root.")
    print("The original project is unchanged. Nothing has been uploaded to GitHub.")


if __name__ == "__main__":
    main()
