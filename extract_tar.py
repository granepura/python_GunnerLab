#!/usr/bin/env python3
"""Extract one or more tar archives with a combined tqdm progress bar.

The script accepts archive files, directories, or shell-expanded wildcards.
Directories are searched recursively, while unrelated files are ignored. It
supports .tar, .tar.gz, .tgz, .tar.bz2, .tbz2, .tar.xz, and .txz archives.

Examples:
    ./extract_tar.py *
    ./extract_tar.py *.tar.gz
    ./extract_tar.py archive.tar.gz -C /path/to/output
    ./extract_tar.py --help

During extraction, one overall progress bar shows the current archive and how
many archives remain. Unless -C/--directory is supplied, files are extracted
into the current working directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import tarfile
from typing import Iterable

from tqdm import tqdm


TAR_SUFFIXES = (".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz")


def is_tar_path(path: Path) -> bool:
    name = path.name.lower()
    return path.is_file() and any(name.endswith(suffix) for suffix in TAR_SUFFIXES)


def find_archives(inputs: Iterable[str]) -> list[Path]:
    archives: set[Path] = set()
    for raw_path in inputs:
        path = Path(raw_path)
        if is_tar_path(path):
            archives.add(path.resolve())
        elif path.is_dir():
            archives.update(
                candidate.resolve()
                for candidate in path.rglob("*")
                if is_tar_path(candidate)
            )
    return sorted(archives, key=lambda path: str(path))


class ProgressReader:
    """Report compressed bytes read by tarfile to a shared progress bar."""

    def __init__(self, file_object, progress: tqdm) -> None:
        self.file_object = file_object
        self.progress = progress
        self.furthest_position = file_object.tell()

    def _report_position(self) -> None:
        position = self.file_object.tell()
        if position > self.furthest_position:
            self.progress.update(position - self.furthest_position)
            self.furthest_position = position

    def read(self, size: int = -1) -> bytes:
        data = self.file_object.read(size)
        self._report_position()
        return data

    def seek(self, *args):
        new_position = self.file_object.seek(*args)
        self._report_position()
        return new_position

    def tell(self) -> int:
        return self.file_object.tell()

    def __getattr__(self, name):
        return getattr(self.file_object, name)


def extract_archive(archive: Path, destination: Path, progress: tqdm) -> None:
    archive_size = archive.stat().st_size
    progress_at_start = progress.n
    with archive.open("rb") as raw_file:
        tracked_file = ProgressReader(raw_file, progress)
        with tarfile.open(fileobj=tracked_file, mode="r:*") as tar:
            # These trusted archives intentionally contain external symlinks.
            # fully_trusted also permits reruns when a previous interrupted
            # extraction has already created one of those symlinks.
            if hasattr(tarfile, "fully_trusted_filter"):
                tar.extractall(destination, filter="fully_trusted")
            else:
                tar.extractall(destination)

        # Buffered decompression can finish before consuming trailing bytes.
        unreported = archive_size - (progress.n - progress_at_start)
        if unreported > 0:
            progress.update(unreported)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract tar archives with one overall progress bar showing the "
            "current archive and how many remain."
        ),
        epilog=(
            "examples:\n"
            "  ./extract_tar.py *\n"
            "  ./extract_tar.py *.tar.gz\n"
            "  ./extract_tar.py archive.tar.gz -C ./output\n\n"
            "supported: .tar, .tar.gz, .tgz, .tar.bz2, .tbz2, .tar.xz, .txz"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=["."],
        help="archives or directories to scan (shell wildcards such as * are supported)",
    )
    parser.add_argument(
        "-C",
        "--directory",
        type=Path,
        default=Path.cwd(),
        help="destination directory (default: current directory)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    archives = find_archives(args.paths)
    if not archives:
        print("No supported tar archives found.")
        return 1

    destination = args.directory.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    total_bytes = sum(archive.stat().st_size for archive in archives)

    with tqdm(
        total=total_bytes,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc="Extracting",
        dynamic_ncols=True,
    ) as progress:
        archive_count = len(archives)
        for index, archive in enumerate(archives, start=1):
            progress.set_postfix_str(
                f"current={archive.name} | left={archive_count - index}", refresh=True
            )
            extract_archive(archive, destination, progress)

    print(f"Extracted {len(archives)} archive(s) into {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
