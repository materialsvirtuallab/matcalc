"""Sets some configuration global variables and locations for matcalc."""

from __future__ import annotations

import pathlib
import shutil

BENCHMARK_DATA_URL = "https://api.github.com/repos/materialsvirtuallab/matcalc/contents/benchmark_data"
BENCHMARK_DATA_DOWNLOAD_URL = "https://raw.githubusercontent.com/materialsvirtuallab/matcalc/main/benchmark_data"
BENCHMARK_DATA_DIR = pathlib.Path.home() / ".cache" / "matcalc"


def clear_cache(*, confirm: bool = True) -> None:
    """Deletes all files in the mattcalc cache. This is used to clean out downloaded benchmarks.

    Args:
        confirm: Whether to ask for confirmation. Default is True.
    """
    answer = "" if confirm else "y"
    while answer not in ("y", "n"):
        answer = input(f"Do you really want to delete everything in {BENCHMARK_DATA_DIR} (y|n)? ").lower().strip()
    if answer == "y":
        try:
            shutil.rmtree(BENCHMARK_DATA_DIR)
        except FileNotFoundError:
            print(f"matcalc cache dir {BENCHMARK_DATA_DIR} not found")  # noqa: T201
