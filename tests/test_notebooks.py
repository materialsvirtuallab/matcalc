from __future__ import annotations

from pathlib import Path

from pytest_notebook.nb_regression import NBRegressionFixture

NB_PATH = Path(__file__).parent / ".." / "examples"


def test_all_notebooks() -> None:
    fixture = NBRegressionFixture(exec_timeout=50)
    fixture.diff_color_words = False
    for f in NB_PATH.iterdir():
        if f.is_file():
            fixture.check(str(f.absolute()))
