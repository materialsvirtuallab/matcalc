from __future__ import annotations

import os
from collections import namedtuple
from typing import TYPE_CHECKING

import pytest
from matcalc.cli import calculate_property
from monty.tempfile import ScratchDir

if TYPE_CHECKING:
    from pymatgen.core import Structure


def test_calculate_property(LiFePO4: Structure) -> None:
    args = namedtuple("args", ["model", "property", "structure", "outfile"])  # noqa: PYI024

    with ScratchDir(".") as _:
        LiFePO4.to(filename="cli_test.cif")
        a = args("M3GNet", "ElasticityCalc", ["cli_test.cif"], "CLI.json")
        calculate_property(a)

        assert os.path.exists(a.outfile)
        assert os.path.exists("cli_test.cif")

        a = args("M3GNet", "BadCalc", ["cli_test.cif"], "CLI.json")
        with pytest.raises(KeyError):
            calculate_property(a)
