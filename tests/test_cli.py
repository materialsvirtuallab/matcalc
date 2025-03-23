from __future__ import annotations

import os
import subprocess
from collections import namedtuple
from importlib.util import find_spec
from typing import TYPE_CHECKING

import pytest
from monty.tempfile import ScratchDir

from matcalc.cli import calculate_property

if TYPE_CHECKING:
    from pymatgen.core import Structure

@pytest.mark.skipif(not find_spec("matgl"), reason="matgl is not installed")
def test_calculate_property(LiFePO4: Structure) -> None:
    args = namedtuple("args", ["model", "property", "structure", "outfile"])  # noqa: PYI024

    with ScratchDir(".") as _:
        cif_file = "cli_test.cif"
        LiFePO4.to(filename=cif_file)
        a = args("M3GNet", "ElasticityCalc", [cif_file], "CLI.json")
        calculate_property(a)

        assert os.path.exists(a.outfile)

        a = args("M3GNet", "BadCalc", ["cli_test.cif"], "CLI.json")
        with pytest.raises(KeyError):
            calculate_property(a)

        subprocess.check_output(
            [
                "matcalc",
                "calc",
                "-p",
                "ElasticityCalc",
                "-s",
                cif_file,
                "-o",
                "cli_results.json",
            ]
        )
        subprocess.check_output(
            [
                "matcalc",
                "calc",
                "-p",
                "ElasticityCalc",
                "-s",
                cif_file,
                "-o",
                "cli_results.yaml",
            ]
        )

        assert os.path.exists("cli_results.json")
        assert os.path.exists("cli_results.yaml")
