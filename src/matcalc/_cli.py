"""Command line interface to matcalc."""

from __future__ import annotations

import argparse
import pprint
import typing

from monty.json import jsanitize
from monty.serialization import dumpfn
from pymatgen.core import Structure

import matcalc as mtc

if typing.TYPE_CHECKING:
    from typing import Any


def calculate_property(args: Any) -> None:
    """
    Implements calculate property.

    :param args:
    :return:
    """
    calculator = mtc.load_fp(args.model)
    mod = mtc.__dict__[args.property](calculator)
    results = []
    for f in args.structure:
        s = Structure.from_file(f)
        results.append(mod.calc(s))
    if args.outfile:
        if "json" not in args.outfile:
            dumpfn(jsanitize(results), args.outfile)
        else:
            dumpfn(results, args.outfile)
    else:
        pprint.pprint(results)  # noqa:T203


def clear_cache(args: Any) -> None:
    """
    Clear the benchmark cache.

    :param args:
    :return:
    """
    mtc.clear_cache(confirm=args.yes)


def main() -> None:
    """Handle main."""
    parser = argparse.ArgumentParser(
        description="""A CLI interface for rapid calculations of materials properties with matcalc. Type
        "matcalc sub-command -h".""",
        epilog="""Author: MatCalc Development Team""",
    )

    subparsers = parser.add_subparsers()

    p_calc = subparsers.add_parser("calc", help="Calculate properties using universal calculators.")

    p_calc.add_argument(
        "-s",
        "--structure",
        dest="structure",
        nargs="+",
        required=True,
        help="Input files containing structure. Any format supported by pymatgen's Structure.from_file method.",
    )

    p_calc.add_argument(
        "-m",
        "--model",
        dest="model",
        type=str,
        choices=mtc.UNIVERSAL_CALCULATORS,
        default="TensorNet",
        help="Universal MLIP to use.",
    )

    p_calc.add_argument(
        "-p",
        "--property",
        dest="property",
        type=str,
        choices=[m for m in dir(mtc) if m.endswith("Calc") and m != "NEBCalc"],
        default="RelaxCalc",
        help="PropCalc to use. Defaults to RelaxCalc.",
    )

    p_calc.add_argument(
        "-o",
        "--outfile",
        dest="outfile",
        type=str,
        nargs="?",
        help="Output file in json or yaml. Defaults to stdout.",
    )

    p_calc.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        default=False,
        action="store_true",
        help="Verbose output.",
    )

    p_calc.set_defaults(func=calculate_property)

    p_clear = subparsers.add_parser("clear", help="Clear cache.")

    p_clear.add_argument(
        "-y",
        "--yes",
        dest="yes",
        action="store_true",
        help="Skip confirmation.",
    )

    p_clear.set_defaults(func=clear_cache)

    args = parser.parse_args()

    return args.func(args)
