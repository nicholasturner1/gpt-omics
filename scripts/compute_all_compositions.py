"""Computes all of the Q, K, and V contribution terms across a model.

Writes a pandas csv as output where each row specifies a single composition term.
"""
import argparse

from gptomics import pairengine, composition as comp
from gptomics.types import ParamMatrix


@pairengine.pair_engine_fn
def main(
    dst_M: ParamMatrix,
    src_M: ParamMatrix,
    center: bool = False,
    denom: str = "wiki",
) -> float:
    return comp.basecomposition(dst_M, src_M, center=center, denom=denom)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--denom",
        help="Which denominator to use",
        default="wiki",
    )

    main(**vars(pairengine.parse_args(ap)))
