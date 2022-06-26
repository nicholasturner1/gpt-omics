"""Computes all of the Q, K, and V contribution terms across a model.

Writes a pandas csv as output where each row specifies a single composition term.
"""
import argparse

import numpy as np

from gptomics import pairengine, composition as comp
from gptomics.types import ParamMatrix


@pairengine.pair_engine_fn
def main(
    dst_M: ParamMatrix,
    src_M: ParamMatrix,
    center: bool = False,
    wikidenom: bool = True,
) -> np.float32:
    return comp.basecomposition(dst_M, src_M, center=center, wikidenom=wikidenom)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--origdenom",
        dest="wikidenom",
        action="store_false",
        help="Compute reverse edges instead of forward edges.",
    )

    main(**vars(pairengine.parse_args(ap)))
