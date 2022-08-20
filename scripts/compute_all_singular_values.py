"""Computes all of the singular values for the QKV composition matrices across a model.

Writes a pandas csv as output where each row specifies a single composition term.
"""
import argparse

import numpy as np

from gptomics import pairengine
from gptomics.svd import SVD


@pairengine.pair_engine_fn
def singular_values(dst_M: SVD, src_M: SVD) -> np.ndarray:
    """
    normalize by the denominator of the composition term
    """
    return (dst_M @ src_M).S


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    singular_values(**vars(pairengine.parse_args(ap)))
