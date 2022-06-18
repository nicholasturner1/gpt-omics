"""Computes all of the Q, K, and V contribution terms across a model.

Writes a pandas csv as output where each row specifies a single composition term.
"""
import time
import argparse
from functools import partial

from gptomics import model, pairengine, composition as comp


def main(
    modelname: str,
    outputfilename: str,
    out_biases: bool = True,
    mlps: bool = True,
    lns: bool = True,
    verbose: bool = True,
    reverse: bool = False,
    wikidenom: bool = False,
) -> None:
    m = model.model_by_name(modelname)

    if verbose:
        print("Starting pair term engine")
        begin = time.time()

    f = partial(comp.basecomposition, center=False, wikidenom=wikidenom)

    df = pairengine.compute_pair_terms(
        m,
        f,
        out_biases=out_biases,
        mlps=mlps,
        lns=lns,
        verbose=verbose,
        reverse=reverse,
    )

    if verbose:
        end = time.time()
        print(f"Pair engine computation complete in {end-begin:.3f}s")

    writeterms(df, outputfilename)


def writeterms(df, outputfilename: str) -> None:
    """Writes the composition terms to a file.

    Args:
        df: The dataframe to write.
        outputfilename: The filename to use when writing the file.
    """
    df.to_csv(outputfilename)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument("modelname", type=str, help="model name")
    ap.add_argument("outputfilename", type=str, help="output filename")
    ap.add_argument(
        "--no_out_biases",
        dest="out_biases",
        action="store_false",
        help="Do not compute terms for attention head bias terms",
    )
    ap.add_argument(
        "--no_mlps", dest="mlps", action="store_false", help="Do not compute MLP terms"
    )
    ap.add_argument(
        "--no_lns",
        dest="lns",
        action="store_false",
        help="Do not compute Layer Norm terms",
    )
    ap.add_argument(
        "--quiet",
        dest="verbose",
        action="store_false",
        help="Do not print progress messages",
    )
    ap.add_argument(
        "--reverse",
        action="store_true",
        help="Compute reverse edges instead of forward edges.",
    )
    ap.add_argument(
        "--wikidenom",
        action="store_true",
        help="Compute reverse edges instead of forward edges.",
    )

    main(**vars(ap.parse_args()))
