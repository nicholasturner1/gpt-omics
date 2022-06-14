"""Computes the variance explained by 1 SV in each composition term.

Writes a pandas csv as output where each row specifies a single composition term.
"""
import time
import argparse

from gptomics import model, pairengine
from gptomics.svd import SVD


def main(
    modelname: str,
    outputfilename: str,
    Obiases: bool = True,
    MLPs: bool = True,
    LNs: bool = True,
    verbose: bool = True,
) -> None:
    m = model.model_by_name(modelname)

    if verbose:
        print("Starting pair term engine")
        begin = time.time()

    df = pairengine.compute_pair_terms(
        m, principal_component, Obiases=Obiases, MLPs=MLPs, LNs=LNs, verbose=verbose
    )

    if verbose:
        end = time.time()
        print(f"Pair engine computation complete in {end-begin:.3f}s")

    writeterms(df, outputfilename)


def principal_component(dst_M: SVD, src_M: SVD):

    sigmas = (dst_M @ src_M).S
    return sigmas[0] ** 2 / sum([sigma ** 2 for sigma in sigmas])
    """
    TODO: use the stacked graphs that nick made in order to visualize
    sigmas vs composition value with x axis as composition value
    """


def writeterms(df, outputfilename: str) -> None:
    """Writes the composition terms to a file.

    Args:
        df: The dataframe to write.
        outputfilename: The filename to use when writing the file.
    """
    df.to_csv(outputfilename)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "modelname", type=str, help="model name in HuggingFace transformers"
    )
    ap.add_argument("outputfilename", type=str, help="output filename")
    ap.add_argument(
        "--no_Obiases",
        dest="Obiases",
        action="store_false",
        help="Do not compute terms for attention head bias terms",
    )
    ap.add_argument(
        "--no_MLPs", dest="MLPs", action="store_false", help="Do not compute MLP terms"
    )
    ap.add_argument(
        "--no_LNs",
        dest="LNs",
        action="store_false",
        help="Do not compute Layer Norm terms",
    )
    ap.add_argument(
        "--quiet",
        dest="verbose",
        action="store_false",
        help="Do not print progress messages",
    )

    main(**vars(ap.parse_args()))
