"""Computes all of the singular values for the QKV composition matrices across a model.

Writes a pandas csv as output where each row specifies a single composition term.
"""
import time
import argparse

from gptomics import model, pairengine
from gptomics.svd import SVD

colnames = pairengine.STDCOLNAMES + ["SV_index"]


def main(
    modelname: str,
    outputfilename: str,
    out_biases: bool = True,
    mlps: bool = True,
    lns: bool = True,
    verbose: bool = True,
    gpu_svd: bool = False,
) -> None:
    m = model.model_by_name(modelname, gpu_svd=gpu_svd)

    if verbose:
        print("Starting pair term engine")
        begin = time.time()

    df = pairengine.compute_pair_terms(
        m,
        singular_values,
        colnames=colnames,
        out_biases=out_biases,
        mlps=mlps,
        lns=lns,
        verbose=verbose,
    )

    if verbose:
        end = time.time()
        print(f"Pair engine computation complete in {end-begin:.3f}s")

    writeterms(df, outputfilename)


def singular_values(dst_M: SVD, src_M: SVD):
    """
    normalize by the denominator of the composition term
    """
    return (dst_M @ src_M).S


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
        "--gpu_svd", action="store_true", help="Compute SVDs on the GPU (naively)"
    )

    main(**vars(ap.parse_args()))
