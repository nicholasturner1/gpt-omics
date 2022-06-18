"""Computes all of the singular values for the weight matrices across a model.

Writes a npz file with the singular values mapped to each tensor name.
"""
import time
import argparse

import numpy as np
from tqdm import tqdm

from gptomics import torchio
from gptomics.svd import SVD


def main(
    modelfilename: str,
    outputfilename: str,
    verbose: bool = True,
    gpu_svd: bool = False,
) -> None:
    if verbose:
        print("Starting computation")
        begin = time.time()

    singularvalues = weight_singular_values(modelfilename, gpu_svd=gpu_svd)

    if verbose:
        end = time.time()
        print(f"Computation complete in {end-begin:.3f}s")

    writenpz(singularvalues, outputfilename)


def weight_singular_values(
    modelfilename: str, gpu_svd: bool = False
) -> dict[str, np.ndarray]:
    """Computes the singular values for each tensor with "weight" in its name."""

    tensor_names = torchio.read_tensor_names(modelfilename)

    svs = dict()

    filtered_names = list(filter(lambda n: "weight" in n, tensor_names))

    for tn in tqdm(filtered_names):
        tensor = torchio.read_tensor(modelfilename, tn).numpy()
        svs[tn] = SVD.frommatrix(tensor, gpu=gpu_svd).S

    return svs


def writenpz(singularvalues: dict, outputfilename: str) -> None:
    """Writes the singular values to a file.

    Args:
        singularvalues: A dict mapping tensor names to singular value (1D) arrays.
        outputfilename: The filename to use when writing the file.
    """
    np.savez_compressed(outputfilename, **singularvalues)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "modelfilename", type=str, help="path to the pytorch_model.bin file"
    )
    ap.add_argument("outputfilename", type=str, help="output filename")
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
