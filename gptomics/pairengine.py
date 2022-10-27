"""Computes all of the Q, K, and V contribution terms across GPT-Neo.

Writes a pandas csv as output where each row specifies a single composition term.
"""
from __future__ import annotations

import time
import argparse
import itertools
from functools import partial
from collections import OrderedDict
from typing import Callable, Union, Optional, Generator

import torch
import pandas as pd

from .svd import SVD
from .types import ParamMatrix
from . import composition as comp
from .model import Model, Layer, SelfAttention, MLP, LayerNorm, model_by_name


STDCOLNAMES = [
    "src_type",
    "src_block",
    "src_index",
    "dst_type",
    "dst_block",
    "dst_index",
    "term_type",
    "term_value",
]

__all__ = ["compute_pair_terms", "pair_engine_fn", "parse_args"]


def compute_pair_terms(
    model: Model,
    f: Callable,
    colnames: list[str] = STDCOLNAMES,
    att_heads: bool = True,
    mlps: bool = True,
    lns: bool = True,
    verbose: int = 1,
    reverse: bool = False,
    blocks: Optional[list[int]] = None,
) -> pd.DataFrame:
    """Computes f across all input/output pairs within a model.

    Args:
        model: A model wrapped as a parameter bag
    Returns:
        A dataframe that describes each composition term.
    """
    terms = list()
    blocks = list(range(0, model.num_blocks)) if blocks is None else blocks

    edge_types = collect_edge_types(att_heads, mlps, lns)

    if reverse:
        blocks = list(reversed(blocks))  # type: ignore[assignment]

    for block in blocks:
        terms.append(
            block_output_terms(
                model, block, f, colnames, edge_types, verbose, reverse, blocks
            )
        )
    if verbose:
        print("")

    return pd.concat(terms, ignore_index=True)


def collect_edge_types(
    att_heads: bool = True, mlps: bool = True, lns: bool = True
) -> list[tuple[Layer, Layer]]:
    """Figures out which types of edge terms the user wants to compute."""
    src_types: list[Layer] = list()
    dst_types: list[Layer] = list()

    if att_heads:
        src_types.append(SelfAttention())
        dst_types.append(SelfAttention())

    if mlps:
        src_types.append(MLP())
        dst_types.append(MLP())

    if lns:
        src_types.append(LayerNorm(0))
        src_types.append(LayerNorm(1))

    return list(itertools.product(src_types, dst_types))


def block_output_terms(
    model: Model,
    src_block: int,
    f: Callable,
    colnames: list[str],
    edge_types: list[tuple[Layer, Layer]],
    verbose: int = 1,
    reverse: bool = False,
    blocks: list[int] = [],
) -> pd.DataFrame:
    """Computes f across all outputs of a given layer.

    Args:
        model: A model wrapped as a parameter bag
        src_layer: The (base 0) index of the layer to analyze.
    Returns:
        A dataframe that describes each composition weight for the source layer.
    """
    rows = list()

    # Timing parameter loading (shown if verbose > 1)
    begin = time.time()

    src_types = [edge[0] for edge in edge_types]
    src_tensors = block_output_tensors(model, src_block, src_types, factored=True)

    end = time.time()

    if verbose > 1:
        print(f"Output parameter loading finished in {end-begin:.3f}s")

    blocks_ = set(blocks)

    def inblocks(b):
        return b in blocks_

    if not reverse:
        dst_blocks = filter(inblocks, range(src_block, model.num_blocks))
    else:
        dst_blocks = filter(
            inblocks, reversed(range(0, src_block + 1))
        )  # type: ignore[assignment]

    for dst_block in dst_blocks:
        if verbose:
            print(
                f"Computing terms between blocks: {src_block}->{dst_block}",
                end="     \r",
            )
        new_rows = block_input_terms(
            model,
            src_block,
            dst_block,
            f,
            edge_types,
            src_tensors,
            reverse,
        )
        rows.extend(new_rows)

    if verbose:
        print("")

    rowdict = {i: row for (i, row) in enumerate(rows)}
    return pd.DataFrame.from_dict(rowdict, orient="index", columns=colnames)


class BlockTensors:
    """A container class to facilitate handling the tensors of a block in order."""

    def __init__(self, layers, tensors):
        assert len(layers) == len(tensors), "mismatched layers & tensors"
        assert all(isinstance(layer, Layer) for layer in layers)

        self.__dict = OrderedDict(zip(layers, tensors))

    def __iter__(self) -> Generator:
        return iter(self.__dict.items())

    def __len__(self) -> int:
        return len(self.layers)

    def __getitem__(self, layer: Layer):
        return self.__dict[layer]

    def __setitem__(self, layer: Layer, value) -> None:
        self.__dict[layer] = value

    @property
    def layers(self):
        return self.__dict.keys()

    @property
    def tensors(self):
        return self.__dict.values()


def block_output_tensors(
    model: Model,
    block: int,
    layertypes: list[Layer] = [SelfAttention(), MLP(), LayerNorm()],
    factored: bool = True,
) -> BlockTensors:
    """Reads all contributions of a single layer to the residual stream."""
    layers = model.block_structure()

    layernorms = None
    tensors: list[Union[None, ParamMatrix, list[ParamMatrix]]] = list()
    for layer in layers:
        if isinstance(layer, SelfAttention):
            if layer in layertypes:
                att_tensors = [
                    model.ov(block, head, factored) for head in range(model.num_heads)
                ]

                # adding the bias in the final spot if present
                out_bias = model.out_bias(block, factored)
                if out_bias is not None:
                    att_tensors.append(model.out_bias(block, factored))

                tensors.append(att_tensors)
            else:
                tensors.append(None)

        elif isinstance(layer, MLP):
            if layer in layertypes:
                tensors.append(
                    [
                        model.mlp_out(block, factored),
                        model.mlp_bias_out(block, factored),
                    ]
                )
            else:
                tensors.append(None)

        elif isinstance(layer, LayerNorm):
            if layer in layertypes:
                if layernorms is None:  # first layer norm
                    layernorms = model.ln_biases(block, factored)
                if isinstance(layernorms, tuple) or isinstance(layernorms, list):
                    tensors.append(layernorms[layer.index])
                else:
                    tensors.append(layernorms)
            else:
                tensors.append(None)

        else:
            raise ValueError(f"unrecognized layer type: {layer}")

    return BlockTensors(layers, tensors)


def block_input_terms(
    model: Model,
    src_block: int,
    dst_block: int,
    f: Callable,
    edge_types: list[tuple[Layer, Layer]],
    src_tensors: BlockTensors,
    reverse: bool = False,
) -> list[list]:
    """Computes f across all inputs of a given block.

    Args:
        model: A model wrapped as a parameter bag
        src_block: The (base 0) index of the source of parameter arguments.
    Returns:
        A dataframe that describes each composition weight for the source block.
    """
    rows = list()

    output_types = set(edge[0] for edge in edge_types)
    input_types = set(edge[1] for edge in edge_types)

    def takes_any_input(dst_layer: Layer) -> bool:
        if not reverse:
            return any(
                model.sends_input_to(src_layer, src_block, dst_layer, dst_block)
                for src_layer in output_types
            )
        else:
            return any(
                model.sends_input_to(src_layer, dst_block, dst_layer, src_block)
                for src_layer in output_types
            )

    for layer in model.block_structure():
        if layer in input_types and takes_any_input(layer):

            if isinstance(layer, SelfAttention):
                layernorm = model.normalizing_layer(layer)
                normed_tensors = apply_layer_norm(
                    model, src_block, dst_block, layernorm, src_tensors, reverse
                )
                rows.extend(
                    att_head_input_terms(
                        model,
                        src_block,
                        dst_block,
                        layer,
                        f,
                        normed_tensors,
                        reverse,
                    )
                )

            elif isinstance(layer, MLP):
                layernorm = model.normalizing_layer(layer)
                normed_tensors = apply_layer_norm(
                    model, src_block, dst_block, layernorm, src_tensors, reverse
                )
                rows.extend(
                    mlp_input_terms(
                        model,
                        src_block,
                        dst_block,
                        layer,
                        f,
                        normed_tensors,
                        reverse,
                    )
                )

            else:
                raise ValueError(f"unrecognized layer type: {layer}")

    return rows


def apply_layer_norm(
    model: Model,
    src_block: int,
    dst_block: int,
    dst_layer: Layer,
    src_tensors: BlockTensors,
    reverse: bool = False,
) -> BlockTensors:
    """Modifies the contribution spaces to account for a layer norm in-place."""
    ln_weights = model.ln_weights(dst_block, factored=False)[dst_layer.index]
    assert isinstance(ln_weights, torch.Tensor)

    ln_scalemat = SVD.frommatrix(torch.diag(ln_weights.ravel()))

    def ln_takes_input(src_layer: Layer) -> bool:
        if not reverse:
            return model.sends_input_to(src_layer, src_block, dst_layer, dst_block)
        else:
            return model.sends_input_to(src_layer, dst_block, dst_layer, src_block)

    def apply_ln(tensor: ParamMatrix) -> ParamMatrix:
        centered = comp.removemean(tensor, "matrix multiply")
        return ln_scalemat @ centered

    normed_layers = list()
    normed_tensors: list[Union[None, ParamMatrix, list[ParamMatrix]]] = list()
    for layer, tensor in src_tensors:
        if ln_takes_input(layer):

            # tensor values within BlockTensors can be a list or a single tensor
            if isinstance(tensor, list):
                normed_layers.append(layer)
                normed_tensors.append(list(map(apply_ln, tensor)))
            elif isinstance(tensor, SVD) or isinstance(tensor, torch.Tensor):
                normed_tensors.append(apply_ln(tensor))
            else:
                pass

        # Accounting for this layer's own unnormalized input to the next layer
        elif layer == dst_layer and src_block == dst_block:
            normed_layers.append(layer)
            normed_tensors.append(tensor)

    return BlockTensors(normed_layers, normed_tensors)


def mlp_input_terms(
    model: Model,
    src_block: int,
    dst_block: int,
    dst_layer: Layer,
    f: Callable,
    src_tensors: BlockTensors,
    reverse: bool = False,
) -> list[list]:
    """Computes the terms of f that describe an input to an MLP layer."""
    # Reading input weight parameters
    mlp_in = model.mlp_in(dst_block, factored=True)

    # Some shortcuts to make the code cleaner below
    def mlp_takes_input(src_layer: Layer) -> bool:
        if not reverse:
            return model.sends_input_to(src_layer, src_block, dst_layer, dst_block)
        else:
            return model.sends_input_to(src_layer, dst_block, dst_layer, src_block)

    def compute_terms(
        src_M: ParamMatrix,
        src_typename: str,
        src_index: int = 0,
    ) -> list:
        value = f(mlp_in, src_M)
        return make_rows(
            src_typename,
            src_block,
            src_index,
            dst_layer.layername,
            dst_block,
            dst_layer.index,
            "mlp_weight",
            value,
        )

    # Actually computing the terms
    rows = list()

    for layer, tensor in src_tensors:
        if mlp_takes_input(layer) and tensor is not None:

            if isinstance(layer, SelfAttention) and tensor is not None:
                for (i, ov) in enumerate(tensor[: model.num_heads]):
                    rows.extend(compute_terms(ov, layer.layername, i))
                # computing terms for the self-attention bias if it exists
                for (i, ob) in enumerate(tensor[model.num_heads :]):
                    rows.extend(compute_terms(ob, "att_bias", i))

            if isinstance(layer, MLP) and tensor is not None:
                rows.extend(compute_terms(tensor[0], "mlp_weight", layer.index))
                rows.extend(compute_terms(tensor[1], "mlp_bias", layer.index))

            if isinstance(layer, LayerNorm) and tensor is not None:
                rows.extend(compute_terms(tensor, "layernorm_bias", layer.index))

    return rows


def att_head_input_terms(
    model: Model,
    src_block: int,
    dst_block: int,
    dst_layer: Layer,
    f: Callable,
    src_tensors: BlockTensors,
    reverse: bool = False,
) -> list[list]:
    """Computes the terms of f that describe an input to a SelfAttention layer."""
    # Reading input weight parameters
    qks = [model.qk(dst_block, i, factored=True) for i in range(model.num_heads)]
    ovs = [model.ov(dst_block, i, factored=True) for i in range(model.num_heads)]

    # Convenience functions
    def att_takes_input(src_layer):
        if not reverse:
            return model.sends_input_to(src_layer, src_block, dst_layer, dst_block)
        else:
            return model.sends_input_to(src_layer, dst_block, dst_layer, src_block)

    def _make_rows(
        term_value,
        term_type: str,
        src_type: str,
        src_index: int = 0,
        dst_index: int = 0,
    ) -> list[list]:
        """A shortcut to avoid repeating args."""
        return make_rows(
            src_type,
            src_block,
            src_index,
            dst_layer.layername,
            dst_block,
            dst_index,
            term_type,
            term_value,
        )

    def compute_terms(
        src_M: ParamMatrix,
        src_type: str,
        src_index: int = 0,
    ):
        """Computes terms for a given input matrix."""
        qterms = [f(qk.T, src_M) for qk in qks]
        kterms = [f(qk, src_M) for qk in qks]
        vterms = [f(ov, src_M) for ov in ovs]

        rows = list()
        for (i, qt) in enumerate(qterms):
            rows.extend(_make_rows(qt, "Q", src_type, src_index, i))

        for (i, kt) in enumerate(kterms):
            rows.extend(_make_rows(kt, "K", src_type, src_index, i))

        for (i, vt) in enumerate(vterms):
            rows.extend(_make_rows(vt, "V", src_type, src_index, i))

        return rows

    # Actual logic
    rows = list()

    for layer, tensor in src_tensors:
        if att_takes_input(layer) and tensor is not None:

            if isinstance(layer, SelfAttention) and tensor is not None:
                for (i, ov) in enumerate(tensor[: model.num_heads]):
                    rows.extend(compute_terms(ov, layer.layername, i))
                # computing terms for the self-attention bias if it exists
                for (i, ob) in enumerate(tensor[model.num_heads :]):
                    rows.extend(compute_terms(ob, "att_bias", i))

            if isinstance(layer, MLP) and tensor is not None:
                rows.extend(compute_terms(tensor[0], "mlp_weight", layer.index))
                rows.extend(compute_terms(tensor[1], "mlp_bias", layer.index))

            if isinstance(layer, LayerNorm) and tensor is not None:
                rows.extend(compute_terms(tensor, "layernorm_bias", layer.index))

    return rows


def make_rows(
    src_type: str,
    src_layer: int,
    src_index: int,
    dst_type: str,
    dst_layer: int,
    dst_index: int,
    term_type: str,
    term_value: Union[Generator, float],
) -> list[list]:
    """Formats the values returned by the callable as rows."""
    fixed_fields = [
        src_type,
        src_layer,
        src_index,
        dst_type,
        dst_layer,
        dst_index,
        term_type,
    ]

    if isinstance(term_value, Generator):
        return [[*fixed_fields, v, i] for (i, v) in enumerate(term_value)]
    else:
        return [[*fixed_fields, term_value]]


def pair_engine_fn(f: Callable) -> Callable:
    """A decorator for making computation scripts easier to write."""

    def wrapped(
        modelname: str,
        outputfilename: str,
        colnames: list[str] = STDCOLNAMES,
        att_heads: bool = True,
        mlps: bool = True,
        lns: bool = True,
        verbose: int = 1,
        reverse: bool = False,
        device: str = "cpu",
        *args,
        **kwargs,
    ) -> None:

        model = model_by_name(modelname, torch.device(device))

        if verbose:
            print("Starting pair term engine")
            begin = time.time()

        partial_f = partial(f, *args, **kwargs)

        df = compute_pair_terms(
            model,
            partial_f,
            colnames=colnames,
            att_heads=att_heads,
            mlps=mlps,
            lns=lns,
            verbose=verbose,
            reverse=reverse,
        )

        if verbose:
            end = time.time()
            print(f"Pair engine computation complete in {end-begin:.3f}s")

        if outputfilename.endswith(".gz"):
            df.to_csv(outputfilename, compression="gzip")
        else:
            df.to_csv(outputfilename)

    return wrapped


def parse_args(ap: Optional[argparse.ArgumentParser] = None) -> argparse.Namespace:
    """An argument parsing function for making computation scripts easier to write."""
    ap = argparse.ArgumentParser() if ap is None else ap

    ap.add_argument("modelname", type=str, help="model name")
    ap.add_argument("outputfilename", type=str, help="output filename")
    ap.add_argument(
        "--no_att_heads",
        dest="att_heads",
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
        "--device", help="Which (torch) device should hold the tensors for computations"
    )

    return ap.parse_args()
