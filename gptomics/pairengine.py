"""Computes all of the Q, K, and V contribution terms across GPT-Neo.

Writes a pandas csv as output where each row specifies a single composition term.
"""
import itertools
from collections.abc import Iterable
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd

from .model import Model, GPTJ
from gptomics import composition as comp


STDCOLNAMES = [
    "src_type",
    "src_layer",
    "src_index",
    "dst_type",
    "dst_layer",
    "dst_index",
    "term_type",
    "term_value",
]


def isGPTJ(model: Model):
    return isinstance(model, GPTJ)


def compute_pair_terms(
    model: Model,
    f: Callable,
    colnames: list[str] = STDCOLNAMES,
    Obiases: bool = True,
    MLPs: bool = True,
    LNs: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Computes f across all input/output pairs within a model.

    Args:
        model: A model wrapped as a parameter bag
    Returns:
        A dataframe that describes each composition term.
    """
    terms = list()
    num_layers = model.num_layers
    for layer in range(0, num_layers - 1):
        terms.append(
            layer_output_terms(model, layer, f, colnames, Obiases, MLPs, LNs, verbose)
        )
    if verbose:
        print("")

    return pd.concat(terms, ignore_index=True)


def layer_output_terms(
    model: Model,
    src_layer: int,
    f: Callable,
    colnames: list[str],
    Obiases: bool = True,
    MLPs: bool = True,
    LNs: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Computes f across all outputs of a given layer.

    Args:
        model: A model wrapped as a parameter bag
        src_layer: The (base 0) index of the layer to analyze.
    Returns:
        A dataframe that describes each composition weight for the source layer.
    """
    rows = list()

    num_heads = model.num_heads

    # Extracting output parameters from the model
    OVs = [
        comp.removemean(
            model.OV(src_layer, head, factored=True), method="matrix multiply"
        )
        for head in range(num_heads)
    ]

    if Obiases:
        Obias = comp.removemean(
            model.Obias(src_layer, factored=True), method="matrix multiply"
        )
    else:
        Obias = None

    if MLPs:
        MLPout = comp.removemean(
            model.MLPout(src_layer, factored=True), method="matrix multiply"
        )
        MLPbias = comp.removemean(
            model.MLPbias_out(src_layer, factored=True), method="matrix multiply"
        )
    else:
        MLPout, MLPbias = None, None

    # Layer norm biases
    # NOTE: these may not pass through an LN before
    # being read by another layer, so they should only be
    # centered once they do.
    if LNs:
        ln_biases = model.layernorm_biases(src_layer, factored=True)
        if not isGPTJ(model):
            ln_biases = (
                comp.removemean(ln_biases[0], method="matrix multiply"),
                ln_biases[1],
            )
        else:
            # one LN per layer
            # only hits another LN by the next layer
            pass
    else:
        ln_biases = None

    for dst_layer in range(src_layer, model.num_layers):
        if verbose:
            print(f"Computing terms between layers: {src_layer}->{dst_layer}", end="     \r")
        new_rows = layer_input_terms(
            model, src_layer, dst_layer, f, MLPs, OVs, Obias, MLPout, MLPbias, ln_biases
        )
        rows.extend(new_rows)

        # center LN bias for the next layer
        if src_layer == dst_layer and LNs:
            if isGPTJ(model):
                ln_biases = comp.removemean(ln_biases, method="matrix multiply")
            else:
                # two bias terms, 1st normalized within layer_input_terms,
                # and we need to replicate that work for the other layers
                ln_biases = (
                    comp.removemean(ln_biases[0], method="matrix multiply"),
                    ln_biases[1],
                )

        if src_layer == dst_layer - 1:
            if not isGPTJ(model):
                # 2nd bias normalized within input_terms for this layer
                ln_biases = (
                    ln_biases[0],
                    comp.removemean(ln_biases[0], method="matrix multiply"),
                )

    rowdict = {i: row for (i, row) in enumerate(rows)}
    return pd.DataFrame.from_dict(rowdict, orient="index", columns=colnames)


def layer_input_terms(
    model: Model,
    src_layer: int,
    dst_layer: int,
    f: Callable,
    computeMLPterms: bool = False,
    OVs: Optional[list[np.ndarray]] = None,
    Obias: Optional[np.ndarray] = None,
    MLPout: Optional[np.ndarray] = None,
    MLPbias: Optional[np.ndarray] = None,
    ln_biases: Optional[Union[tuple[np.ndarray, np.ndarray], np.ndarray]] = None,
) -> list[list]:
    """Computes f across all inputs of a given layer.

    Args:
        model: A model wrapped as a parameter bag
        src_layer: The (base 0) index of the source of parameter arguments.
    Returns:
        A dataframe that describes each composition weight for the source layer.
    """
    rows = list()

    def takes_any_input(dst_type: str) -> bool:
        defined = list()
        if OVs is not None:
            defined.append("att_head")
        if Obias is not None:
            defined.append("att_head")
        if MLPout is not None:
            defined.append("mlp_weight")
        if MLPbias is not None:
            defined.append("mlp_bias")
        if ln_biases is not None:
            defined.append("layernorm_bias")

        return any(
            model.sends_input_to(src_type, src_layer, dst_type, dst_layer)
            for src_type in defined
        )

    if takes_any_input("att_head"):
        rows.extend(
            att_head_input_terms(
                model, src_layer, dst_layer, f, OVs, Obias, MLPout, MLPbias, ln_biases
            )
        )

    if src_layer == dst_layer - 1 and not isGPTJ(model):
        # center second LN bias
        ln_biases = (
            ln_biases[0],
            comp.removemean(ln_biases[1], method="matrix multiply"),
        )

    if computeMLPterms and takes_any_input("mlp_weight"):
        rows.extend(
            mlp_input_terms(
                model, src_layer, dst_layer, f, OVs, Obias, MLPout, MLPbias, ln_biases
            )
        )

    return rows


def mlp_input_terms(
    model: Model,
    src_layer: int,
    dst_layer: int,
    f: Callable,
    OVs: list[np.ndarray],
    Obias: Optional[np.ndarray] = None,
    MLPout: Optional[np.ndarray] = None,
    MLPbias: Optional[np.ndarray] = None,
    ln_biases: Optional[Union[tuple[np.ndarray, np.ndarray], np.ndarray]] = None,
) -> list[list]:
    """"""
    rows = list()

    MLPin = model.MLPin(dst_layer, factored=True)

    dst_type = "mlp_weight"
    dst_index = 0
    term_type = "mlp"

    def mlp_takes_input(src_type):
        return model.sends_input_to(src_type, src_layer, dst_type, dst_layer)

    def compute_term(
        src_M: np.ndarray, src_type: str, src_index: int = 0,
    ):
        value = f(MLPin, src_M)
        return make_rows(
            src_type,
            src_layer,
            src_index,
            dst_type,
            dst_layer,
            dst_index,
            term_type,
            value,
        )

    # attention head input
    if OVs is not None and mlp_takes_input("att_head"):
        for (i, OV) in enumerate(OVs):
            rows.extend(compute_term(OV, "att_head", i))

    if Obias is not None and mlp_takes_input("att_head"):
        rows.extend(compute_term(Obias, "att_bias"))

    # MLP input
    if MLPout is not None and mlp_takes_input("mlp_weight"):
        rows.extend(compute_term(MLPout, "mlp_weight"))

    if MLPbias is not None and mlp_takes_input("mlp_bias"):
        rows.extend(compute_term(MLPbias, "mlp_bias"))

    # LN input
    if ln_biases is not None and mlp_takes_input("layernorm_bias"):
        if isGPTJ(model):
            # single bias
            rows.extend(compute_term(ln_biases, "layernorm_bias"))
        else:
            # two biases
            rows.extend(compute_term(ln_biases[0], "layernorm_bias1"))
            if src_layer != dst_layer:
                rows.extend(compute_term(ln_biases[1], "layernorm_bias2"))

    return rows


def att_head_input_terms(
    model: Model,
    src_layer: int,
    dst_layer: int,
    f: Callable,
    src_OVs: list[np.ndarray],
    Obias: Optional[np.ndarray] = None,
    MLPout: Optional[np.ndarray] = None,
    MLPbias: Optional[np.ndarray] = None,
    ln_biases: Optional[Union[tuple[np.ndarray, np.ndarray], np.ndarray]] = None,
) -> list[list]:
    """"""

    QKs = [model.QK(dst_layer, i, factored=True) for i in range(model.num_heads)]
    dst_OVs = [model.OV(dst_layer, i, factored=True) for i in range(model.num_heads)]

    dst_type = "att_head"

    # Convenience functions
    def att_takes_input(src_type):
        return model.sends_input_to(src_type, src_layer, dst_type, dst_layer)

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
            src_layer,
            src_index,
            dst_type,
            dst_layer,
            dst_index,
            term_type,
            term_value,
        )

    def compute_terms(
        src_M: np.ndarray, src_type: str, src_index: int = 0,
    ):
        """Computes terms for a given input matrix."""
        Qterms = [f(QK.T, src_M) for QK in QKs]
        Kterms = [f(QK, src_M) for QK in QKs]
        Vterms = [f(OV, src_M) for OV in dst_OVs]

        rows = list()
        for (i, Qt) in enumerate(Qterms):
            rows.extend(_make_rows(Qt, "Q", src_type, src_index, i))

        for (i, Kt) in enumerate(Kterms):
            rows.extend(_make_rows(Kt, "K", src_type, src_index, i))

        for (i, Vt) in enumerate(Vterms):
            newrows = _make_rows(Vt, "V", src_type, src_index, i)
            rows.extend(newrows)

        return rows

    # Actual logic
    rows = list()

    # attention head input
    if src_OVs is not None and att_takes_input("att_head"):
        for i in range(len(src_OVs)):
            rows.extend(compute_terms(src_OVs[i], "att_head", i))

    if Obias is not None and att_takes_input("att_head"):
        rows.extend(compute_terms(Obias, "att_bias"))

    # MLP input
    if MLPout is not None and att_takes_input("mlp_weight"):
        rows.extend(compute_terms(MLPout, "mlp_weight"))

    if MLPbias is not None and att_takes_input("mlp_bias"):
        rows.extend(compute_terms(MLPbias, "mlp_bias"))

    # LN biases
    if ln_biases is not None and att_takes_input("layernorm_bias"):
        if isGPTJ(model):
            # single bias
            rows.extend(compute_terms(ln_biases, "layernorm_bias"))
        elif src_layer != dst_layer:
            # two biases
            rows.extend(compute_terms(ln_biases[0], "layernorm_bias1"))
            rows.extend(compute_terms(ln_biases[1], "layernorm_bias2"))

    return rows


def make_rows(
    src_type: str,
    src_layer: int,
    src_index: int,
    dst_type: str,
    dst_layer: int,
    dst_index: int,
    term_type: str,
    term_value: Union[Iterable, float],
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

    if isinstance(term_value, Iterable):
        return [[*fixed_fields, v, i] for (i, v) in enumerate(term_value)]
    else:
        return [[*fixed_fields, term_value]]
