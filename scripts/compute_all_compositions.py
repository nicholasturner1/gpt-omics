"""Computes all of the Q, K, and V contribution terms across GPT-Neo.

Writes a pandas csv as output where each row specifies a single composition term.
"""
import argparse

import numpy as np
import pandas as pd
from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoForCausalLM

from gptomics import transformersio, gptneo, composition as comp


COLNAMES = [
    "src_type",
    "src_layer",
    "src_index",
    "dst_type",
    "dst_layer",
    "dst_index",
    "contrib_type",
    "value",
]


def main(modelname: str, outputfilename: str) -> None:
    model = transformersio.load_model(modelname)

    terms = computeallterms(model)

    writeterms(terms, outputfilename)


def computeallterms(model: GPTNeoForCausalLM) -> pd.DataFrame:
    """Computes all QKV composition terms across a model.

    Args:
        model: A Huggingface Transformers model for GPT-Neo.
    Returns:
        A dataframe that describes each composition term.
    """
    terms = list()
    num_layers = model.config.num_layers
    for layer in range(0, num_layers - 1):
        print(f"Computing output weights for layer: {layer}/{num_layers - 2}", end="\r")
        terms.append(layer_output_weights(model, layer))
    print("")

    return pd.concat(terms, ignore_index=True)


def layer_output_weights(model: GPTNeoForCausalLM, src_layer: int) -> pd.DataFrame:
    """Computes all QKV contribution weights for the outputs of a given layer.

    Args:
        model: A Huggingface Transformers model for GPT-Neo.
        src_layer: The (base 0) index of the layer to analyze.
    Returns:
        A dataframe that describes each composition weight for the source layer.
    """
    rows = list()

    num_heads = model.config.num_heads

    # attention heads
    for src_head in range(num_heads):
        head_rows = src_output_terms(
            model,
            "att_head",
            src_layer,
            src_head,
            gptneo.OV(model, src_layer, src_head),
        )
        rows.extend(head_rows)

    # Self-attention layer output bias
    Obias_rows = src_output_terms(
        model, "att_bias", src_layer, src_head, gptneo.Obias(model, src_layer)
    )
    rows.extend(Obias_rows)

    # MLP
    mlp_weight_rows = src_output_terms(
        model, "mlp_weight", src_layer, 0, gptneo.MLPout(model, src_layer)
    )
    rows.extend(mlp_weight_rows)

    # MLP bias
    mlp_bias_rows = src_output_terms(
        model, "mlp_bias", src_layer, 0, gptneo.MLPbias_out(model, src_layer)
    )
    rows.extend(mlp_bias_rows)

    # Layer norm biases
    # NOTE: these may not pass through a layer norm for every input
    # so they should be centered only once they do.
    ln_bias_1, ln_bias_2 = gptneo.layernorm_biases(model, src_layer)
    ln_bias_1_rows = src_output_terms(
        model, "layernorm_bias1", src_layer, 0, ln_bias_1, adaptive_centering=True
    )
    rows.extend(ln_bias_1_rows)
    ln_bias_2_rows = src_output_terms(
        model, "layernorm_bias2", src_layer, 0, ln_bias_2, adaptive_centering=True
    )
    rows.extend(ln_bias_2_rows)

    rowdict = {i: row for (i, row) in enumerate(rows)}
    return pd.DataFrame.from_dict(rowdict, orient="index", columns=COLNAMES)


def src_output_terms(
    model: GPTNeoForCausalLM,
    src_type: str,
    src_layer: int,
    src_index: int,
    output_space: np.ndarray,
    adaptive_centering: bool = False,
) -> list[list]:
    """Computes all composition terms for an output matrix of a single source.

    Args:
        model: A Huggingface Transformers model for GPT-Neo.
        src_type: A description of the type of source (e.g., "head", "mlp_weight").
        src_layer: The (base 0) index of the layer to analyze.
        src_index: The (base 0) index of the source within the source type.
            This equals 0 for every type other than attention heads.
        output_space: The matrix of (likely centered) weights describing the output
            subspace for the source.
        adaptive_centering: Whether to center the output_space according to when it
            passes through a layernorm. This should be used for layernorm biases.
    Returns:
        A list of rows following the COLNAMES structure for each composition term.
    """
    rows = list()

    if not adaptive_centering:
        output_space = comp.removemean(output_space)

    # Finish off this layer if necessary
    if src_type.startswith("att") or src_type == "layernorm_bias1":
        rows.append(
            mlp_contribution(
                model, src_type, src_layer, src_index, src_layer, output_space
            )
        )

    # first layer norm bias passes through another norm here
    if adaptive_centering and src_type == "layernorm_bias1":
        output_space = comp.removemean(output_space)

    # Moving on to other layers
    for dst_layer in range(src_layer + 1, model.config.num_layers):
        # self-attention layer
        for dst_head in range(model.config.num_heads):
            head_rows = head_contributions(
                model, src_type, src_layer, src_index, dst_layer, dst_head, output_space
            )
            rows.extend(head_rows)

        # second layer norm bias will pass through another norm here
        if (
            adaptive_centering
            and dst_layer == src_layer + 1
            and src_type == "layernorm_bias2"
        ):
            output_space = comp.removemean(output_space)

        # MLP layer
        rows.append(
            mlp_contribution(
                model, src_type, src_layer, src_index, dst_layer, output_space
            )
        )

    return rows


def mlp_contribution(
    model: GPTNeoForCausalLM,
    src_type: str,
    src_layer: int,
    src_index: int,
    dst_layer: int,
    output_space: np.ndarray,
) -> list:
    """Computes the composition term of an output space to the MLP input.

    Args:
        model: A Huggingface Transformers model for GPT-Neo.
        src_type: A description of the type of source (e.g., "head", "mlp_weight").
        src_layer: The (base 0) index of the layer to analyze.
        src_index: The (base 0) index of the source within the source type.
            This equals 0 for every type other than attention heads.
        dst_layer: The (base 0) index of the destination layer to analyze.
        output_space: The matrix of centered weights describing the output subspace
            for the source.
    Returns:
        A row following the COLNAMES structure for the composition term.
    """
    contrib = comp.MLP_in_contribution(
        gptneo.MLPin(model, src_layer), output_space, center=False
    )

    return [
        src_type,
        src_layer,
        src_index,
        "mlp_weight",
        dst_layer,
        0,
        "mlp",
        contrib,
    ]


def head_contributions(
    model: GPTNeoForCausalLM,
    src_type: str,
    src_layer: int,
    src_index: int,
    dst_layer: int,
    dst_head: int,
    output_space: np.ndarray,
) -> list[list]:
    """Computes the composition QKV terms of an output space.

    Args:
        model: A Huggingface Transformers model for GPT-Neo.
        src_type: A description of the type of source (e.g., "head", "mlp_weight").
        src_layer: The (base 0) index of the layer to analyze.
        src_index: The (base 0) index of the source within the source type.
            This equals 0 for every type other than attention heads.
        dst_layer: The (base 0) index of the destination layer to analyze.
        dst_head: The (base 0) index of the destination attention head to analyze.
        output_space: The matrix of centered weights describing the output subspace
            for the source.
    Returns:
        A list of rows following the COLNAMES structure for each composition term.
    """
    QK = gptneo.QK(model, dst_layer, dst_head)
    OV = gptneo.OV(model, dst_layer, dst_head)

    Qcomp = comp.Qcomposition(QK, output_space, center=False)
    Kcomp = comp.Kcomposition(QK, output_space, center=False)
    Vcomp = comp.Vcomposition(OV, output_space, center=False)

    def row(contrib: np.float32, contribtype: str) -> list:
        return [
            src_type,
            src_layer,
            src_index,
            "head",
            dst_layer,
            dst_head,
            contribtype,
            contrib,
        ]

    return [row(Qcomp, "Q"), row(Kcomp, "K"), row(Vcomp, "V")]


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

    main(**vars(ap.parse_args()))
