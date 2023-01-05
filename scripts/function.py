"""Functional analysis given an input prompt."""
import time
import argparse
from typing import Optional

import torch
from transformers import AutoTokenizer

from gptomics import transformersio, functional as func


def main(
    modelname: str,
    prompt_filename: str,
    output_prefix: str,
    cuda: bool = True,
    block: Optional[int] = None,
    heads: Optional[list[int]] = None,
    attention: bool = True,
    logit_attribution: bool = True,
    save_tokens: bool = True,
) -> None:
    """Reads a prompt from disk, computes attention patterns and logit attributions."""
    assert (
        attention or logit_attribution
    ), "nothing to compute (no attention or logit attribution)"

    print("Reading prompt")
    start = time.time()
    prompt = readprompt(prompt_filename)
    print(f"Complete in {time.time() - start:.3f}s")

    print("Loading model")
    start = time.time()
    model = transformersio.load_model(modelname)
    print(f"Complete in {time.time() - start:.3f}s")

    print("Tokenizing prompt")
    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(modelname)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    tokens = tokenizer.convert_ids_to_tokens(list(input_ids.squeeze()))
    print(f"Complete in {time.time() - start:.3f}s")

    if attention:
        print("Computing attention")
        start = time.time()
        attn = func._attention_pattern(model, input_ids, cuda)
        print(f"Complete in {time.time() - start:.3f}s")

    if logit_attribution:
        print("Computing logit attributions")
        start = time.time()
        # one head at a time if passed
        if heads is not None:
            attrs = list()
            for head in heads:
                attrs.append(
                    func._logit_attribution(
                        model,
                        input_ids,
                        block=block,
                        head=head,
                        cuda=cuda,
                    )
                )
            attr = torch.cat(attrs)

        else:
            attr = func._logit_attribution(
                model,
                input_ids,
                block=block,
                cuda=cuda,
            )
        print(f"Complete in {time.time() - start:.3f}s")

    print("Saving results")
    start = time.time()
    if attention:
        writetensor(attn, output_prefix, "attention")
    if logit_attribution:
        writetensor(attr, output_prefix, "logit_attr")
    if save_tokens:
        writetokens(tokens, output_prefix)
    print(f"Complete in {time.time() - start:.3f}s")


def readprompt(filename: str):
    """Reads a text file prompt and returns the contents."""
    with open(filename) as f:
        return f.read().strip()


def writetensor(t: torch.Tensor, output_prefix: str, tag: str) -> None:
    torch.save(t.clone(), f"{output_prefix}_{tag}.pt")


def writetokens(tokens: list[str], output_prefix: str) -> None:
    with open(f"{output_prefix}_tokens.txt", "w+") as f:
        f.write(" ".join(tokens))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument("modelname")
    ap.add_argument("prompt_filename")
    ap.add_argument("output_prefix")

    ap.add_argument("--heads", type=int, nargs="*", default=None)
    ap.add_argument("--block", type=int, default=None)

    ap.add_argument("--no-cuda", dest="cuda", action="store_false")
    ap.add_argument("--no-attention", dest="attention", action="store_false")
    ap.add_argument("--no-attribution", dest="logit_attribution", action="store_false")
    ap.add_argument("--no-tokens", dest="save_tokens", action="store_false")

    main(**vars(ap.parse_args()))
