"""Prefix matching and copying scores."""
import time
import random
import argparse
from typing import Optional, Union

import torch
from transformers import AutoTokenizer

from gptomics import transformersio, functional as func


def main(
    modelname: str,
    output_prefix: str,
    cuda: bool = True,
    block: Optional[int] = None,
    heads: Optional[list[int]] = None,
    prefix_matching: bool = True,
    copying: bool = True,
    seed: Optional[int] = None,
    num_samples: int = 5,
    seqlen: int = 25,
    repeats: int = 4,
    bos_token: bool = True,
) -> None:
    """Computes prefix matching and copying scores on random prompts."""
    assert (
        prefix_matching or copying
    ), "nothing to compute (no prefix_matching or copying)"

    print("Loading model")
    start = time.time()
    model = transformersio.load_model(modelname)
    print(f"Complete in {time.time() - start:.3f}s")
    tokenizer = AutoTokenizer.from_pretrained(modelname)

    print("Loading tokenizer")
    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(modelname)
    print(f"Complete in {time.time() - start:.3f}s")

    random.seed(seed)

    pm_scores = list()
    cp_scores = list()
    for i in range(num_samples):
        print(f"Generating random prompt {i+1}")
        start = time.time()
        # seed already set above
        input_ids = func.random_prompt(tokenizer, seqlen, repeats, bos_token)
        print(f"Complete in {time.time() - start:.3f}s")

        if prefix_matching:
            print("Computing prefix matching score")
            start = time.time()
            pm_scores.append(
                func._compute_prefix_matching_score(
                    model, input_ids, seqlen, repeats, bos_token, cuda=cuda
                )[0]
            )
            print(f"Complete in {time.time() - start:.3f}s")

        if copying:
            print("Computing copying score")
            start = time.time()
            # one head at a time if passed since the logit attr fn
            # is more memory efficient this way
            if heads is not None:
                cp_scores.append(
                    torch.cat(
                        [
                            func._compute_copying_score(
                                model,
                                input_ids,
                                seqlen,
                                repeats,
                                bos_token,
                                block=block,
                                head=head,
                                cuda=cuda,
                            )[0]
                            for head in heads
                        ]
                    )
                )

            else:
                cp_scores.append(
                    func._compute_copying_score(
                        model,
                        input_ids,
                        seqlen,
                        repeats,
                        bos_token,
                        block=block,
                        cuda=cuda,
                    )[0]
                )
                print(f"Complete in {time.time() - start:.3f}s")

    print("Saving means")
    start = time.time()
    if prefix_matching:
        writetensor(sum(pm_scores) / len(pm_scores), output_prefix, "prefix_matching")
    if copying:
        writetensor(sum(cp_scores) / len(cp_scores), output_prefix, "copying")
    print(f"Complete in {time.time() - start:.3f}s")


def readprompt(filename: str):
    """Reads a text file prompt and returns the contents."""
    with open(filename) as f:
        return f.read().strip()


def writetensor(t: Union[torch.Tensor, float], output_prefix: str, tag: str) -> None:
    t = t.clone() if isinstance(t, torch.Tensor) else t
    torch.save(t, f"{output_prefix}_{tag}.pt")


def writetokens(tokens: list[str], output_prefix: str) -> None:
    with open(f"{output_prefix}_tokens.txt", "w+") as f:
        f.write(" ".join(tokens))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument("modelname")
    ap.add_argument("output_prefix")

    ap.add_argument("--heads", type=int, nargs="*", default=None)
    ap.add_argument("--block", type=int, default=None)

    ap.add_argument("--seqlen", type=int, default=25)
    ap.add_argument("--repeats", type=int, default=4)
    ap.add_argument("--num_samples", type=int, default=5)
    ap.add_argument("--seed", type=int, default=None)

    ap.add_argument("--no-cuda", dest="cuda", action="store_false")
    ap.add_argument(
        "--no-prefix-matching", dest="prefix_matching", action="store_false"
    )
    ap.add_argument("--no-copying", dest="copying", action="store_false")
    ap.add_argument("--no-bos-token", dest="bos_token", action="store_false")

    main(**vars(ap.parse_args()))
