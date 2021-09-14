# as provided in discussions of the issue
# https://github.com/pytorch/fairseq/issues/2120#issuecomment-647429120

import argparse
import os
from typing import List

import torch

from fairseq.data import Dictionary


def load_dict(langs: List[str], path: str) -> Dictionary:
    d = Dictionary.load(path)
    for l in langs:
        d.add_symbol(f"[{l}]")
    d.add_symbol("<mask>")
    return d


def main() -> None:
    parser = argparse.ArgumentParser(description="Trims pre-trained mBART model for fine-tuning.")
    parser.add_argument("--pre-train-dir", type=str, required=True, help="The pre-trained mBART model directory.")
    parser.add_argument("--ft-dict", type=str, required=True, help="The fine-tuning model dictionary.")
    parser.add_argument("--langs", type=str, required=True, help="The pre-trained model languages.")
    parser.add_argument("--output", type=str, required=True, help="The trimmed mBART model.")
    args = parser.parse_args()

    langs = args.langs.split(",")
    pre_dict = load_dict(langs, os.path.join(args.pre_train_dir, "dict.txt"))
    ft_dict = load_dict(langs, args.ft_dict)
    data = torch.load(os.path.join(args.pre_train_dir, "model.pt"))
    model = data["model"]

    mapping: List[int] = []
    for i in range(len(ft_dict)):
        word = ft_dict[i]
        mapping.append(pre_dict.index(word))

    for name in ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]:
        pre_tensor: torch.Tensor = model[name]
        ft_tensor = torch.zeros(
            [len(ft_dict), 1024], dtype=pre_tensor.dtype, layout=pre_tensor.layout, device=pre_tensor.device,
        )
        for ft_i, pre_i in enumerate(mapping):
            ft_tensor[ft_i] = pre_tensor[pre_i]
        model[name] = ft_tensor

    torch.save(data, args.output)


if __name__ == "__main__":
    main()
