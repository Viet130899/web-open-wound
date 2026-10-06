"""Convert an mlx-lm LoRA adapter (adapters.safetensors + adapter_config.json,
as produced by `mlx_lm.lora`) into a HuggingFace PEFT adapter that can be
loaded with `PeftModel.from_pretrained` on top of a standard `transformers`
checkpoint of the same base model.

mlx-lm stores, per target linear layer:
    <module_path>.lora_a   shape [in_features, r]
    <module_path>.lora_b   shape [r, out_features]
PEFT (via nn.Linear(in, r) / nn.Linear(r, out)) expects:
    base_model.model.<module_path>.lora_A.default.weight   shape [r, in_features]
    base_model.model.<module_path>.lora_B.default.weight   shape [out_features, r]
so each tensor just needs transposing and the key needs the PEFT prefix/suffix.

Usage:
    python convert_mlx_lora_to_peft.py <adapter_dir> --base-model <hf_repo_or_path> [--out <out_dir>]

<adapter_dir> must contain the original `adapter_config.json` + `adapters.safetensors`
from mlx-lm. The converted files are written to <out_dir> (default: <adapter_dir>),
and the original mlx files are moved into <adapter_dir>/mlx_source/ so nothing is lost.
"""
import argparse
import json
import re
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def convert(adapter_dir: Path, base_model: str, out_dir: Path) -> None:
    mlx_config_path = adapter_dir / "adapter_config.json"
    mlx_weights_path = adapter_dir / "adapters.safetensors"
    with open(mlx_config_path, encoding="utf-8") as f:
        mlx_cfg = json.load(f)

    rank = mlx_cfg["lora_parameters"]["rank"]
    scale = mlx_cfg["lora_parameters"]["scale"]
    alpha = scale * rank  # PEFT scaling = alpha / r, mlx scaling = scale

    converted: dict[str, torch.Tensor] = {}
    target_modules: set[str] = set()
    layers: set[int] = set()

    with safe_open(mlx_weights_path, framework="pt") as f:
        for key in f.keys():
            m = re.match(r"model\.layers\.(\d+)\.(.+)\.(lora_a|lora_b)$", key)
            if not m:
                raise ValueError(f"Unrecognized mlx-lm LoRA key: {key}")
            layer_idx, module_path, ab = m.groups()
            layers.add(int(layer_idx))
            target_modules.add(module_path.rsplit(".", 1)[-1])

            tensor = f.get_tensor(key).T.contiguous()  # transpose: mlx [in,r]/[r,out] -> peft [r,in]/[out,r]
            slot = "lora_A" if ab == "lora_a" else "lora_B"
            # Checkpoint format (get_peft_model_state_dict) keeps the "base_model.model."
            # prefix but has no adapter-name segment — unlike the live module's raw
            # state_dict(), which additionally inserts ".default" before ".weight".
            new_key = f"base_model.model.model.layers.{layer_idx}.{module_path}.{slot}.weight"
            converted[new_key] = tensor.to(torch.bfloat16)

    peft_config = {
        "auto_mapping": None,
        "base_model_name_or_path": base_model,
        "revision": None,
        "task_type": "CAUSAL_LM",
        "peft_type": "LORA",
        "r": rank,
        "lora_alpha": alpha,
        "lora_dropout": 0.0,
        "bias": "none",
        "fan_in_fan_out": False,
        "target_modules": sorted(target_modules),
        "layers_to_transform": sorted(layers),
        "layers_pattern": "layers",
        "modules_to_save": None,
        "inference_mode": True,
    }

    # Back up the mlx-lm originals *before* writing anything to out_dir, since
    # out_dir is usually the same directory as adapter_dir (in-place conversion)
    # and would otherwise clobber them.
    backup_dir = adapter_dir / "mlx_source"
    backup_dir.mkdir(exist_ok=True)
    with open(backup_dir / "adapter_config.json", "w", encoding="utf-8") as f:
        json.dump(mlx_cfg, f, indent=2)
    shutil.move(str(mlx_weights_path), backup_dir / "adapters.safetensors")

    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(converted, out_dir / "adapter_model.safetensors")
    with open(out_dir / "adapter_config.json", "w", encoding="utf-8") as f:
        json.dump(peft_config, f, indent=2)

    print(f"Converted {len(converted)} tensors, {len(layers)} layers, "
          f"target_modules={sorted(target_modules)}, r={rank}, alpha={alpha}")
    print(f"Wrote PEFT adapter to {out_dir}")
    print(f"Original mlx files backed up to {backup_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("adapter_dir", type=Path)
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    convert(args.adapter_dir, args.base_model, args.out or args.adapter_dir)
