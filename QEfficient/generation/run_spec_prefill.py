from __future__ import annotations

import argparse
import os
import time
from typing import List, Optional

import numpy as np

from QEfficient.generation.base_infer import TextGeneration
from QEfficient.generation.spec_prefill import KeepConfig, SpecPrefillEngine
from QEfficient.utils import load_hf_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(
        "Speculative prefill validation (k=0) using spec_prefill + base prefill_from_ids."
    )
    parser.add_argument("--spec-qpc", required=True, help="Path to speculator QPC (with prefill_queries).")
    parser.add_argument("--base-qpc", required=True, help="Path to base model QPC.")
    parser.add_argument("--model-name", required=True, help="HF tokenizer card for both spec/base models.")
    parser.add_argument(
        "--prompt",
        default="This is a longer prompt to exercise multiple chunks. " * 8,
        help="Prompt string.",
    )
    parser.add_argument("--ctx-len", type=int, default=128, help="Context length used by both engines.")
    parser.add_argument("--gen-len", type=int, default=16, help="Decode steps for parity check.")
    parser.add_argument("--prompt-file", type=str, default=None,
                        help="Read the entire file as a single prompt (e.g., a 100k-token prompt). If set, overrides --prompt.")
    # Separate device groups for speculator and base engines
    parser.add_argument("--spec-device-ids", default="[0]",
                        help="Speculator device IDs, e.g. [0] or [0,1]")
    parser.add_argument("--base-device-ids", default="[0]",
                        help="Base model device IDs, e.g. [1] or [2,3]")
    parser.add_argument(
        "--keep-percentage",
        type=float,
        default=0.10,
        help="Percentage to keep (0..1) in k=0.",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=32, help="Chunk size when chunked keep is enabled."
    )
    parser.add_argument(
        "--no-chunk", action="store_true", help="Disable chunked keep; use per-token percentage."
    )
    args = parser.parse_args()

    os.environ.setdefault("QEFF_SPEC_DEBUG", "1")

    # ---- Parse device lists (accepts "[0,1]" or "0,1" or "[0]") ----
    def _parse_ids(s: Optional[str]) -> List[int]:
        if not s:
            return [0]
        parts = s.strip().strip("[]").split(",")
        ids = [int(p) for p in parts if p.strip() != ""]
        return ids if ids else [0]

    spec_device_ids: List[int] = _parse_ids(args.spec_device_ids)
    base_device_ids: List[int] = _parse_ids(args.base_device_ids)
    print(f"[devices] spec={spec_device_ids}  base={base_device_ids}")

    tokenizer = load_hf_tokenizer(pretrained_model_name_or_path=args.model_name)
    # Load prompt: either raw string from --prompt or full file content from --prompt-file
    if args.prompt_file:
        with open(args.prompt_file, "r", encoding="utf-8") as fh:
            prompt_text = fh.read()
    else:
        prompt_text = args.prompt
    spec = SpecPrefillEngine(
        tokenizer=tokenizer,
        qpc_path=args.spec_qpc,
        ctx_len=int(args.ctx_len),
        device_id=spec_device_ids,
    )
    base = TextGeneration(
        tokenizer=tokenizer,
        qpc_path=args.base_qpc,
        ctx_len=int(args.ctx_len),
        device_id=base_device_ids,
    )._qaic_model

    print("\n[1/2] Unit: base.prefill_from_ids parity (identity keep)")
    out_full, pos_ids, gen_len = base.run_prefill(
        [prompt_text], generation_len=int(args.gen_len), prefill_logit_bs=1
    )
    orig_seq_len = int(pos_ids.max())
    enc_full = tokenizer(prompt_text, return_tensors="np")
    ids_full = enc_full["input_ids"][:, :orig_seq_len]
    pos_full = np.arange(orig_seq_len, dtype=np.int64).reshape(1, -1)

    out_ids, seq_len_ret, padded_len, num_chunks = base.prefill_from_ids(
        ids_full, pos_full, prefill_logit_bs=1
    )
    assert "logits" in out_ids, "prefill_from_ids missing logits"
    assert seq_len_ret == orig_seq_len, f"orig_seq_len mismatch: {seq_len_ret} != {orig_seq_len}"
    print(
        f"[unit] OK  orig_seq_len={seq_len_ret}  padded_len={padded_len}  num_chunks={num_chunks}"
    )

    print("\n[2/2] Integration: spec prefill → importance → keep_idx → base.prefill_from_ids")
    keep_cfg = KeepConfig(
        strategy="percentage",
        percentage=float(args.keep_percentage),
        chunk=(not args.no_chunk),
        chunk_size=int(args.chunk_size),
    )
    ret = spec.prune_and_base_prefill(
        base_engine=base,
        prompt=prompt_text,
        pool_kernel_size=13,
        keep_cfg=keep_cfg,
        prefill_logit_bs=1,
    )
    print("[k=0 integration]", ret)


if __name__ == "__main__":
    main()

