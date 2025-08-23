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
    parser.add_argument("--device-ids", default="[0]", help="Device IDs like [0] or [0,1].")
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

    try:
        device_ids: Optional[List[int]] = [
            int(x) for x in args.device_ids.strip("[] ").split(",") if x.strip()
        ]
    except Exception:
        device_ids = [0]

    tokenizer = load_hf_tokenizer(pretrained_model_name_or_path=args.model_name)
    spec = SpecPrefillEngine(
        tokenizer=tokenizer,
        qpc_path=args.spec_qpc,
        ctx_len=int(args.ctx_len),
        device_id=device_ids,
    )
    base = TextGeneration(
        tokenizer=tokenizer,
        qpc_path=args.base_qpc,
        ctx_len=int(args.ctx_len),
        device_id=device_ids,
    )._qaic_model

    print("\n[1/2] Unit: base.prefill_from_ids parity (identity keep)")
    out_full, pos_ids, gen_len = base.run_prefill(
        [args.prompt], generation_len=int(args.gen_len), prefill_logit_bs=1
    )
    orig_seq_len = int(pos_ids.max())
    enc_full = tokenizer(args.prompt, return_tensors="np")
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

    print(
        "\n[2/2] Integration: spec prefill → importance → keep_idx → base.prefill_from_ids"
    )
    keep_cfg = KeepConfig(
        strategy="percentage",
        percentage=float(args.keep_percentage),
        chunk=(not args.no_chunk),
        chunk_size=int(args.chunk_size),
    )
    res = spec.prefill_and_score(
        args.prompt, pool_kernel_size=13, keep_cfg=keep_cfg
    )
    keep_idx = res["keep_idx"]
    orig_seq_len = res["orig_seq_len"]
    assert keep_idx.size > 0 and keep_idx[-1] == orig_seq_len - 1

    enc = tokenizer(args.prompt, return_tensors="np")
    final_ids = enc["input_ids"][:, keep_idx]
    final_pos = keep_idx.reshape(1, -1).astype(np.int64)

    t0 = time.perf_counter()
    out_full2, pos_ids2, _ = base.run_prefill(
        [args.prompt], generation_len=int(args.gen_len), prefill_logit_bs=1
    )
    ttft_baseline_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    out_pruned, seq_len_used, pad_len_pruned, num_chunks_pruned = base.prefill_from_ids(
        final_ids, final_pos, prefill_logit_bs=1
    )
    ttft_pruned_s = time.perf_counter() - t1

    print(
        f"[spec→base] orig_seq_len={orig_seq_len} kept={keep_idx.size} "
        f"TTFT baseline={ttft_baseline_s*1000:.1f}ms pruned={ttft_pruned_s*1000:.1f}ms "
        f"(pad_len={pad_len_pruned}, chunks={num_chunks_pruned})"
    )

    def _next_token_id(outputs: dict) -> int:
        logits = outputs["logits"]
        if logits.ndim == 2:
            logits = np.expand_dims(logits, 1)
        return int(logits.argmax(2)[0, 0])

    next_full = _next_token_id(out_full2)
    next_pruned = _next_token_id(out_pruned)
    print(
        f"[parity] first decode token match: {next_full == next_pruned} "
        f"({next_full} vs {next_pruned})"
    )


if __name__ == "__main__":
    main()

