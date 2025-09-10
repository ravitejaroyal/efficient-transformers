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
    parser.add_argument(
        "--look-ahead",
        type=int,
        default=0,
        help="Number of decode anchors for look-ahead (default: 0).",
    )
    parser.add_argument(
        "--layers-for-scoring",
        default="all",
        choices=["all", "last4", "last1"],
        help="Which layers to use for host scoring aggregation."
    )
    # --- simple policy gates ---
    parser.add_argument(
        "--min-spec-len",
        type=int,
        default=4096,
        help="Skip speculative prefill when prompt length S < this value (default: 4096).",
    )
    parser.add_argument(
        "--max-keep-for-spec",
        type=float,
        default=0.80,
        help="Skip speculative prefill when keep percentage is too high (default: 0.80 == keep >80%).",
    )
    # Optional: print and/or save pruned-path base model output (for verification)
    parser.add_argument(
        "--print-pruned-output",
        action="store_true",
        help="Print the base model decode output after pruned prefill (speculative path).",
    )
    parser.add_argument(
        "--save-pruned-output",
        type=str,
        default=None,
        help="If set, save the pruned-path base decode text to this file.",
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
    # Set which layers to keep for scoring
    spec._layers_sel = args.layers_for_scoring
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
    # --- compute prompt length S once for gating ---
    enc_for_len = tokenizer(prompt_text, return_tensors="np")
    S = int(enc_for_len["input_ids"].shape[1])
    p = float(args.keep_percentage)
    length_gate = (S < int(args.min_spec_len))
    high_keep_gate = (p > float(args.max_keep_for_spec))

    if length_gate or high_keep_gate:
        reason = "short_prompt" if length_gate else "high_keep"
        print(f"[gate] SKIP SpecPrefill (reason={reason})  S={S}  min_spec_len={args.min_spec_len}  keep={p:.2f}  max_keep_for_spec={args.max_keep_for_spec:.2f}")
        # Baseline only (faithful TTFT for comparison)
        t0 = time.time()
        out_base_full, pos_full, _ = base.run_prefill(
            [prompt_text], generation_len=int(args.gen_len), prefill_logit_bs=1
        )
        ttft_base_full_s = time.time() - t0
        print("\n[TTFT]")
        print(f" S={S} kept={S} (no pruning)")
        print(" base_full={:.1f} ms | spec_device+host=SKIPPED | base_prefill_pruned=SKIPPED | speculative_total=SKIPPED".format(ttft_base_full_s*1000.0))
        # No speculative output in this branch
        return
    else:
        print(f"[gate] RUN SpecPrefill  S={S}  keep={p:.2f}  L_sel={args.layers_for_scoring}")

    ret = spec.prune_and_base_prefill(
        base_engine=base,
        prompt=prompt_text,
        pool_kernel_size=13,
        keep_cfg=keep_cfg,
        prefill_logit_bs=1,
        layers_sel=args.layers_for_scoring,
        gen_len=int(args.gen_len) if args.gen_len is not None else None,
        look_ahead=int(args.look_ahead),
    )

    # --- Pretty, accurate TTFT breakdown ---
    try:
        S = ret.get("S", None)
        kept = ret.get("kept", None)
        ttft_base_full_ms = ret["ttft_baseline_s"] * 1000.0
        ttft_spec_only_ms = ret["ttft_spec_only_s"] * 1000.0  # spec device prefill + host assemble/score
        ttft_base_pruned_ms = ret["ttft_base_pruned_only_s"] * 1000.0
        ttft_spec_total_ms = ret["ttft_speculative_s"] * 1000.0

        print("\n[TTFT]")
        if S is not None and kept is not None:
            print(f"  S={S}  kept={kept}")
        print(
            "  base_full={:.1f} ms | spec_device+host={:.1f} ms | base_prefill_pruned={:.1f} ms | speculative_total={:.1f} ms"
            .format(ttft_base_full_ms, ttft_spec_only_ms, ttft_base_pruned_ms, ttft_spec_total_ms)
        )
    except Exception as e:
        print(f"[TTFT] warning: could not format timing breakdown ({e}); raw ret: {ret}")

    # --- Optional: print/save pruned-path base decode output for verification ---
    gen_txt = ret.get("generated_text_pruned", None)
    if gen_txt is not None:
        if args.print_pruned_output:
            # Print a trimmed preview (avoid flooding console on long outputs)
            preview = gen_txt if len(gen_txt) <= 400 else (gen_txt[:400] + " …")
            print("\n[pruned:base:output]")
            print(preview)
        if args.save_pruned_output:
            try:
                with open(args.save_pruned_output, "w", encoding="utf-8") as fh:
                    fh.write(gen_txt)
                print(f"[pruned:base:output] saved to {args.save_pruned_output}")
            except Exception as e:
                print(f"[pruned:base:output] failed to save to {args.save_pruned_output}: {e}")
    else:
        if args.print_pruned_output or args.save_pruned_output:
            print("[pruned:base:output] no generated_text_pruned present; ensure --gen-len > 0 was passed through.")


if __name__ == "__main__":
    main()

