from __future__ import annotations

import argparse
import os
import time
from typing import List, Optional, Tuple

import numpy as np
import evaluate  # Optional ROUGE (only runs when refs are available)
from datasets import load_dataset  # <-- dataset support

from QEfficient.generation.base_infer import TextGeneration
from QEfficient.generation.spec_prefill import KeepConfig, SpecPrefillEngine
from QEfficient.utils import load_hf_tokenizer


# ---------- Apples-to-apples helpers (opt-in) ----------

def build_messages(user_input: str, context: str) -> List[dict]:
    """Mirror the GPU script's message construction."""
    system_content = (
        "You are a careful assistant that writes concise, faithful summaries of long U.S. "
        "government reports. Capture the main purpose, scope, key findings, and recommendations. "
        "Avoid speculation."
    )
    user_content = f"{user_input}\n\n[DOCUMENT START]\n{context}\n[DOCUMENT END]"
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


def compute_prompt_budget_from_ctx(
    ctx_len: int, max_new_tokens: int, *, buffer_tokens: int = 64
) -> int:
    """
    Deterministic budget using the compile-time ctx_len:
    budget = ctx_len - max_new_tokens - buffer
    """
    max_pos = int(ctx_len) if ctx_len and ctx_len > 0 else 8192
    budget = max_pos - int(max_new_tokens) - int(buffer_tokens)
    return max(512, budget)


def truncate_instr_ctx_like_gpu(
    tokenizer,
    instr: str,
    ctx: str,
    *,
    prompt_budget: int,
    head_tail: bool,
) -> Tuple[str, str]:
    """
    Reproduce the GPU script’s pre-templating truncation accounting.
    Returns possibly-truncated (instr, ctx). (We leave instr unchanged here; GPU path truncates ctx.)
    """
    # Fixed wrappers (same as the GPU script)
    sys_txt = (
        "You are a careful assistant that writes concise, faithful summaries of long U.S. "
        "government reports. Capture the main purpose, scope, key findings, and recommendations. "
        "Avoid speculation."
    )
    pre_ctx = "{instr}\n\n[DOCUMENT START]\n"
    post_ctx = "\n[DOCUMENT END]"

    # Tokenize wrappers for truncation accounting
    sys_ids = tokenizer(sys_txt, add_special_tokens=False).input_ids
    pre_ids = tokenizer(pre_ctx.format(instr=""), add_special_tokens=False).input_ids
    post_ids = tokenizer(post_ctx, add_special_tokens=False).input_ids

    instr_ids = tokenizer(instr, add_special_tokens=False).input_ids
    ctx_ids = tokenizer(ctx, add_special_tokens=False).input_ids

    overhead = len(sys_ids) + len(pre_ids) + len(post_ids) + len(instr_ids)
    allowed_ctx = max(0, prompt_budget - overhead)

    if len(ctx_ids) > allowed_ctx:
        if head_tail and allowed_ctx > 1:
            head = allowed_ctx // 2
            tail = allowed_ctx - head
            ctx_ids = (
                ctx_ids[:head]
                + tokenizer("\n", add_special_tokens=False).input_ids
                + ctx_ids[-tail:]
            )
        else:
            ctx_ids = ctx_ids[:allowed_ctx]
        ctx = tokenizer.decode(ctx_ids, skip_special_tokens=True)

    return instr, ctx


def apply_hf_format_to_pair(
    tokenizer,
    instr: str,
    ctx: str,
) -> str:
    """
    Chat-template the (instr, ctx) pair into a STRING (not tokenized).
    Trim trailing whitespace to avoid newline sink as the last token.
    """
    messages = build_messages(instr, ctx)
    formatted_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,               # IMPORTANT: return string; QEff re-tokenizes internally
        add_generation_prompt=True,   # same as GPU path
    )
    return formatted_text.rstrip()    # avoid trailing '\n' / spaces at end


# -------------------------------------------------------


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
    # CHANGED: default=None so "entire output" mode kicks in when omitted
    parser.add_argument(
        "--gen-len",
        type=int,
        default=None,
        help="Decode steps. If omitted, decode until EOS or remaining context limit.",
    )
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
    parser.add_argument(
        "--refs-file",
        type=str,
        default=None,
        help="Optional: path to a text file with one reference per line, aligned with --prompt-file or the single --prompt.",
    )
    # Apples-to-apples (GPU parity) – opt-in
    parser.add_argument(
        "--match-hf-format",
        action="store_true",
        help="Apply the same system prompt, wrappers, pre-templating truncation, and chat template as the GPU script.",
    )
    parser.add_argument(
        "--truncate-to",
        type=int,
        default=None,
        help="Token budget for prompt (instruction+context) BEFORE chat templating. If None, derive from --ctx-len.",
    )
    parser.add_argument(
        "--head-tail",
        action="store_true",
        help="When truncating, keep head and tail (middle cut) instead of head-only.",
    )
    # ------------------- Dataset options (LongBench-like) -------------------
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Optional HF dataset path (e.g., 'THUDM/LongBench'). If set, prompts+refs are loaded from dataset (ignore --prompt/--prompt-file).",
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default="gov_report",
        help="Dataset configuration name (e.g., 'gov_report' for LongBench).",
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default="test",
        help="Dataset split to load (e.g., 'test').",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=50,
        help="-1 for all; otherwise take the first N examples from the dataset split.",
    )
    # Field mapping for dataset → (instruction, context, reference)
    parser.add_argument(
        "--input-field",
        type=str,
        default="input",
        help="Dataset field that contains the instruction/user input (default: 'input' for LongBench).",
    )
    parser.add_argument(
        "--context-field",
        type=str,
        default="context",
        help="Dataset field that contains the document/context (default: 'context' for LongBench).",
    )
    parser.add_argument(
        "--reference-field",
        type=str,
        default="answers",
        help="Dataset field that contains reference(s); string or list[str] (default: 'answers' for LongBench).",
    )
    args = parser.parse_args()

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
    # (Matches GPU habit) Ensure we have a PAD if model lacks one
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ------------------- Build prompts + references -------------------
    prompts: List[Tuple[str, str]] = []
    refs: Optional[List[str]] = None

    if args.dataset:
        # Load dataset (e.g., THUDM/LongBench gov_report)
        ds = load_dataset(args.dataset, args.dataset_config, split=args.dataset_split)
        if args.max_examples != -1:
            ds = ds.select(range(min(len(ds), args.max_examples)))
        prompts = []
        refs = []
        # Extract fields; LongBench gov_report: input (instr), context (doc), answers (list[str])
        for ex in ds:
            instr = (ex.get(args.input_field) or "") if args.input_field else ""
            ctx = (ex.get(args.context_field) or "") if args.context_field else ""
            ref_raw = ex.get(args.reference_field, "")
            if isinstance(ref_raw, list) and ref_raw:
                ref = ref_raw[0]
            elif isinstance(ref_raw, str):
                ref = ref_raw
            else:
                ref = ""
            prompts.append((instr, ctx))
            refs.append(ref)
    else:
        # Fallback to CLI prompt(s)
        if args.prompt_file:
            with open(args.prompt_file, "r", encoding="utf-8") as fh:
                plain_prompts = [line.strip() for line in fh if line.strip()]
        else:
            plain_prompts = [args.prompt]
        prompts = [("", ptxt) for ptxt in plain_prompts]
        if args.refs_file:
            try:
                with open(args.refs_file, "r", encoding="utf-8") as rf:
                    refs = [line.strip() for line in rf]
            except Exception as e:
                print(f"[ROUGE] warning: failed to read --refs-file ({e}); skipping ROUGE.")
                refs = None

    # Apples-to-apples: compute prompt budget (outside timing paths)
    prompt_budget: Optional[int] = None
    if args.match_hf_format:
        prompt_budget = (
            int(args.truncate_to)
            if args.truncate_to is not None
            else compute_prompt_budget_from_ctx(args.ctx_len, args.gen_len or 0, buffer_tokens=64)
        )

    def format_prompt(instr: str, ctx: str) -> str:
        if args.match_hf_format:
            assert prompt_budget is not None
            instr_tr, ctx_tr = truncate_instr_ctx_like_gpu(
                tokenizer, instr, ctx, prompt_budget=prompt_budget, head_tail=args.head_tail
            )
            return apply_hf_format_to_pair(tokenizer, instr_tr, ctx_tr)
        plain = ctx if not instr else f"{instr}\n\n{ctx}"
        return plain.rstrip()

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

    if os.getenv("QEFF_SPEC_DEBUG", "") and prompts:
        debug_prompt_text = format_prompt(*prompts[0])
        print("\n[1/2] Unit: base.prefill_from_ids parity (identity keep)")
        gen_len_pass = args.gen_len if args.gen_len is not None else None
        out_full, pos_ids, gen_len = base.run_prefill(
            [debug_prompt_text], generation_len=gen_len_pass, prefill_logit_bs=1
        )
        import numpy as _np  # avoid shadowing

        orig_seq_len = int(pos_ids.max())
        enc_full = tokenizer(debug_prompt_text, return_tensors="np")
        ids_full = enc_full["input_ids"][:, :orig_seq_len]
        pos_full = _np.arange(orig_seq_len, dtype=_np.int64).reshape(1, -1)

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

    # Collect predictions and aligned references for optional ROUGE scoring (outside timing)
    preds: List[str] = []
    aligned_refs: List[str] = []

    # --- loop over prompts ---
    for idx, pair in enumerate(prompts):
        instr, ctx = pair

        prompt_text = format_prompt(instr, ctx)

        # --- compute prompt length S once for gating ---
        enc_for_len = tokenizer(prompt_text, return_tensors="np")
        S = int(enc_for_len["input_ids"].shape[1])
        p = float(args.keep_percentage)
        length_gate = (S < int(args.min_spec_len))
        high_keep_gate = (p > float(args.max_keep_for_spec))

        if length_gate or high_keep_gate:
            reason = "short_prompt" if length_gate else "high_keep"
            print(
                f"[gate] SKIP SpecPrefill (reason={reason})  S={S}  min_spec_len={args.min_spec_len}  "
                f"keep={p:.2f}  max_keep_for_spec={args.max_keep_for_spec:.2f}"
            )
            # Baseline only (faithful TTFT for comparison)
            t0 = time.time()
            gen_len_pass = args.gen_len if args.gen_len is not None else None
            base.run_prefill(
                [prompt_text], generation_len=gen_len_pass, prefill_logit_bs=1
            )
            ttft_base_full_s = time.time() - t0
            print("\n[TTFT]")
            print(f" S={S} kept={S} (no pruning)")
            print(
                " base_full={:.1f} ms | spec_device=SKIPPED | host_scoring=SKIPPED | "
                "base_prefill_pruned=SKIPPED | speculative_total=SKIPPED".format(ttft_base_full_s * 1000.0)
            )
            continue
        else:
            print(f"[gate] RUN SpecPrefill  S={S}  keep={p:.2f}  L_sel={args.layers_for_scoring}")

        # Speculative prefill (timings inside SpecPrefillEngine remain unchanged)
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

        # --- Pretty, accurate TTFT breakdown (from ret) ---
        try:
            S = ret.get("S", None)
            kept = ret.get("kept", None)
            ttft_base_full_ms = ret["ttft_baseline_s"] * 1000.0
            ttft_spec_device_ms = ret["ttft_spec_device_s"] * 1000.0
            ttft_host_scoring_ms = ret["ttft_host_scoring_s"] * 1000.0
            ttft_base_pruned_ms = ret["ttft_base_pruned_only_s"] * 1000.0
            ttft_spec_total_ms = ret["ttft_speculative_s"] * 1000.0

            print("\n[TTFT]")
            if S is not None and kept is not None:
                print(f"  S={S}  kept={kept}")
            print(
                "  base_full={:.1f} ms | spec_device={:.1f} ms | host_scoring={:.1f} ms | "
                "base_prefill_pruned={:.1f} ms | speculative_total={:.1f} ms".format(
                    ttft_base_full_ms,
                    ttft_spec_device_ms,
                    ttft_host_scoring_ms,
                    ttft_base_pruned_ms,
                    ttft_spec_total_ms,
                )
            )
        except Exception as e:
            print(f"[TTFT] warning: could not format timing breakdown ({e}); raw ret: {ret}")

        # --- Optional: print/save pruned-path base decode output for verification ---
        gen_txt = ret.get("generated_text_pruned", None)
        if gen_txt is not None:
            preds.append(gen_txt.strip())
            if refs is not None and idx < len(refs):
                aligned_refs.append(refs[idx])

            if args.print_pruned_output:
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

    # ---- ROUGE scoring when dataset provided (or aligned refs available) ----
    try:
        if args.dataset and refs is not None and len(preds) == len(refs) and len(preds) > 0:
            rouge = evaluate.load("rouge")
            scores = rouge.compute(
                predictions=preds,
                references=refs,
                rouge_types=["rouge1", "rouge2", "rougeL", "rougeLsum"],
            )
            print("\n[ROUGE] results over {} example(s):".format(len(preds)))
            print(
                "  rouge1={:.4f}  rouge2={:.4f}  rougeL={:.4f}  rougeLsum={:.4f}".format(
                    scores.get("rouge1", 0.0),
                    scores.get("rouge2", 0.0),
                    scores.get("rougeL", 0.0),
                    scores.get("rougeLsum", 0.0),
                )
            )
        elif not args.dataset and aligned_refs and len(preds) == len(aligned_refs):
            rouge = evaluate.load("rouge")
            scores = rouge.compute(
                predictions=preds,
                references=aligned_refs,
                rouge_types=["rouge1", "rouge2", "rougeL", "rougeLsum"],
            )
            print("\n[ROUGE] results over {} example(s):".format(len(preds)))
            print(
                "  rouge1={:.4f}  rouge2={:.4f}  rougeL={:.4f}  rougeLsum={:.4f}".format(
                    scores.get("rouge1", 0.0),
                    scores.get("rouge2", 0.0),
                    scores.get("rougeL", 0.0),
                    scores.get("rougeLsum", 0.0),
                )
            )
        elif refs is not None:
            n_refs = len(refs)
            print(f"[ROUGE] warning: could not compute ROUGE (preds={len(preds)} vs refs={n_refs}).")
    except Exception as e:
        print(f"[ROUGE] warning: failed to compute ROUGE ({e}).")


if __name__ == "__main__":
    main()
