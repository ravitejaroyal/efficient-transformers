# TTFT(spec_device) vs. TTFT(base_full) Investigation

## 1. Timing windows (code audit)
- `SpecPrefillEngine.prefill_and_score()` measures the speculative prefill wall clock as `t_run_prefill_s = t1 - t0` immediately after calling `self.run_prefill(...)`, while the subsequent assemble/score/select timers (`t_assemble_s`, `t_score_s`, `t_select_s`) carve out the host-side work.【F:QEfficient/generation/spec_prefill.py†L975-L1078】
- `prune_and_base_prefill()` maps the timers onto the user-facing TTFT metrics, setting `ttft_spec_device_s = t_run_prefill_s` and accumulating the host-side terms for `ttft_host_scoring_s`.【F:QEfficient/generation/spec_prefill.py†L1140-L1173】
- The baseline timings wrap the base engine calls: `ttft_baseline_s` measures `base_engine.run_prefill(...)`, and `ttft_base_pruned_only_s` captures `base_engine.prefill_from_ids(...)`.【F:QEfficient/generation/spec_prefill.py†L1204-L1218】
- Because `prefill_and_score()` starts timing before invoking `run_prefill`, the tokenizer padding/max-length expansion in `SpecPrefillEngine.run_prefill()` (including the `padding=True` bootstrap and `padding="max_length"` re-tokenization) executes inside the timed window for `TTFT(spec_device)`.【F:QEfficient/generation/spec_prefill.py†L187-L229】
- The base path mirrors that flow: `BaseInference.run_prefill()` performs the same two-stage tokenization before chunking, so the tokenizer cost is fully included in `TTFT(base_full)` as measured in the wrapper.【F:QEfficient/generation/base_infer.py†L691-L739】【F:QEfficient/generation/spec_prefill.py†L1204-L1209】

## 2. Speculative prefill I/O logging attempt
Running the provided command with `QEFF_SPEC_IO_TIMING=1 QEFF_SPEC_DEBUG=1` fails in this container because optional dependencies and the QAIC runtime are unavailable (`ModuleNotFoundError: evaluate` and "QAIC SDK is not installed" warning).【afb42a†L1-L12】 Without the QAIC stack or the actual QPC paths, the spec I/O counters are not emitted, so device↔host byte totals cannot be captured here.

| Category | Bytes | Time (ms) | Notes |
|----------|-------|-----------|-------|
| prefill_queries | — | — | Requires QAIC runtime to execute `self._session.run(...)` and populate `get_last_io_metrics()`; not available in this container.【F:QEfficient/generation/spec_prefill.py†L333-L378】 |
| past_key | — | — | Same limitation as above. |
| past_value | — | — | Spec path skips these outputs, but confirmation needs runtime logs.【F:QEfficient/generation/spec_prefill.py†L323-L354】 |
| logits | — | — | Would be reported per chunk when execution succeeds. |
| other | — | — | Aggregates any remaining bindings per `QAICInferenceSession.get_last_io_metrics()`.【F:QEfficient/generation/cloud_infer.py†L260-L295】 |

To collect the numbers, rerun the command on a QAIC-enabled host with the dependencies pre-installed (e.g., `pip install evaluate` and the QAIC SDK), preserving the `QEFF_SPEC_IO_TIMING=1 QEFF_SPEC_DEBUG=1` flags.

## 3. Binding differences vs. base
- The spec path always binds an explicit FP32 buffer for `prefill_queries` before entering the chunk loop and keeps it active across all chunks, as noted by the inline comment and debug print.【F:QEfficient/generation/spec_prefill.py†L203-L223】 Because the placeholder is `np.float32`, every element consumes four bytes, and the runtime must DMA that tensor back on each chunk whenever the QPC marks it non-partial.
- During chunking the engine re-enables retained-state key outputs only on the final chunk while keeping `prefill_queries` bound, which forces repeated transfers of that tensor even when intermediate retained-state keys stay skipped.【F:QEfficient/generation/spec_prefill.py†L333-L383】
- The base engine only binds a logits placeholder; there is no `prefill_queries` output, so its chunk loop transfers substantially less data per iteration.【F:QEfficient/generation/base_infer.py†L726-L777】 This asymmetric binding pattern is a plausible explanation for the larger `TTFT(spec_device)` once the actual DMA volumes are confirmed.
- The code path that collects tensors for host scoring expects `prefill_queries` with shape `[L, H, D]`, underscoring that the transfer grows linearly with the number of layers retained and heads per layer.【F:QEfficient/generation/spec_prefill.py†L480-L501】 Given float32 storage, per-chunk traffic is `L × H × D × 4` bytes.

## 4. Chunking and sequence length parity
- Both spec and base engines compute `num_chunks = -(padded_len // -self._prefill_seq_len)` and round the padded length to the specialization multiple before running the chunk loop.【F:QEfficient/generation/spec_prefill.py†L195-L200】【F:QEfficient/generation/base_infer.py†L711-L719】 Thus, assuming the spec and base QPCs were compiled with the same `_prefill_seq_len`, they will iterate over the same number of chunks for a given padded prompt length `S` (here `S=10741` from the supplied log snippet). Without runtime access, `_prefill_seq_len` must be read from the QPC metadata (exposed via `SpecPrefillEngine._prefill_seq_len`) during execution.
- The spec loop tracks per-chunk timings and can print the top three slowest slices when `QEFF_SPEC_DEBUG` is enabled (`[spec:prefill] ... worst=[(chunk, seconds), ...]`).【F:QEfficient/generation/spec_prefill.py†L333-L378】【F:QEfficient/generation/spec_prefill.py†L400-L413】 Inspect those diagnostics on a hardware run to spot outliers.

## 5. Device specialization separation
- `QAICInferenceSession.__init__` loads `allowed_shapes`, builds the `binding_index_map`, and, when multiple device IDs are supplied, programs the devMapping string before loading/activating the QPC.【F:QEfficient/generation/cloud_infer.py†L63-L114】 That initialization happens independently for the spec and base engines, so the CLI assignment of `[0..7]` vs. `[8..15]` ensures disjoint device groups.

## 6. Interpretation and next steps
- `TTFT(spec_device)` strictly captures the first pass through `SpecPrefillEngine.run_prefill()`, which includes tokenizer work, buffer binding, the chunk loop, and the repeated `prefill_queries` DMA enforced by `self._session.set_buffers({"prefill_queries": ...})`. No host-side scoring time is mixed in because `ttft_spec_device_s` is assigned directly from `t_run_prefill_s` before assembly/scoring/select steps begin.【F:QEfficient/generation/spec_prefill.py†L975-L1078】【F:QEfficient/generation/spec_prefill.py†L1140-L1173】
- The repeated transfer of FP32 `prefill_queries` on every chunk is the largest structural difference versus the base engine, which only returns logits placeholders. Once actual IO totals are gathered, compare the aggregate `prefill_queries` bytes to the base logits traffic to confirm whether the bandwidth explains the ~4× slowdown. If the numbers do not align, inspect chunk-level outliers (per the `[spec:prefill] ... worst=` log) and the `"other"` IO bucket to check for additional buffers being shuttled unexpectedly.【F:QEfficient/generation/spec_prefill.py†L333-L378】【F:QEfficient/generation/cloud_infer.py†L260-L295】 Re-running with the same flags on QAIC hardware is the next required step to obtain those measurements.
