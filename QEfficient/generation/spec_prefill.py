from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
import os

from QEfficient.generation.base_infer import write_io_files
from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.utils.logging_utils import logger

# Changes:
#  * multi-anchor look-ahead scoring and averaging
#  * block Top-K selection at call-site
#  * vectorized head scoring with layer streaming
#  * clarified "next predicted token" log and added debug prints
@dataclass
class KeepConfig:
    strategy: str = "percentage"  # only "percentage" in this step
    percentage: float = 0.1  # 10%
    chunk: bool = True
    chunk_size: int = 32


class SpecPrefillEngine:
    """
    Mirror QEfficient/generation/text_generation_inference.py::run_prefill (variable names & logic)
    for a SPECULATOR QPC. Reuses QAICInferenceSession and exposes a helper to collect tensors
    needed for speculative scoring (prefill_queries and per-layer past_key.*_RetainedState).
    """

    def __init__(
        self,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        qpc_path: str,
        full_batch_size: Optional[int] = None,
        ctx_len: Optional[int] = None,
        device_id: Optional[List[int]] = None,
        enable_debug_logs: bool = False,
        write_io_dir: Optional[str] = None,
        is_tlm: Optional[int] = None,
    ) -> None:
        self._ctx_len = ctx_len
        self._write_io_dir = write_io_dir
        self.is_tlm = is_tlm
        # optional look-ahead anchors (default: disabled)
        self._look_ahead = 0
        self._last_outputs = None

        # Load QPC
        self._session = QAICInferenceSession(
            qpc_path, device_id, enable_debug_logs=enable_debug_logs
        )

        # Fetch variables from the QPC
        self._vocab_size = self._fetch_vocab_size()
        self.batch_size, self._prefill_seq_len = (
            self._fetch_batch_size_prefill_seq_len()
        )
        self._decode_seq_len = self._fetch_decode_seq_len()
        self.full_batch_size = (
            full_batch_size if full_batch_size else self._fetch_full_batch_size()
        )

        # Initialize storage variables
        self.batch_index = None
        self._prompt_to_lora_id_mapping_prefill = None
        self._prompt_to_lora_id_mapping_decode = None
        self.generated_ids = None
        self.decode_input_ids = None
        self.decode_pos_ids = None
        self.generation_len = None

        self.tokenizer = tokenizer
        self._set_tokenizer_params()

        # Which layers to fetch for host scoring ("all", "last4", "last1"); default "all"
        self._layers_sel: str = "all"
        # Actual layer indices kept after skip_buffers (computed in run_prefill)
        self._kept_layers: List[int] = []

        # Cache for first prefill pass: assembled later in run_prefill()
        # keys: chunks_keys(List[List[np.ndarray]]), chunks_pos(List[np.ndarray]),
        #       chunks_ids(List[np.ndarray]), Q_final(np.ndarray)
        self._prefill_cache: Optional[Dict[str, Any]] = None

        # Speculative prefill uses past_key.* for scoring; skip past_* inputs and past_value outputs.
        past_inputs = [n for n in self._session.input_names if n.startswith("past_")]
        past_outs = [n for n in self._session.output_names if n.startswith("past_value.")]
        self._session.skip_buffers(past_inputs + past_outs)

    def reset_cache(self) -> None:
        """Clear per-prompt state so each new prompt actually runs prefill."""
        # force a fresh prefill on the next call
        self._prefill_cache = None
        self._last_outputs = None
        # reset I/O tallies so [spec:io-total] is per-prompt (not cumulative)
        try:
            self._prefill_io_totals = {
                k: {"bytes": 0, "ms": 0.0}
                for k in ("prefill_queries", "past_key", "past_value", "logits", "other")
            }
            self._prefill_io_total_ms = 0.0
        except Exception:
            pass
        # ensure kept layers are recomputed by run_prefill()
        self._kept_layers = []

    def _set_tokenizer_params(self):
        """
        Sets the tokenizer parameters for the model.
        """
        if self.tokenizer.padding_side != "right":
            logger.warning(
                "Please use padding_side='right' while initializing the tokenizer"
            )
            self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    # --- session probing helpers (runtime parity) ---
    def _fetch_vocab_size(self):
        """Fetches and caches the vocabulary size from the session's allowed shapes."""
        if getattr(self, "_vocab_size", None) is not None:
            return self._vocab_size
        if self._session.allowed_shapes:
            self._vocab_size = [
                x[self._session.binding_index_map["logits"]]
                for x in self._session.allowed_shapes
            ][0][1][2]
        else:
            self._vocab_size = self._session.bindings[
                self._session.binding_index_map["logits"]
            ].dims[2]
        return self._vocab_size

    def _fetch_batch_size_prefill_seq_len(self):
        """Fetches batch size and prefill sequence length from the session."""
        if self._session.allowed_shapes:
            batch_size = max(
                [
                    x[self._session.binding_index_map["input_ids"]][1][0]
                    for x in self._session.allowed_shapes
                ]
            )
            prefill_seq_len = max(
                [
                    x[self._session.binding_index_map["input_ids"]][1][1]
                    for x in self._session.allowed_shapes
                ]
            )
        else:
            batch_size, prefill_seq_len = self._session.bindings[
                self._session.binding_index_map["input_ids"]
            ].dims
        return batch_size, prefill_seq_len

    def _fetch_decode_seq_len(self):
        """Fetches decode sequence length from the session."""
        decode_seq_len = None
        if self._session.allowed_shapes:
            decode_seq_len = min(
                [
                    x[self._session.binding_index_map["input_ids"]][1][1]
                    for x in self._session.allowed_shapes
                ]
            )
        return decode_seq_len

    def _fetch_full_batch_size(self):
        """Fetches full batch size from the session."""
        full_batch_size = None
        if "batch_index" in self._session.binding_index_map:
            if self._session.allowed_shapes:
                full_batch_size, _ = [
                    x[self._session.binding_index_map["batch_index"]][1][0]
                    for x in self._session.allowed_shapes
                ]
            else:
                full_batch_size, _ = self._session.bindings[
                    self._session.binding_index_map["batch_index"]
                ].dims
        return full_batch_size

    def _fetch_generation_len(self, generation_len, max_gen_len):
        """Fetches the generation length for the model."""
        if generation_len is None:
            if self._ctx_len is None:
                raise ValueError("At least one of ctx_len or generation_len is needed")
            generation_len = max_gen_len
        elif generation_len > max_gen_len:
            logger.warning(
                "Passed generation_len is greater than allowed length. "
                "Make sure this model supports sliding window, such as Mistral"
            )
        if generation_len <= 0:
            raise ValueError("generation length should be greater than zero")
        return generation_len

    # --- PREFILL: identical variable names & logic ---
    def run_prefill(
        self, prompt, generation_len, prefill_logit_bs=1, decode_batch_id=None
    ):
        """Runs prefill for a given prompt and generation length."""
        # ---- one-time wall clock for the whole spec prefill ----
        t_prefill_start = time.perf_counter()

        inputs = self.tokenizer(prompt, return_tensors="np", padding=True)
        position_ids = inputs["attention_mask"].sum(1, keepdims=True)
        padded_len = inputs["input_ids"].shape[1]
        num_chunks = -(padded_len // -self._prefill_seq_len)
        padded_len = num_chunks * self._prefill_seq_len

        max_gen_len = self._ctx_len - position_ids.max()
        generation_len = self._fetch_generation_len(generation_len, max_gen_len)
        # Bind persistent output buffers for logits so the runtime always sees a real buffer.
        logits_out_placeholder = np.zeros(
            (prefill_logit_bs, 1, self._vocab_size), dtype=np.float32
        )
        self._session.set_buffers({"logits": logits_out_placeholder})

        # Re-bind 'prefill_queries' for ALL prefill chunks (required for some QPCs that mark it non-partial).
        pq_idx = self._session.binding_index_map.get("prefill_queries", None)
        try:
            if pq_idx is not None:
                if self._session.allowed_shapes:
                    dims = list(self._session.allowed_shapes[0][pq_idx][1])  # prefill spec = 0
                else:
                    dims = list(self._session.bindings[pq_idx].dims)
                pq_placeholder = np.empty(tuple(dims), dtype=np.float32)
                self._session.set_buffers({"prefill_queries": pq_placeholder})
                if os.getenv("QEFF_SPEC_DEBUG", ""):
                    print(f"[spec:gate] prefill_queries bound with dims={tuple(dims)} dtype=float32", flush=True)
        except Exception:
            pass

        inputs = self.tokenizer(
            prompt, return_tensors="np", padding="max_length", max_length=padded_len
        )
        inputs["position_ids"] = np.where(
            inputs.pop("attention_mask"), np.arange(padded_len), -1
        )
        inputs.pop("token_type_ids", None)

        if decode_batch_id is not None:
            inputs["batch_index"] = decode_batch_id
        if self.is_tlm:
            inputs["num_logits_to_keep"] = np.zeros((1, 1))

        if self._prompt_to_lora_id_mapping_prefill:
            if self.full_batch_size:
                inputs["lora_ids"] = np.array(
                    self._prompt_to_lora_id_mapping_prefill.popleft(), dtype=np.int64
                ).reshape(1, 1)
            else:
                batch_lora_ids = [
                    self._prompt_to_lora_id_mapping_prefill.popleft()
                    for i in range(self.batch_size)
                ]
                inputs["lora_ids"] = np.array(batch_lora_ids, dtype=np.int64).reshape(
                    self.batch_size, 1
                )

        # ---- Debug: dump available outputs/bindings so we can see retained-state names ----
        if os.getenv("QEFF_SPEC_DEBUG", ""):
            try:
                outs = list(getattr(self._session, "output_names", []))
                print(
                    f"[spec:debug] output_names({len(outs)}): {outs[:20]}{' …' if len(outs)>20 else ''}",
                    flush=True,
                )
                try:
                    bind_names = [b.name for b in getattr(self._session, "bindings", [])]
                    print(
                        f"[spec:debug] bindings({len(bind_names)}): {bind_names[:20]}{' …' if len(bind_names)>20 else ''}",
                        flush=True,
                    )
                except Exception:
                    pass
            except Exception:
                pass

        # Determine which retained-state key bindings exist and which to keep
        pk_names = [
            n
            for n in self._session.output_names
            if n.startswith("past_key.") and n.endswith("_RetainedState")
        ]
        present_layers = sorted(
            {int(n.split(".")[1].split("_")[0]) for n in pk_names}
        )
        if not present_layers:
            raise RuntimeError(
                "[spec] No past_key.*_RetainedState outputs found in QPC; cannot harvest keys."
            )

        # Decide which layer indices to keep
        if self._layers_sel == "last1":
            keep_set = {max(present_layers)}
        elif self._layers_sel == "last4":
            max_layer = max(present_layers)
            keep_set = {li for li in present_layers if li >= max_layer - 3}
        else:
            keep_set = set(present_layers)

        # Build binding-name lists for skip/enable
        keep_bindings = [f"past_key.{li}_RetainedState" for li in keep_set]
        skip_bindings = [
            f"past_key.{li}_RetainedState" for li in present_layers if li not in keep_set
        ]

        # Skip now; kept bindings will be re-enabled on the final chunk
        if skip_bindings:
            self._session.skip_buffers(skip_bindings)
        if keep_bindings:
            self._session.skip_buffers(keep_bindings)

        self._kept_layers = sorted(list(keep_set))
        if os.getenv("QEFF_SPEC_DEBUG", ""):
            try:
                print(
                    f"[spec] layers_sel={self._layers_sel} kept_layers={self._kept_layers}",
                    flush=True,
                )
            except Exception:
                pass


        # Collect per-chunk tensors for host scoring
        chunks_keys: List[List[np.ndarray]] = []
        chunks_pos: List[np.ndarray] = []
        chunks_ids: List[np.ndarray] = []
        outputs_last = None
        per_chunk_times: List[Tuple[int, float]] = []
        io_totals = {k: {"bytes": 0, "ms": 0.0} for k in (
            "prefill_queries",
            "past_key",
            "past_value",
            "logits",
            "other",
        )}
        io_total_ms = 0.0

        want_logits = bool(os.getenv("QEFF_SPEC_DEBUG", ""))

        for i in range(num_chunks):
            chunk_inputs = inputs.copy()
            chunk_inputs["input_ids"] = inputs["input_ids"][
                :, i * self._prefill_seq_len : (i + 1) * self._prefill_seq_len
            ]
            chunk_inputs["position_ids"] = inputs["position_ids"][
                :, i * self._prefill_seq_len : (i + 1) * self._prefill_seq_len
            ]
            last = (i == num_chunks - 1)

            # Enable heavy outputs only for the final chunk: retained-state keys (prefill_queries stays bound)
            if last:
                names_to_enable: List[str] = []
                if keep_bindings:
                    names_to_enable += keep_bindings
                if names_to_enable:
                    e0 = time.perf_counter()
                    try:
                        self._session.enable_outputs(names_to_enable)
                    except Exception as e:
                        raise RuntimeError(f"[spec] enable_outputs failed: {e}")
                    e_ms = (time.perf_counter() - e0) * 1000.0
                    if os.getenv("QEFF_SPEC_IO_TIMING", "") or os.getenv("QEFF_SPEC_DEBUG", ""):
                        print(f"[spec:io] enable_outputs_ms={e_ms:.3f}", flush=True)

            t0 = time.perf_counter()
            outputs = self._session.run(chunk_inputs)
            dt = time.perf_counter() - t0
            per_chunk_times.append((i, dt))
            # accumulate device->host IO for this chunk
            io = self._session.get_last_io_metrics()
            cats = io.get("categories", {})
            for k in io_totals.keys():
                v = cats.get(k, {})
                io_totals[k]["bytes"] += int(v.get("bytes", 0))
                io_totals[k]["ms"] += float(v.get("ms", 0.0))
            io_total_ms += float(io.get("total_ms", 0.0))
            if os.getenv("QEFF_SPEC_IO_TIMING", "") or os.getenv("QEFF_SPEC_DEBUG", ""):
                cats_str = " ".join(
                    f"{k}:{v['bytes']}B/{v['ms']:.3f}ms" for k, v in sorted(cats.items())
                )
                print(
                    f"[spec:io] chunk {i} total={io.get('total_bytes',0)}B "
                    f"{io.get('total_ms',0.0):.3f}ms cats={cats_str}",
                    flush=True,
                )
            outputs_last = outputs

            # (Optional) immediately skip keys again to avoid accidental DMA during decode anchors
            if last and keep_bindings:
                self._session.skip_buffers(keep_bindings)

            # record positions and ids for *every* chunk
            chunks_pos.append(chunk_inputs["position_ids"].copy())
            chunks_ids.append(chunk_inputs["input_ids"].copy())

            if self._write_io_dir is not None:
                write_io_files(
                    inputs,
                    outputs,
                    self._write_io_dir,
                    "prefill",
                    "aic_batch_io",
                    True,
                    False,
                )

        # Save and optionally print per-prefill IO summary
        self._prefill_io_totals = io_totals
        self._prefill_io_total_ms = io_total_ms
        if (os.getenv("QEFF_SPEC_IO_TIMING", "") or os.getenv("QEFF_SPEC_DEBUG", "")) and io_total_ms:
            cats = " ".join(
                f"{k}:{io_totals[k]['bytes']}B/{io_totals[k]['ms']:.3f}ms"
                for k in sorted(io_totals.keys())
            )
            print(
                f"[spec:io] summary total={sum(io_totals[k]['bytes'] for k in io_totals)}B "
                f"{io_total_ms:.3f}ms cats={cats}",
                flush=True,
            )

        if outputs_last is None:
            raise RuntimeError("No outputs from prefill; empty prompt?")

        if want_logits and "logits" in outputs_last:
            last_logits = outputs_last["logits"][0, -1]
            token_id = int(last_logits.argmax())
            token_text = self.tokenizer.decode([token_id], skip_special_tokens=True)
            if os.getenv("QEFF_SPEC_DEBUG", ""):
                print(
                    f"[spec:prefill] next predicted token: {token_text!r}",
                    flush=True,
                )

        # Cache last outputs for possible look-ahead
        self._last_outputs = outputs_last

        # Cache Q_final (prefill_queries from last chunk)
        if "prefill_queries" not in outputs_last:
            raise KeyError("prefill_queries not found in last prefill outputs")
        Q_final = outputs_last["prefill_queries"]
        if Q_final.ndim == 4 and Q_final.shape[1] == 1:
            # squeeze batch if present
            Q_final = Q_final[:, 0, :, :]

        if not self._kept_layers:
            raise RuntimeError("[spec] kept_layers is empty; cannot harvest retained-state keys.")

        # Harvest *retained-state* keys *once* from the final chunk (full ctx_len per layer), for kept layers
        per_layer_keys_last = []
        for li in self._kept_layers:
            key_name = f"past_key.{li}_RetainedState"
            if key_name not in outputs_last:
                raise RuntimeError(f"{key_name} missing in last-chunk outputs")
            per_layer_keys_last.append(outputs_last[key_name])  # [1,H_kv,ctx_len,D]
        # Store as a single "chunk" (we'll slice to S_total later)
        chunks_keys: List[List[np.ndarray]] = [per_layer_keys_last]

        # store cache for host scoring
        self._prefill_cache = {
            "chunks_keys": chunks_keys,
            "chunks_pos": chunks_pos,
            "chunks_ids": chunks_ids,
            "Q_final": Q_final,
            "outputs_last": outputs_last,
        }

        t_prefill_end = time.perf_counter()
        total_t = t_prefill_end - t_prefill_start
        avg_t = total_t / max(1, num_chunks)
        if os.getenv("QEFF_SPEC_DEBUG", ""):
            try:
                worst = sorted(per_chunk_times, key=lambda x: x[1], reverse=True)[:3]
                worst_fmt = ", ".join([f"({ci}, {dt:.3f}s)" for ci, dt in worst])
                print(
                    f"[spec:prefill] chunks={num_chunks}  total={total_t:.3f}s  avg={avg_t:.3f}s  worst=[{worst_fmt}]",
                    flush=True,
                )
            except Exception:
                pass

        return outputs_last, position_ids, generation_len

    # --- helper to collect tensors for scoring from a prefill 'outputs' dict ---
    def collect_for_scoring(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract speculator tensors needed for scoring:
          - prefill_queries: np.ndarray [L,H,D]
          - past_keys: List[np.ndarray] len L, each [1,H_kv,S,D]
        Returns: {"prefill_queries": ..., "past_keys": ...}
        """
        if "prefill_queries" not in outputs:
            raise KeyError(
                "prefill_queries not found in outputs (ensure spec QPC was exported with this output)"
            )
        prefill_queries = outputs["prefill_queries"]
        past_keys: List[np.ndarray] = []
        idx = 0
        while True:
            key_name = f"past_key.{idx}_RetainedState"
            if key_name not in outputs:
                break
            past_keys.append(outputs[key_name])
            idx += 1
        if not past_keys:
            raise KeyError(
                "No past_key.{i}_RetainedState outputs found in speculator outputs"
            )
        return {"prefill_queries": prefill_queries, "past_keys": past_keys}

    def _softmax_axis(self, x: np.ndarray, axis: int) -> np.ndarray:
        """Numerically-stable softmax along `axis`."""
        x = x - np.max(x, axis=axis, keepdims=True)
        ex = np.exp(x, dtype=np.float32)
        return ex / np.sum(ex, axis=axis, keepdims=True)

    # ---------- NEW: stable softmax along 1-D vector ----------
    def _softmax_1d_safe(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        x: [S] float32 logits
        mask: optional bool [S]; if provided, masked positions get 0 prob
        returns: [S] float32
        """
        z = x.copy()
        if mask is not None:
            # set masked to very large negative
            z[~mask] = -1e30
        m = np.max(z, axis=0, keepdims=False)
        ex = np.exp(z - m, dtype=np.float32)
        if mask is not None:
            ex[~mask] = 0.0
        s = np.sum(ex, axis=0, dtype=np.float32)
        # avoid div-zero
        s = np.maximum(s, 1e-30)
        return ex / s

    # ---------- NEW: build global K per layer & global position ids ----------
    def _assemble_pos_ids_only(
        self,
        chunks_pos: List[np.ndarray],
        chunks_ids: List[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Concatenate position and id arrays across chunks without touching retained-state keys.
        Returns (pos_global, ids_global, S_total).
        """
        pos_global_list: List[np.ndarray] = []
        ids_global_list: List[np.ndarray] = []
        S_total = 0

        for ci, pos_ids in enumerate(chunks_pos):
            p = pos_ids.reshape(-1).astype(np.int64)
            mask_valid = p >= 0
            vl = int(mask_valid.sum())
            if vl == 0:
                continue
            pos_global_list.append(p[mask_valid])
            ids_chunk = chunks_ids[ci].reshape(-1).astype(np.int64)
            ids_global_list.append(ids_chunk[:vl])
            S_total += vl

        if S_total == 0:
            raise RuntimeError("No valid tokens assembled from chunks_pos")

        pos_global = np.concatenate(pos_global_list, axis=0).astype(np.int64, copy=False)
        ids_global = np.concatenate(ids_global_list, axis=0).astype(np.int64, copy=False)
        return pos_global, ids_global, S_total

    def _assemble_global_keys(
        self,
        chunks_keys: List[List[np.ndarray]],  # here length may be 1: final-chunk retained state [1,H_kv,ctx_len,D]
        chunks_pos: List[np.ndarray],         # per-chunk position_ids [1,S_chunk] (pads=-1 in last chunk tail)
        chunks_ids: List[np.ndarray],         # per-chunk input_ids [1,S_chunk] (exactly what was fed)
    ) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray, int, int, int]:
        """
        Concatenate per-chunk keys per layer along token axis (exclude pads) and build global pos_ids.

        Returns:
          K_global: list length L, each [H_kv, S_total, D] float16 or float32
          pos_global: [S_total] int64 (>=0)
          S_total, H_kv, D
        """
        # infer layer count from the (single) kept-keys list
        L = len(chunks_keys[0])
        # collect per-layer lists of [H_kv, valid_len_i, D]
        per_layer_buffers: List[List[np.ndarray]] = [[] for _ in range(L)]
        pos_global_list: List[np.ndarray] = []
        ids_global_list: List[np.ndarray] = []

        H_kv = None
        D = None
        S_total = 0

        # Build global pos & ids from *all* chunks
        for ci, pos_ids in enumerate(chunks_pos):
            p = pos_ids.reshape(-1).astype(np.int64)
            mask_valid = (p >= 0)
            vl = int(mask_valid.sum())
            if vl == 0:
                continue
            pos_global_list.append(p[mask_valid])
            ids_chunk = chunks_ids[ci].reshape(-1).astype(np.int64)
            ids_global_list.append(ids_chunk[:vl])
            S_total += vl

        if S_total == 0:
            raise RuntimeError("No valid tokens assembled from chunks_pos")

        # Keys were fetched only once from the last chunk: retained-state [1,H_kv,ctx_len,D]
        keys_per_layer = chunks_keys[0]
        for li in range(L):
            k = keys_per_layer[li]
            if k is None:
                raise RuntimeError(f"Missing retained-state keys for layer {li}")
            if H_kv is None or D is None:
                H_kv = int(k.shape[1])
                D = int(k.shape[3])
            # slice retained-state to global valid length, drop batch dim -> [H_kv, S_total, D]
            k_slice = k[0, :, :S_total, :].astype(np.float16, copy=False)
            per_layer_buffers[li].append(k_slice)

        # concatenate per-layer lists along token axis (here each has single slice already)
        K_global: List[np.ndarray] = []
        for li in range(L):
            if len(per_layer_buffers[li]) == 0:
                raise RuntimeError(f"No key buffers for layer {li}")
            K_concat = np.concatenate(per_layer_buffers[li], axis=1).astype(np.float16, copy=False)
            K_global.append(K_concat)

        pos_global = np.concatenate(pos_global_list, axis=0).astype(np.int64, copy=False)  # [S_total]
        ids_global = np.concatenate(ids_global_list, axis=0).astype(np.int64, copy=False)  # [S_total]
        return K_global, pos_global, ids_global, S_total, H_kv, D

    # ---------- NEW: global scoring respecting GQA and padding ----------
    def _score_global_importance(
        self,
        Q_final: np.ndarray,           # [L, H, D] (float16/32)
        K_global: List[np.ndarray],    # len L; each [H_kv, S, D] (float16)
        pos_global: np.ndarray,        # [S] int64; pad positions excluded already
        *,
        layers_sel: Optional[str] = "all",   # "all", "last4", "last1"
        agg_heads: str = "max",              # "max" or "sum" (accepts "lse" as alias)
        smooth_window: Optional[int] = None  # e.g., 3 or 5; None to skip
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Compute importance[i] over full sequence S:
          1) per layer/head: softmax( q . K^T / sqrt(D) ) over S (pad already removed)
          2) aggregate heads (max or sum across heads)
          3) max across selected layers (paper §3.2.2)
          4) optional smoothing

        Returns:
          importance [S] float32
          diag: some diagnostics (optional)
        """
        L, H, D = int(Q_final.shape[0]), int(Q_final.shape[1]), int(Q_final.shape[2])
        diag: Dict[str, float] = {}
        S = int(pos_global.shape[0])
        if S == 0:
            raise ValueError("Empty global sequence length S")

        # choose layers
        if layers_sel == "last1":
            layer_ids = [L - 1]
        elif layers_sel == "last4":
            layer_ids = list(range(max(L - 4, 0), L))
        else:
            layer_ids = list(range(L))

        # ---- GQA invariants (assert hard and print once when debug is on) ----
        H_kv = int(K_global[layer_ids[0]].shape[0])
        D_k = int(K_global[layer_ids[0]].shape[2])
        assert H_kv > 0, f"GQA invalid: H_kv={H_kv}"
        assert D_k == D, f"D mismatch Q vs K: Dq={D}, Dk={D_k}"
        assert H % H_kv == 0, f"GQA mismatch: H={H}, H_kv={H_kv} (H must be a multiple of H_kv)"
        group_size = H // H_kv
        # Example: [spec:invariants] S_total=64539 Q_final=(32,32,128) K0=(8,64539,128) H=32 H_kv=8 group=4
        print(
            f"[spec:invariants] S_total={S} Q_final=({L},{H},{D}) "
            f"K0=({H_kv},{S},{D_k}) H={H} H_kv={H_kv} group={group_size}",
            flush=True,
        )

        # precompute sqrt(D)
        scale = 1.0 / math.sqrt(float(D))

        # Vectorized head scoring with layer streaming
        layer_max = None
        head_to_kv = (np.arange(H) // group_size).astype(np.int64)
        for li in layer_ids:
            K_l = K_global[li].astype(np.float32, copy=False)  # [H_kv,S,D]
            Q_l = Q_final[li].astype(np.float32, copy=False)   # [H,D]
            K_sel = K_l[head_to_kv, :, :]                      # [H,S,D]
            logits = np.einsum("hsd,hd->hs", K_sel, Q_l) * scale  # [H,S]
            logits -= logits.max(axis=1, keepdims=True)
            probs = np.exp(logits, dtype=np.float32)
            probs /= np.maximum(probs.sum(axis=1, keepdims=True), 1e-30)
            # Head aggregation: "max" (paper default) or "sum" (previously labeled "lse")
            if agg_heads in ("sum", "lse"):
                head_ag = probs.sum(axis=0)
            else:
                head_ag = probs.max(axis=0)
            layer_max = head_ag if layer_max is None else np.maximum(layer_max, head_ag)

        importance = layer_max  # [S]

        # optional smoothing (simple moving average; length-preserving)
        if smooth_window is not None and smooth_window > 1:
            w = int(smooth_window)
            S = importance.shape[0]
            # Clamp window to sequence length to avoid shrinking output
            w = max(1, min(w, S))
            if w > 1:
                # Uniform kernel and same-length convolution
                kernel = np.ones((w,), dtype=np.float32) / float(w)
                importance = np.convolve(importance, kernel, mode="same").astype(
                    np.float32, copy=False
                )
            # hard invariant: preserve length
            assert importance.shape[0] == S, f"smoothing altered length: {importance.shape[0]} != {S}"

        # diagnostics
        if os.getenv("QEFF_SPEC_ASSERT", ""):
            # quick softmax sanity on one head (first layer/head)
            g0 = 0
            q0 = Q_final[layer_ids[0], 0, :].astype(np.float32, copy=False)
            z0 = (K_global[layer_ids[0]][g0, :, :].astype(np.float32, copy=False) @ q0) * scale
            a0 = self._softmax_1d_safe(z0)
            diag["softmax_sum_l0h0"] = float(np.sum(a0))
        return importance, diag

    # ---------- NEW: top-% selection ----------
    def _select_global_topk(
        self,
        importance: np.ndarray,      # [S] float32
        keep_fraction: float,
        *,
        force_last: bool = True
    ) -> np.ndarray:
        S = int(importance.shape[0])
        k = max(1, int(math.ceil(keep_fraction * S)))
        # top-k by partial argpartition
        idx = np.argpartition(-importance, k-1)[:k]
        idx.sort()
        if force_last and (S-1) not in idx:
            # insert last; maintain sort
            idx = np.unique(np.concatenate([idx, np.array([S-1], dtype=idx.dtype)]))
        return idx

    def _select_blocks_topk(
        self,
        importance: np.ndarray,    # [S] float32
        keep_fraction: float,
        chunk_size: int,
        *,
        force_last: bool = True,
    ) -> np.ndarray:
        import math
        S = int(importance.shape[0])
        if chunk_size <= 0:
            return self._select_global_topk(importance, keep_fraction, force_last=force_last)

        num_blocks = (S + chunk_size - 1) // chunk_size
        block_scores = np.empty((num_blocks,), dtype=np.float32)
        for b in range(num_blocks):
            s = b * chunk_size
            e = min(S, s + chunk_size)
            block_scores[b] = float(np.mean(importance[s:e]))

        keep_blocks = max(1, int(math.ceil(keep_fraction * num_blocks)))
        topb = np.argpartition(-block_scores, keep_blocks - 1)[:keep_blocks]
        topb.sort()

        kept = []
        for b in topb.tolist():
            s = b * chunk_size
            e = min(S, s + chunk_size)
            kept.append(np.arange(s, e, dtype=np.int64))
        idx = np.unique(np.concatenate(kept))
        if force_last and (S - 1) not in idx:
            idx = np.unique(np.concatenate([idx, np.array([S - 1], dtype=np.int64)]))
        return idx

    def _collect_anchor_queries(self, N: int, S: int) -> List[np.ndarray]:
        anchors: List[np.ndarray] = [self._prefill_cache["Q_final"]]
        if N <= 0:
            return anchors

        if (
            ("prefill_queries" not in self._session.binding_index_map)
            or (not self._session.allowed_shapes)
            or (len(self._session.allowed_shapes) < 2)
        ):
            return anchors

        idx = self._session.binding_index_map["prefill_queries"]
        _, dec_dims = self._session.allowed_shapes[1][idx]
        pq_decode = np.empty(tuple(dec_dims), dtype=np.float32)

        next_pos = S
        for _ in range(N):
            last = getattr(self, "_last_outputs", None)
            if last is None or "logits" not in last:
                break
            next_token = int(np.argmax(last["logits"][0, -1, :]))
            # Optional early stop on EOS if tokenizer is available
            eos_id = getattr(self.tokenizer, "eos_token_id", None)
            try:
                if eos_id is not None and next_token == int(eos_id):
                    break
            except Exception:
                pass
            dec_inputs = {
                "input_ids": np.array([[next_token]], dtype=np.int64),
                "position_ids": np.array([[next_pos]], dtype=np.int64),
            }
            self._session.set_buffers({"prefill_queries": pq_decode})
            out = self._session.run(dec_inputs)
            self._last_outputs = out
            if "prefill_queries" in out:
                anchors.append(out["prefill_queries"].copy())
            next_pos += 1

        if os.getenv("QEFF_SPEC_DEBUG", ""):
            # Keep this lightweight; token strings are optional and may not be available here
            print(f"[spec:anchors] used={len(anchors)}", flush=True)
        return anchors

    def _score_importance_multi_anchor(
        self,
        K_global: List[np.ndarray],
        Q_anchors: List[np.ndarray],
        pos_global: np.ndarray,
        *,
        layers_sel: str,
        agg_heads: str,
        smooth_window: Optional[int],
    ) -> np.ndarray:
        acc = []
        for Q_final in Q_anchors:
            imp, _ = self._score_global_importance(
                Q_final,
                K_global,
                pos_global,
                layers_sel=layers_sel,
                agg_heads=agg_heads,
                smooth_window=None,
            )
            acc.append(imp.astype(np.float32, copy=False))

        importance = np.mean(np.stack(acc, axis=0), axis=0, dtype=np.float32)
        if smooth_window and smooth_window > 1:
            w = max(1, min(int(smooth_window), importance.shape[0]))
            if w > 1:
                kernel = np.ones((w,), dtype=np.float32) / float(w)
                importance = np.convolve(importance, kernel, mode="same").astype(np.float32)
        return importance

    def _avg_pool1d_same(self, x: np.ndarray, kernel: int) -> np.ndarray:
        """
        Moving-average along last axis with 'same' output length.
        Works for odd or even kernels by asymmetric edge padding.
        x: [..., S] -> returns [..., S]
        """
        if kernel is None or kernel <= 1:
            return x

        # Asymmetric padding ensures output length matches input length
        pad_left = kernel // 2
        pad_right = kernel - 1 - pad_left  # total padding = kernel - 1

        xpad = np.pad(
            x,
            [(0, 0)] * (x.ndim - 1) + [(pad_left, pad_right)],
            mode="edge",
        )
        # Prepend zero for sliding window via cumsum
        cs = np.cumsum(xpad, axis=-1, dtype=np.float32)
        zero = np.zeros_like(xpad[..., :1], dtype=np.float32)
        cs = np.concatenate([zero, cs], axis=-1)

        out = (cs[..., kernel:] - cs[..., :-kernel]) / float(kernel)
        return out

    def compute_importance(
        self,
        prefill_queries: np.ndarray,  # [L,H,D]
        past_keys: List[np.ndarray],  # len L, each [1,H_kv,S,D]
        pool_kernel_size: Optional[int] = 13,
    ) -> np.ndarray:
        """Compute token-importance [S] for k=0."""
        # ---- sanitize inputs (defensive) ----
        # Convert to float32 and replace NaN/+Inf/-Inf with 0.0 so softmax is well-posed
        q = np.nan_to_num(
            prefill_queries.astype(np.float32, copy=False),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )  # [L,H,D]

        k_list = []
        for k in past_keys:
            if k.ndim != 4:
                raise ValueError(f"past_key must be [1,H_kv,S,D], got {k.shape}")
            k_list.append(k.squeeze(0))  # [H_kv,S,D]
        K = np.stack(k_list, axis=0).astype(np.float32, copy=False)  # [L,H_kv,S,D]
        K = np.nan_to_num(K, nan=0.0, posinf=0.0, neginf=0.0)

        L, H, D = q.shape
        _, H_kv, S, Dk = K.shape
        if Dk != D:
            raise ValueError(f"D mismatch: queries D={D} vs keys D={Dk}")

        if H != H_kv:
            if H % H_kv != 0:
                raise ValueError(f"H ({H}) must be multiple of H_kv ({H_kv})")
            rep = H // H_kv
            K = np.repeat(K, repeats=rep, axis=1)  # [L,H,S,D]

        scale = 1.0 / math.sqrt(float(D))
        scores = np.einsum("lhd,lhsd->lhs", q, K, optimize=True) * scale
        attn = self._softmax_axis(scores, axis=-1)  # [L,H,S]
        attn = self._avg_pool1d_same(attn, kernel=pool_kernel_size)

        assert (
            attn.shape[-1] == S
        ), f"avg_pool1d_same produced length {attn.shape[-1]} != expected {S}"

        # Optional numerics check: softmax sums ~1 along last axis
        if os.getenv("QEFF_SPEC_ASSERT", ""):
            if not pool_kernel_size or pool_kernel_size <= 1:
                sums = np.sum(attn, axis=-1)
                assert np.allclose(
                    sums, 1.0, atol=1e-3
                ), f"softmax sums not ~1; max dev: {np.max(np.abs(sums - 1.0))}"

        attn_lh = attn.reshape(L * H, S)
        importance = np.max(attn_lh, axis=0).astype(np.float32, copy=False)
        return importance

    def select_tokens(self, importance: np.ndarray, keep_cfg: KeepConfig) -> np.ndarray:
        """Keep indices per policy."""
        if keep_cfg.strategy != "percentage":
            raise ValueError(f"Unsupported keep strategy: {keep_cfg.strategy}")
        S = int(importance.shape[0])
        must_keep = np.array([S - 1], dtype=np.int64)

        if keep_cfg.chunk:
            cs = int(keep_cfg.chunk_size)
            if cs <= 0:
                raise ValueError("chunk_size must be > 0")
            n_chunks = (S + cs - 1) // cs
            chunk_scores = np.empty((n_chunks,), dtype=np.float32)
            for i in range(n_chunks):
                start = i * cs
                end = min((i + 1) * cs, S)
                chunk_scores[i] = float(
                    np.mean(importance[start:end], dtype=np.float32)
                )
            k_chunks = max(1, int(math.ceil(n_chunks * keep_cfg.percentage)))
            keep_chunk_idx = np.argpartition(-chunk_scores, k_chunks - 1)[:k_chunks]
            keep_chunk_idx.sort()
            kept_slices = [
                np.arange(i * cs, min((i + 1) * cs, S), dtype=np.int64)
                for i in keep_chunk_idx.tolist()
            ]
            kept = (
                np.concatenate(kept_slices, axis=0)
                if kept_slices
                else np.empty((0,), dtype=np.int64)
            )
        else:
            k = max(1, int(math.ceil(S * keep_cfg.percentage)))
            kept = np.argpartition(-importance, k - 1)[:k].astype(np.int64)
            kept.sort()

        kept = np.unique(np.concatenate([kept, must_keep], axis=0))
        return kept


    

    def prefill_and_score(
        self,
        prompt: str,
        *,
        pool_kernel_size: int = 13,
        keep_cfg: Optional[KeepConfig] = None,
        layers_sel: str = "all",
    ) -> Dict[str, Any]:
        """
        Run speculator prefill (once if needed), assemble global tensors from cached chunks,
        host-score, select keep_idx.
        """
        if keep_cfg is None:
            keep_cfg = KeepConfig()
        if keep_cfg.strategy != "percentage":
            raise NotImplementedError(f"keep strategy {keep_cfg.strategy!r}")

        t0 = time.perf_counter()
        # Ensure we have a first-pass cache. If absent, run prefill once to populate.
        if self._prefill_cache is None:
            _ = self.run_prefill(prompt, generation_len=None, prefill_logit_bs=1)
        if self._prefill_cache is None:
            raise RuntimeError("Prefill cache missing after run_prefill")
        t1 = time.perf_counter()

        chunks_keys = self._prefill_cache.get("chunks_keys")
        chunks_pos = self._prefill_cache["chunks_pos"]
        chunks_ids = self._prefill_cache["chunks_ids"]
        Q_final = self._prefill_cache["Q_final"]
        outputs_last = self._prefill_cache.get("outputs_last")
        if self._kept_layers:
            Q_final = Q_final[self._kept_layers, :, :]
        diag = {}

        if chunks_keys is None:
            raise RuntimeError(
                "Retained-state keys missing; ensure QPC exports past_key.*_RetainedState"
            )

        K_global, pos_global, ids_global, S_total, H_kv, D = self._assemble_global_keys(
            chunks_keys, chunks_pos, chunks_ids
        )
        t2 = time.perf_counter()

        Q_anchors = self._collect_anchor_queries(getattr(self, "_look_ahead", 0), S_total)
        importance = self._score_importance_multi_anchor(
            K_global,
            Q_anchors,
            pos_global,
            layers_sel=layers_sel,
            agg_heads="max",
            smooth_window=pool_kernel_size if pool_kernel_size and pool_kernel_size > 1 else None,
        )
        t3 = time.perf_counter()

        if os.getenv("QEFF_SPEC_ASSERT", ""):
            print(f"[spec:diag] softmax_sum_l0h0={diag.get('softmax_sum_l0h0', None)}")

        if importance is None or pos_global is None or ids_global is None or S_total is None:
            raise RuntimeError("Importance/positions assembly failed")

        # Hard sanity checks (not gated)
        if importance.shape[0] != S_total:
            raise AssertionError(f"importance len {importance.shape[0]} != S {S_total}")
        if ids_global.shape[0] != S_total:
            raise AssertionError(f"ids_global len {ids_global.shape[0]} != S {S_total}")

        blocks = bool(keep_cfg and getattr(keep_cfg, "chunk", True))
        if blocks:
            keep_idx = self._select_blocks_topk(
                importance,
                keep_cfg.percentage,
                keep_cfg.chunk_size,
                force_last=True,
            )
        else:
            keep_idx = self._select_global_topk(
                importance, keep_cfg.percentage, force_last=True
            )
        if os.getenv("QEFF_SPEC_DEBUG", ""):
            try:
                print(
                    f"[spec:select] blocks={blocks} chunk_size={getattr(keep_cfg, 'chunk_size', None)} "
                    f"kept={len(keep_idx)}/{S_total}",
                    flush=True,
                )
            except Exception:
                pass
        keep_idx = np.union1d(
            keep_idx.astype(np.int64, copy=False), np.array([S_total - 1], dtype=np.int64)
        )
        try:
            last_id = int(ids_global[S_total - 1])
            last_tok = self.tokenizer.convert_ids_to_tokens([last_id])[0]
            print(f"[spec:prefill] last prompt token: {last_tok!r}")
        except Exception:
            pass
        t4 = time.perf_counter()

        # collect timing diagnostics for caller
        t_run_prefill_s = t1 - t0
        t_assemble_s = t2 - t1
        t_score_s = t3 - t2
        t_select_s = t4 - t3

        if os.getenv("QEFF_SPEC_DEBUG", ""):
            try:
                print(
                    "[spec:host] run_prefill=%.3fs assemble=%.3fs score=%.3fs select=%.3fs S=%d layers_sel=%s"
                    % (
                        t1 - t0,
                        t2 - t1,
                        t3 - t2,
                        t4 - t3,
                        S_total,
                        layers_sel,
                    ),
                    flush=True,
                )
            except Exception:
                pass

        # Post-invariants
        if keep_idx.size == 0 or keep_idx[-1] != S_total - 1:
            raise AssertionError("must keep last real token")
        if keep_idx.min() < 0 or keep_idx.max() >= S_total:
            raise AssertionError("keep_idx out of range")

        return {
            "importance": importance,  # [S_total]
            "keep_idx": keep_idx,  # sorted, includes S_total-1
            "S": S_total,
            "shapes": {
                "prefill_queries": tuple(Q_final.shape),
                "first_key": tuple(K_global[0].shape),
            },
            "ids_global": ids_global,  # [S_total] int64 (exact device-fed tokens)
            # timing breakdown
            "t_run_prefill_s": t_run_prefill_s,
            "t_assemble_s": t_assemble_s,
            "t_score_s": t_score_s,
            "t_select_s": t_select_s,
        }

    def prune_and_base_prefill(
        self,
        base_engine,
        prompt: str,
        *,
        pool_kernel_size: int = 13,
        keep_cfg: Optional[KeepConfig] = None,
        prefill_logit_bs: int = 1,
        layers_sel: str = "all",
        gen_len: Optional[int] = None,
        look_ahead: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Step 4.3: spec prefill+score -> build pruned ids/pos -> run base prefill
        (baseline and pruned) with simple timings.
        """
        # Persist look-ahead for downstream helpers (defaults to 0 if absent)
        self._look_ahead = int(look_ahead) if look_ahead is not None else 0
        # ---- TTFT(spec_only): spec prefill + score + select ----
        res = self.prefill_and_score(
            prompt,
            pool_kernel_size=pool_kernel_size,
            keep_cfg=keep_cfg,
            layers_sel=layers_sel,
        )
        t_run_prefill_s = res.get("t_run_prefill_s", 0.0)
        t_assemble_s = res.get("t_assemble_s", 0.0)
        t_score_s = res.get("t_score_s", 0.0)
        t_select_s = res.get("t_select_s", 0.0)
        ttft_spec_device_s = t_run_prefill_s
        ttft_host_scoring_s = t_assemble_s + t_score_s + t_select_s
        ttft_spec_only_s = ttft_spec_device_s + ttft_host_scoring_s
        # aggregated device->host IO across prefill
        io_totals = getattr(
            self,
            "_prefill_io_totals",
            {k: {"bytes": 0, "ms": 0.0} for k in (
                "prefill_queries",
                "past_key",
                "past_value",
                "logits",
                "other",
            )},
        )
        io_total_ms = float(getattr(self, "_prefill_io_total_ms", 0.0))
        if os.getenv("QEFF_SPEC_IO_TIMING", "") or os.getenv("QEFF_SPEC_DEBUG", ""):
            cats_str = " ".join(
                f"{k}:{io_totals[k]['bytes']}B/{io_totals[k]['ms']:.1f}ms"
                for k in sorted(io_totals.keys())
            )
            spec_compute_ms = max(ttft_spec_device_s * 1000.0 - io_total_ms, 0.0)
            print(
                f"[spec:io-total] {cats_str} | io_total={io_total_ms:.1f}ms | "
                f"spec_compute_only={spec_compute_ms:.1f}ms",
                flush=True,
            )
        keep_idx = res["keep_idx"]
        S = res["S"]
        ids_global = res.get("ids_global", None)
        # Unconditional invariants (catch regressions early)
        imp = res["importance"]
        assert imp.shape[0] == S, f"importance len {imp.shape[0]} != S {S}"
        assert keep_idx.size > 0 and keep_idx[-1] == S - 1, "must keep last real token"
        assert keep_idx.min() >= 0 and keep_idx.max() < S, "keep_idx out of range"
        assert ids_global is not None and ids_global.shape[0] == S, "ids_global missing/mismatch"
        # Build pruned inputs from the *exact* device-fed ids (no re-tokenization)
        if ids_global is None:
            raise RuntimeError("ids_global not present in scoring result")
        final_ids = ids_global[keep_idx].reshape(1, -1).astype(np.int64, copy=False)
        final_pos = keep_idx.reshape(1, -1).astype(np.int64, copy=False)

        # ---- debug: print pruned tokens using device-fed ids ----
        try:
            if os.getenv("QEFF_SPEC_DEBUG", ""):
                ids_list = final_ids[0].tolist()
                tok_list = self.tokenizer.convert_ids_to_tokens(ids_list)
                k_preview = keep_idx[:32].tolist()
                t_preview = tok_list[:32]
                print(f"[spec:pruned] kept={len(ids_list)}/{S} keep_idx[:32]={k_preview}")
                print(f"[spec:pruned] tokens[:32]={t_preview}")
                txt_preview = self.tokenizer.decode(ids_list, skip_special_tokens=True)
                print(f"[spec:pruned] text_preview={txt_preview[:120]!r}")
        except Exception:
            pass

        # ---- TTFT(base_full): baseline base prefill on full prompt ----
        t0 = time.perf_counter()
        out_base_full, pos_full, _ = base_engine.run_prefill(
            prompt, generation_len=None, prefill_logit_bs=prefill_logit_bs
        )
        ttft_baseline_s = time.perf_counter() - t0

        # ---- TTFT(base_pruned_only): base prefill on pruned ids/pos ----
        t1 = time.perf_counter()
        out_base_pruned, _, padded_len, num_chunks = base_engine.prefill_from_ids(
            final_ids, final_pos, prefill_logit_bs=prefill_logit_bs
        )
        t2 = time.perf_counter()
        ttft_base_pruned_only_s = t2 - t1
        ttft_speculative_s = ttft_spec_only_s + ttft_base_pruned_only_s

        # ---- OPTIONAL: run decode on the pruned path to capture the speculative base output ----
        generated_text_pruned = None
        try:
            if gen_len is None:
                # Single-call decode until EOS or remaining context budget.
                remaining = max(1, int(self._ctx_len) - int(S))
                # Optional: skip gracefully if decode specialization is missing
                try:
                    session = getattr(base_engine, "_session", None)
                    allowed = getattr(session, "allowed_shapes", None)
                    has_decode = bool(
                        session and allowed is not None and len(allowed) >= 2
                    )
                except Exception:
                    has_decode = True  # attempt anyway; runtime will enforce
                if has_decode:
                    base_engine.initialize_decode_inputs(1, 1, remaining)
                    next_pos = np.array([[S]], dtype=np.int64)
                    base_engine.update_decode_input(out_base_pruned, next_pos, generation_len=remaining)
                    decode_inputs = base_engine.prepare_decode_inputs()
                    _ = base_engine.run_decode(decode_inputs, generation_len=remaining)
                    generated = self.tokenizer.batch_decode(
                        base_engine.generated_ids, skip_special_tokens=True
                    )
                    if isinstance(generated, list) and len(generated) > 0:
                        generated_text_pruned = generated[0]
            elif int(gen_len) > 0:
                # Existing fixed-length behavior
                base_engine.initialize_decode_inputs(1, 1, int(gen_len))
                next_pos = np.array([[S]], dtype=np.int64)
                base_engine.update_decode_input(
                    out_base_pruned, next_pos, generation_len=int(gen_len)
                )
                decode_inputs = base_engine.prepare_decode_inputs()
                _ = base_engine.run_decode(decode_inputs, generation_len=int(gen_len))
                generated = self.tokenizer.batch_decode(
                    base_engine.generated_ids, skip_special_tokens=True
                )
                if isinstance(generated, list) and len(generated) > 0:
                    generated_text_pruned = generated[0]
        except Exception:
            # Decode is optional; do not affect TTFT accounting
            pass

        # Always print a compact TTFT summary (ms)
        print(
            f"[4.3] S={S} kept={keep_idx.size} "
            f"TTFT(base_full)={ttft_baseline_s*1000:.1f}ms "
            f"TTFT(spec_device)={ttft_spec_device_s*1000:.1f}ms "
            f"TTFT(host_scoring)={ttft_host_scoring_s*1000:.1f}ms "
            f"TTFT(base_pruned_only)={ttft_base_pruned_only_s*1000:.1f}ms "
            f"TTFT(speculative)={ttft_speculative_s*1000:.1f}ms"
        )
        return {
            "S": S,
            "kept": int(keep_idx.size),
            "keep_idx": keep_idx,
            "ttft_baseline_s": ttft_baseline_s,
            "ttft_spec_only_s": ttft_spec_only_s,
            "ttft_spec_device_s": ttft_spec_device_s,
            "ttft_host_scoring_s": ttft_host_scoring_s,
            "ttft_base_pruned_only_s": ttft_base_pruned_only_s,
            "ttft_speculative_s": ttft_speculative_s,
            "padded_len_pruned": padded_len,
            "num_chunks_pruned": num_chunks,
            "generated_text_pruned": generated_text_pruned,
        }
# --------------------- simple __main__ validator ---------------------
def main() -> None:
    import argparse

    from QEfficient.utils import load_hf_tokenizer

    parser = argparse.ArgumentParser(
        description="Step 4.1: speculator prefill parity check (mirrors runtime prefill)."
    )
    parser.add_argument(
        "--spec-qpc", required=True, help="Path to speculator QPC directory (…/qpc)"
    )
    parser.add_argument(
        "--model-name",
        default="meta-llama/Meta-Llama-3-8B",
        help="HF tokenizer card for speculator",
    )
    parser.add_argument(
        "--prompt", default="Hello from spec prefill on AI100.", help="Prompt string"
    )
    parser.add_argument(
        "--prompt-len", type=int, default=16, help="prefill chunk length"
    )
    parser.add_argument("--ctx-len", type=int, default=128, help="context length")
    parser.add_argument(
        "--device-ids", default="[0]", help="Device IDs like [0] or [0,1]"
    )
    parser.add_argument(
        "--prefill-logit-bs", type=int, default=1, help="logits placeholder batch size"
    )
    args = parser.parse_args()

    try:
        dev_ids = [
            int(x) for x in args.device_ids.strip("[] ").split(",") if x.strip() != ""
        ]
    except Exception:
        print(
            f"[warn] Could not parse --device-ids={args.device_ids!r}, defaulting to [0]"
        )
        dev_ids = [0]

    tok = load_hf_tokenizer(args.model_name)
    eng = SpecPrefillEngine(
        spec_qpc_path=args.spec_qpc,
        tokenizer=tok,
        ctx_len=int(args.ctx_len),
        prefill_seq_len=int(args.prompt_len),
        device_ids=dev_ids,
    )

    # Build keep policy (use existing KeepConfig; do not redefine it)
    keep_cfg = KeepConfig(
        strategy="percentage",
        percentage=0.1,  # keep 10%
        chunk=True,
        chunk_size=32,
    )
    pool_kernel_size = 13

    # Step 4.2: prefill + score + select (handles padding/capacity inside)
    res = eng.prefill_and_score(
        args.prompt,
        pool_kernel_size=pool_kernel_size,
        keep_cfg=keep_cfg,
    )

    # Minimal anti-padding invariants
    S = res["S"]
    imp = res["importance"]
    keep_idx = res["keep_idx"]
    assert (
        imp.shape[0] == S
    ), f"importance len {imp.shape[0]} != S {S}"
    if keep_idx.size > 0:
        assert (
            keep_idx.dtype == np.int64
        ), f"keep_idx dtype {keep_idx.dtype} != np.int64"
        assert keep_idx.min() >= 0, f"negative keep index {keep_idx.min()}"
        assert (
            keep_idx.max() < S
        ), f"padded index detected: {keep_idx.max()} >= S {S}"
        assert (
            keep_idx[-1] == S - 1
        ), f"must keep last real token {S-1}; got {keep_idx[-1]}"

    # One-line summary (quiet by default)
    kept_pct = (100.0 * len(keep_idx) / S) if S > 0 else 0.0
    shapes = res.get("shapes", {})
    print(
        f"[score] S={S} kept={len(keep_idx)} ({kept_pct:.1f}%) "
        f"prefill_queries={shapes.get('prefill_queries')} "
        f"first_key={shapes.get('first_key', 'unavailable')} "
        f"last_kept={keep_idx[-1] if keep_idx.size else 'NA'}"
    )


if __name__ == "__main__":
    main()
