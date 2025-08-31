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

        # Speculative prefill uses past_key.* for scoring; skip past_* inputs and past_value outputs.
        past_inputs = [n for n in self._session.input_names if n.startswith("past_")]
        past_outs = [n for n in self._session.output_names if n.startswith("past_value.")]
        self._session.skip_buffers(past_inputs + past_outs)

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
        inputs = self.tokenizer(prompt, return_tensors="np", padding=True)
        position_ids = inputs["attention_mask"].sum(1, keepdims=True)
        padded_len = inputs["input_ids"].shape[1]
        num_chunks = -(padded_len // -self._prefill_seq_len)
        padded_len = num_chunks * self._prefill_seq_len

        max_gen_len = self._ctx_len - position_ids.max()
        generation_len = self._fetch_generation_len(generation_len, max_gen_len)

        logits_out_placeholder = np.zeros(
            (prefill_logit_bs, 1, self._vocab_size), dtype=np.float32
        )
        self._session.set_buffers({"logits": logits_out_placeholder})

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

        for i in range(num_chunks):
            chunk_inputs = inputs.copy()
            chunk_inputs["input_ids"] = inputs["input_ids"][
                :, i * self._prefill_seq_len : (i + 1) * self._prefill_seq_len
            ]
            chunk_inputs["position_ids"] = inputs["position_ids"][
                :, i * self._prefill_seq_len : (i + 1) * self._prefill_seq_len
            ]
            outputs = self._session.run(chunk_inputs)
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

        last_logits = outputs["logits"][0, -1]
        token_id = int(last_logits.argmax())
        token_text = self.tokenizer.decode([token_id], skip_special_tokens=True)
        if os.getenv("QEFF_SPEC_DEBUG", ""):
            print(f"[spec:prefill] final token: {token_text!r}", flush=True)
        return outputs, position_ids, generation_len

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
    def _assemble_global_keys(
        self,
        chunks_keys: List[List[np.ndarray]],   # len(chunks) list of per-layer key arrays [1,H_kv,S_chunk,D] FP16
        chunks_pos: List[np.ndarray],          # len(chunks) list of position_ids [1,S_chunk] int64
        chunks_ids: List[np.ndarray],          # len(chunks) list of input_ids [1,S_chunk] int64 (exactly what was fed)
    ) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray, int, int, int]:
        """
        Concatenate per-chunk keys per layer along token axis (exclude pads) and build global pos_ids.

        Returns:
          K_global: list length L, each [H_kv, S_total, D] float16 or float32
          pos_global: [S_total] int64 (>=0)
          S_total, H_kv, D
        """
        # infer shapes from first non-empty chunk/layer
        L = len(chunks_keys[0])
        # collect per-layer lists of [H_kv, valid_len_i, D]
        per_layer_buffers: List[List[np.ndarray]] = [[] for _ in range(L)]
        pos_global_list: List[np.ndarray] = []
        ids_global_list: List[np.ndarray] = []

        H_kv = None
        D = None
        S_total = 0

        for ci, (keys_per_layer, pos_ids) in enumerate(zip(chunks_keys, chunks_pos)):
            # pos_ids is [1, S_chunk]
            p = pos_ids.reshape(-1).astype(np.int64)
            mask_valid = (p >= 0)
            valid_len = int(mask_valid.sum())
            if valid_len == 0:
                continue
            pos_global_list.append(p[mask_valid])
            # append the exact input_ids used for this chunk's valid region
            ids_chunk = chunks_ids[ci].reshape(-1).astype(np.int64)
            ids_global_list.append(ids_chunk[:valid_len])
            S_total += valid_len

            for li in range(L):
                # chunk key tensor shape [1, H_kv, S_chunk, D]
                k = keys_per_layer[li]
                if k is None:
                    raise RuntimeError(f"Missing keys for layer {li} in chunk {ci}")
                if H_kv is None or D is None:
                    H_kv = int(k.shape[1])
                    D = int(k.shape[3])
                # slice to valid_len, drop batch dim
                k_slice = k[0, :, :valid_len, :].astype(np.float16, copy=False)
                per_layer_buffers[li].append(k_slice)  # [H_kv, valid_len, D]

        if H_kv is None or D is None:
            raise RuntimeError("Unable to infer H_kv/D from keys; no valid chunks?")

        # concatenate per-layer lists along token axis
        K_global: List[np.ndarray] = []
        for li in range(L):
            if len(per_layer_buffers[li]) == 0:
                raise RuntimeError(f"No key buffers for layer {li}")
            # [H_kv, S_total, D]
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
        agg_heads: str = "max",              # "max" or "lse"
        smooth_window: Optional[int] = None  # e.g., 3 or 5; None to skip
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Compute importance[i] over full sequence S:
          1) per layer/head: softmax( q . K^T / sqrt(D) ) over S (pad already removed)
          2) aggregate heads (max or log-sum-exp)
          3) mean across selected layers
          4) optional smoothing

        Returns:
          importance [S] float32
          diag: some diagnostics (optional)
        """
        L, H, D = int(Q_final.shape[0]), int(Q_final.shape[1]), int(Q_final.shape[2])
        S = int(pos_global.shape[0])
        if S == 0:
            raise ValueError("Empty global sequence length S")

        # choose layers
        if layers_sel == "last1":
            layer_ids = [L-1]
        elif layers_sel == "last4":
            layer_ids = list(range(max(L-4, 0), L))
        else:
            layer_ids = list(range(L))
        L_sel = len(layer_ids)

        # derive GQA group size: Hkv taken from K_global[0].shape[0]
        H_kv = int(K_global[layer_ids[0]].shape[0])
        group_size = H // H_kv
        if group_size * H_kv != H:
            # fallback to repeat (rare)
            group_size = 1

        # output per-layer aggregated [S]
        layer_scores = np.empty((L_sel, S), dtype=np.float32)

        # precompute sqrt(D)
        scale = 1.0 / math.sqrt(float(D))

        # per layer
        ell = 0
        for li in layer_ids:
            K_l = K_global[li].astype(np.float32, copy=False)        # [H_kv, S, D] fp32
            Q_l = Q_final[li].astype(np.float32, copy=False)         # [H, D]
            # per head logits -> softmax -> attention over S
            # We'll aggregate across heads as we go
            if agg_heads == "lse":
                # log-sum-exp over heads -> use logits then combine
                head_vals = np.empty((H, S), dtype=np.float32)
                for h in range(H):
                    g = h // group_size
                    q = Q_l[h, :]                                    # [D]
                    z = K_l[g, :, :] @ q                             # [S]
                    z = z * scale
                    a = self._softmax_1d_safe(z)                     # [S]
                    head_vals[h, :] = a
                # aggregate heads: lse or mean; use lse for "lse"
                # log-sum-exp in prob-space ~ sum
                a_heads = np.log(np.sum(head_vals, axis=0) + 1e-30)  # [S]
                a_heads = np.exp(a_heads, dtype=np.float32)          # back to prob-like scale
            else:
                # max over heads (default)
                head_max = np.zeros((S,), dtype=np.float32)
                for h in range(H):
                    g = h // group_size
                    q = Q_l[h, :]
                    z = K_l[g, :, :] @ q                             # [S]
                    z = z * scale
                    a = self._softmax_1d_safe(z)                     # [S]
                    # max across heads
                    head_max = np.maximum(head_max, a)
                a_heads = head_max                                   # [S]

            layer_scores[ell, :] = a_heads
            ell += 1

        # mean across layers
        importance = np.mean(layer_scores, axis=0, dtype=np.float32)  # [S]

        # optional smoothing (simple moving average)
        if smooth_window is not None and smooth_window > 1:
            w = int(smooth_window)
            pad = w // 2
            # pad edges by reflection
            padded = np.pad(importance, (pad, pad), mode="edge")
            csum = np.cumsum(padded, dtype=np.float32)
            sm = (csum[w:] - csum[:-w]) / float(w)
            importance = sm.astype(np.float32, copy=False)

        # diagnostics
        diag = {}
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

    # ---- helper: get last token id+text from a prefill outputs dict ----
    @staticmethod
    def _next_token_from_outputs(outputs: Dict[str, Any], tokenizer) -> Tuple[Optional[int], Optional[str]]:
        logits = outputs.get("logits", None)
        if logits is None:
            return None, None
        if logits.ndim == 2:
            logits = np.expand_dims(logits, 1)  # [B,V] -> [B,1,V]
        tok_id = int(logits.argmax(2)[0, 0])
        tok_text = tokenizer.decode([tok_id], skip_special_tokens=True)
        return tok_id, tok_text

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
        pool_kernel_size: int = 13,     # kept but unused in the global path
        keep_cfg: Optional[KeepConfig] = None,
        layers_sel: str = "all",         # "all"|"last4"|"last1"
    ) -> Dict[str, Any]:
        """
        Run spec prefill, stitch global keys, compute importance and select keep indices (paper-faithful).
        """
        if keep_cfg is None:
            keep_cfg = KeepConfig(strategy="percentage", percentage=0.1, chunk=True, chunk_size=32)

        # 1) Run prefill loop, collect per-chunk outputs & pos ids
        #    We reuse run_prefill's chunk loop by copying logic inline to collect data
        inputs = self.tokenizer(prompt, return_tensors="np", padding=True)
        padded_len = inputs["input_ids"].shape[1]
        num_chunks = -(padded_len // -self._prefill_seq_len)
        padded_len = num_chunks * self._prefill_seq_len

        # pre-allocate logit buffer (as in run_prefill)
        try:
            vocab_size = self._fetch_vocab_size()
            logits_out_placeholder = np.zeros((1, 1, vocab_size), dtype=np.float32)
            self._session.set_buffers({"logits": logits_out_placeholder})
        except Exception:
            pass

        inputs = self.tokenizer(prompt, return_tensors="np", padding="max_length", max_length=padded_len)
        inputs["position_ids"] = np.where(inputs.pop("attention_mask"), np.arange(padded_len), -1)
        inputs.pop("token_type_ids", None)

        # Collect containers
        chunks_keys: List[List[np.ndarray]] = []   # list per-chunk of per-layer key arrays
        chunks_pos: List[np.ndarray] = []          # list per-chunk pos_ids
        chunks_ids: List[np.ndarray] = []          # list per-chunk input_ids slices
        outputs_last = None

        for i in range(num_chunks):
            chunk_inputs = inputs.copy()
            chunk_inputs["input_ids"] = inputs["input_ids"][:, i*self._prefill_seq_len:(i+1)*self._prefill_seq_len]
            chunk_inputs["position_ids"] = inputs["position_ids"][:, i*self._refill_seq_len:(i+1)*self._prefill_seq_len] if False else inputs["position_ids"][:, i*self._prefill_seq_len:(i+1)*self._prefill_seq_len]  # typo guard
            outputs = self._session.run(chunk_inputs)
            outputs_last = outputs  # save last-chunk outputs to get Q_final
            # collect per-layer keys for this chunk
            per_layer_keys: List[np.ndarray] = []
            idx = 0
            while True:
                key_name = f"past_key.{idx}_RetainedState"
                if key_name not in outputs:
                    break
                per_layer_keys.append(outputs[key_name])  # [1,H_kv,S_chunk,D] float16
                idx += 1
            chunks_keys.append(per_layer_keys)
            chunks_pos.append(chunk_inputs["position_ids"].copy())
            # store exact input_ids fed to device for this chunk
            chunks_ids.append(chunk_inputs["input_ids"].copy())

        if outputs_last is None:
            raise RuntimeError("No outputs from prefill; empty prompt?")

        # 2) Fetch Q_final (prefill_queries) from last chunk
        if "prefill_queries" not in outputs_last:
            raise KeyError("prefill_queries not in last-chunk outputs")
        Q_final = outputs_last["prefill_queries"]  # expected [L, H, D] or similar
        if Q_final.ndim == 4 and Q_final.shape[1] == 1:
            Q_final = Q_final[:, 0, :, :]  # squeeze batch if needed

        # 3) Assemble global keys per layer & global pos ids
        K_global, pos_global, ids_global, S_total, H_kv, D = self._assemble_global_keys(
            chunks_keys, chunks_pos, chunks_ids
        )

        # 4) Score global importance respecting GQA; aggregate heads/layers; optional smoothing
        importance, diag = self._score_global_importance(
            Q_final=Q_final.astype(np.float32, copy=False),
            K_global=K_global,
            pos_global=pos_global,
            layers_sel=layers_sel,
            agg_heads="max",
            smooth_window=pool_kernel_size if pool_kernel_size and pool_kernel_size > 1 else None,
        )

        if os.getenv("QEFF_SPEC_ASSERT", ""):
            print(f"[spec:diag] softmax_sum_l0h0={diag.get('softmax_sum_l0h0', None)}")

        # 5) Select global top-% (always keep last real token)
        keep_idx = self._select_global_topk(importance, keep_cfg.percentage, force_last=True)
        keep_idx = keep_idx.astype(np.int64, copy=False)

        return {
            "importance": importance,  # [S_total]
            "keep_idx": keep_idx,      # sorted, includes S_total-1
            "S": S_total,
            "shapes": {
                "prefill_queries": tuple(Q_final.shape),
                "first_key": tuple(K_global[0].shape),
            },
            "ids_global": ids_global,  # [S_total] int64 (exact device-fed tokens)
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
    ) -> Dict[str, Any]:
        """
        Step 4.3: spec prefill+score -> build pruned ids/pos -> run base prefill
        (baseline and pruned) with simple timings.
        """
        # ---- TTFT(spec_only): spec prefill + score + select ----
        t_spec_start = time.perf_counter()
        res = self.prefill_and_score(
            prompt,
            pool_kernel_size=pool_kernel_size,
            keep_cfg=keep_cfg,
            layers_sel=layers_sel,
        )
        t_spec_end = time.perf_counter()
        ttft_spec_only_s = t_spec_end - t_spec_start
        keep_idx = res["keep_idx"]
        S = res["S"]
        ids_global = res.get("ids_global", None)
        if os.getenv("QEFF_SPEC_ASSERT", ""):
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
            if gen_len is not None and int(gen_len) > 0:
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
            pass

        # Always print a compact TTFT summary (ms)
        t_rect = 0
        print(
            f"[4.3] S={S} kept={keep_idx.size} "
            f"TTFT(base_full)={ttft_baseline_s*1000:.1f}ms "
            f"TTFT(spec_only)={t_rect*0+ttft_spec_only_s*1000:.1f}ms ".replace(
                "t_rect*0+", ""
            )
            + f"TTFT(base_pruned_only)={ttft_base_pruned_only_s*1000:.1f}ms "
            + f"TTFT(speculative)={ttft_speculative_s*1000:.1f}ms"
        )

        return {
            "S": S,
            "kept": int(keep_idx.size),
            "keep_idx": keep_idx,
            "ttft_baseline_s": ttft_baseline_s,
            "ttft_spec_only_s": ttft_spec_only_s,
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
    print(
        f"[score] S={S} kept={len(keep_idx)} ({kept_pct:.1f}%) "
        f"prefill_queries={res['shapes']['prefill_queries']} "
        f"first_key={res['shapes']['first_key']} "
        f"last_kept={keep_idx[-1] if keep_idx.size else 'NA'}"
    )


if __name__ == "__main__":
    main()
