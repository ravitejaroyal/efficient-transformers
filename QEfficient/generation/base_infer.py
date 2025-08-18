from __future__ import annotations
import numpy as np
from typing import Any, Dict, Optional, Tuple, List
from QEfficient.generation.cloud_infer import QAICInferenceSession

class BaseInferenceEngine:
    """
    Mirror QEfficient/generation/text_generation_inference.py prefill/decode paths
    for a base model QPC. Reuses QAICInferenceSession and keeps variable names identical.
    """

    def __init__(
        self,
        base_qpc_path: str,
        tokenizer,
        ctx_len: int,
        prefill_seq_len: int,
        device_ids: Optional[List[int]] = None,
    ) -> None:
        self._session = QAICInferenceSession(base_qpc_path, device_ids=device_ids)
        self._tokenizer = tokenizer
        # _set_tokenizer_params() parity
        if getattr(self._tokenizer, "padding_side", "right") != "right":
            self._tokenizer.padding_side = "right"
        if getattr(self._tokenizer, "pad_token_id", None) is None:
            self._tokenizer.pad_token_id = getattr(self._tokenizer, "eos_token_id", 0)
        self._ctx_len = int(ctx_len)
        self._prefill_seq_len = int(prefill_seq_len)
        # runtime-like caches for decode
        self.decode_input_ids: Optional[np.ndarray] = None
        self.decode_pos_ids: Optional[np.ndarray] = None
        self._vocab_size: Optional[int] = None
        self.batch_size: int = 1            # single-batch helper
        self._decode_seq_len: int = 1       # single-step decode per call

    # --- session probing helpers (runtime parity) ---
    def _fetch_vocab_size(self) -> int:
        if self._vocab_size is not None:
            return self._vocab_size
        bidx = self._session.binding_index_map["logits"]
        dims = list(self._session.bindings[bidx].dims)
        self._vocab_size = int(dims[-1])
        return self._vocab_size

    def _fetch_generation_len(self, generation_len: Optional[int], max_gen_len: int) -> int:
        if generation_len is None:
            return int(max_gen_len)
        return int(min(int(generation_len), int(max_gen_len)))

    # --- PREFILL: mirror text_generation_inference.run_prefill ---
    def run_prefill(
        self,
        prompt: str,
        generation_len: Optional[int] = None,
        prefill_logit_bs: int = 1,
        decode_batch_id: Optional[np.ndarray] = None,
    ) -> Tuple[Dict[str, Any], np.ndarray, int]:
        """
        Runs prefill for a given prompt and generation length.
        Returns: (outputs, position_ids, generation_len)
        """

        # First pass (padding=True)
        inputs = self._tokenizer(prompt, return_tensors="np", padding=True)
        position_ids = inputs["attention_mask"].sum(1, keepdims=True)
        padded_len = inputs["input_ids"].shape[1]
        # ceil divide to chunk count
        num_chunks = -(padded_len // -self._prefill_seq_len)
        # Convert to a multiple of prefill_seq_len
        padded_len = num_chunks * self._prefill_seq_len

        # Compute generation length parity
        max_gen_len = self._ctx_len - position_ids.max()
        generation_len = self._fetch_generation_len(generation_len, max_gen_len)

        # Preallocate logits buffer (parity with runtime)
        try:
            vocab_size = self._fetch_vocab_size()
            logits_out_placeholder = np.zeros((prefill_logit_bs, 1, vocab_size), dtype=np.float32)
            self._session.set_buffers({"logits": logits_out_placeholder})
        except Exception:
            pass

        # Second pass: pad to padded_len and build position_ids with np.where
        inputs = self._tokenizer(prompt, return_tensors="np", padding="max_length", max_length=padded_len)
        inputs["position_ids"] = np.where(
            inputs.pop("attention_mask"),
            np.arange(padded_len, dtype=np.int64),
            -1
        )
        inputs.pop("token_type_ids", None)

        if decode_batch_id is not None:
            inputs["batch_index"] = decode_batch_id

        # Chunk loop (same names & slicing)
        for i in range(num_chunks):
            chunk_inputs = inputs.copy()
            chunk_inputs["input_ids"]    = inputs["input_ids"][:,    i*self._prefill_seq_len:(i+1)*self._prefill_seq_len].astype(np.int64, copy=False)
            chunk_inputs["position_ids"] = inputs["position_ids"][:, i*self._prefill_seq_len:(i+1)*self._prefill_seq_len].astype(np.int64, copy=False)
            outputs = self._session.run(chunk_inputs)

        return outputs, position_ids, generation_len

    # --- DECODE: mirror text_generation_inference.run_decode (single-batch, single-step) ---
    def prepare_decode_inputs(self) -> Dict[str, np.ndarray]:
        """
        Create the decode inputs dict (single-batch, single-step) from self.decode_input_ids/pos.
        Parity with runtime non-TLM path.
        """
        if self.decode_input_ids is None or self.decode_pos_ids is None:
            raise RuntimeError("Call update_decode_seed(...) first to set decode_input_ids/pos.")
        decode_inputs: Dict[str, np.ndarray] = {}
        decode_inputs["input_ids"] = self.decode_input_ids.astype(np.int64, copy=False)
        decode_inputs["position_ids"] = self.decode_pos_ids.astype(np.int64, copy=False)
        return decode_inputs

    def update_decode_seed(self, outputs: Dict[str, Any], position_ids: np.ndarray) -> None:
        """
        Seed the first decode step using prefill outputs (parity with update_decode_input).
        """
        logits = outputs["logits"]
        if logits.ndim == 2:
            logits = np.expand_dims(logits, 1)   # [B,1,V]
        next_token_id = logits.argmax(2)         # [B,1]
        # Store the generated values
        self.decode_input_ids = next_token_id.astype(np.int64, copy=False)
        self.decode_pos_ids = position_ids.astype(np.int64, copy=False)

    def run_decode(
        self,
        decode_inputs: Dict[str, np.ndarray],
        generation_len: int,
    ) -> Tuple[int, np.ndarray]:
        """
        Default method for running decode (single-batch).
        Returns: (num_token, generated_ids) where generated_ids is [1, generation_len] (including seed).
        """
        generated_ids: List[np.ndarray] = []
        generated_ids.append(decode_inputs["input_ids"][:, -1])

        finished_sequences = (decode_inputs["input_ids"] == self._tokenizer.eos_token_id)
        num_token = 0
        for num_token in range(1, generation_len):
            outputs = self._session.run(decode_inputs)
            logits = outputs["logits"]
            if logits.ndim == 2:
                logits = np.expand_dims(logits, 1)  # [B,1,V]
            argmax = logits.argmax(2)                # [B,1]

            # Prepare inputs for next iteration (parity with runtime)
            decode_inputs["input_ids"] = argmax.astype(np.int64, copy=False)
            decode_inputs["position_ids"][:, -1] += 1
            generated_ids.append(argmax[:, -1])

            finished_sequences |= (argmax == self._tokenizer.eos_token_id)
            if finished_sequences.all():
                break

        return num_token, np.stack(generated_ids, axis=1)


# --------------------- simple __main__ validator ---------------------
if __name__ == "__main__":
    import argparse
    import sys
    from QEfficient.utils import load_hf_tokenizer

    parser = argparse.ArgumentParser(
        description="Base inference parity check: run_prefill + run_decode (mirrors runtime)."
    )
    parser.add_argument("--base-qpc", required=True, help="Path to base model QPC directory (…/qpc)")
    parser.add_argument("--model-name", default="meta-llama/Meta-Llama-3-70B", help="HF tokenizer card for base")
    parser.add_argument("--prompt", default="Hello from base prefill on AI100.", help="Prompt string")
    parser.add_argument("--prompt-len", type=int, default=16, help="prefill chunk length")
    parser.add_argument("--ctx-len", type=int, default=128, help="context length")
    parser.add_argument("--gen-len", type=int, default=8, help="number of decode steps (tokens) to run")
    parser.add_argument("--device-ids", default="[0]", help="Device IDs like [0] or [0,1]")
    parser.add_argument("--prefill-logit-bs", type=int, default=1, help="logits placeholder batch size")
    args = parser.parse_args()

    try:
        dev_ids = [int(x) for x in args.device_ids.strip("[] ").split(",") if x.strip() != ""]
    except Exception:
        print(f"[warn] Could not parse --device-ids={args.device_ids!r}, defaulting to [0]")
        dev_ids = [0]

    tok = load_hf_tokenizer(args.model_name)
    eng = BaseInferenceEngine(
        base_qpc_path=args.base_qpc,
        tokenizer=tok,
        ctx_len=int(args.ctx_len),
        prefill_seq_len=int(args.prompt_len),
        device_ids=dev_ids,
    )

    # 1) prefill
    outputs, position_ids, generation_len = eng.run_prefill(
        args.prompt, generation_len=None, prefill_logit_bs=args.prefill_logit_bs
    )
    if "logits" not in outputs:
        print("[fail] logits missing from prefill outputs"); sys.exit(2)

    # 2) seed decode and run a few steps
    eng.update_decode_seed(outputs, position_ids)
    decode_inputs = eng.prepare_decode_inputs()
    num_token, gen_ids = eng.run_decode(decode_inputs, generation_len=int(args.gen_len))
    print(f"[base] ran decode steps: {num_token}  generated_ids shape: {gen_ids.shape}")

    sys.exit(0)
