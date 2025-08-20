from __future__ import annotations
import numpy as np
from typing import Any, Dict, Optional, Tuple, List
from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.generation.text_generation_inference import write_io_files

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
        # guard attributes referenced by runtime-style code paths
        self._write_io_dir = None
        self.full_batch_size = None
        self.batch_index = None

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
    def run_prefill(self, prompt, generation_len, prefill_logit_bs=1, decode_batch_id=None):
        """
        Runs prefill for a given prompt and generation length.

        This method tokenize the prompt and calculates the padded length and number of chunks. Calculates the
        maximum generation length and fetches the generation length. If a batch index for prefill is provided, it sets the batch index in the inputs. The method then runs prefill for each chunk and updates the inputs and outputs.

        Args:
            prompt (str): The prompt for which to run prefill.
            generation_len (int): The generation length.
            prefill_logit_bs (int, optional): The prefill logit batch size. Defaults to 1.

        Returns:
            outputs (dict): The outputs of the prefill.
            position_ids (array): The position IDs.
            generation_len (int): The generation length.
        """
        # Run prefill
        inputs = self._tokenizer(prompt, return_tensors="np", padding=True)
        position_ids = inputs["attention_mask"].sum(1, keepdims=True)
        padded_len = inputs["input_ids"].shape[1]
        num_chunks = -(padded_len // -self._prefill_seq_len)  # ceil divide without float
        padded_len = num_chunks * self._prefill_seq_len  # Convert to a multiple of prompt_len

        # Initialize variables specific to request
        # Calculate the max generation length.
        max_gen_len = self._ctx_len - position_ids.max()
        generation_len = self._fetch_generation_len(generation_len, max_gen_len)

        # Set the prefill logic buffer
        self._fetch_vocab_size()
        logits_out_placeholder = np.zeros((prefill_logit_bs, 1, self._vocab_size), dtype=np.float32)
        self._session.set_buffers({"logits": logits_out_placeholder})

        inputs = self._tokenizer(prompt, return_tensors="np", padding="max_length", max_length=padded_len)
        inputs["position_ids"] = np.where(inputs.pop("attention_mask"), np.arange(padded_len), -1)
        inputs.pop("token_type_ids", None)

        if decode_batch_id is not None:
            inputs["batch_index"] = decode_batch_id

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
                write_io_files(inputs, outputs, self._write_io_dir, "prefill", "aic_batch_io", True, False)
        return (
            outputs,
            position_ids,
            generation_len,
        )

    def prefill_from_ids(
        self,
        final_ids: np.ndarray,
        final_pos: np.ndarray,
        prefill_logit_bs: int = 1,
    ) -> Tuple[Dict[str, Any], int, int, int]:
        """
        Run prefill given prebuilt input_ids and position_ids arrays.
        Mirrors run_prefill variable names & chunking/padding/dtypes.
        Returns: (outputs, S, padded_len, num_chunks)
        """
        # Ensure shapes/dtypes
        ids = final_ids.astype(np.int64, copy=False)   # [1,K]
        pos = final_pos.astype(np.int64, copy=False)   # [1,K] absolute positions
        S = int(ids.shape[1])

        # ceil divide
        padded_len = S
        num_chunks = -(padded_len // -self._prefill_seq_len)
        padded_len = num_chunks * self._prefill_seq_len

        # Build padded inputs (same policy as tokenizer path)
        pad_id = getattr(self._tokenizer, "pad_token_id", getattr(self._tokenizer, "eos_token_id", 0))
        if padded_len > S:
            pad_width = padded_len - S
            inputs = {
                "input_ids":    np.pad(ids, ((0, 0), (0, pad_width)), constant_values=pad_id).astype(np.int64, copy=False),
                "position_ids": np.pad(pos, ((0, 0), (0, pad_width)), constant_values=-1).astype(np.int64, copy=False),
            }
        else:
            inputs = {"input_ids": ids, "position_ids": pos}

        # Preallocate logits (parity with run_prefill)
        try:
            vocab_size = self._fetch_vocab_size()
            logits_out_placeholder = np.zeros((prefill_logit_bs, 1, vocab_size), dtype=np.float32)
            self._session.set_buffers({"logits": logits_out_placeholder})
        except Exception:
            pass

        # Chunk loop (names & slicing identical)
        for i in range(num_chunks):
            chunk_inputs = inputs.copy()
            chunk_inputs["input_ids"]    = inputs["input_ids"][:,    i*self._prefill_seq_len:(i+1)*self._prefill_seq_len].astype(np.int64, copy=False)
            chunk_inputs["position_ids"] = inputs["position_ids"][:, i*self._prefill_seq_len:(i+1)*self._prefill_seq_len].astype(np.int64, copy=False)
            outputs = self._session.run(chunk_inputs)

        return outputs, S, padded_len, num_chunks

    # --- DECODE: mirror text_generation_inference.run_decode (single-batch, single-step) ---
    def prepare_decode_inputs(self):
        """
        This function creates the decode inputs.

        Returns:
            dict: The decode inputs.
        """
        batch_size = self.full_batch_size if self.full_batch_size is not None else self.batch_size
        decode_inputs = {}
        decode_inputs["input_ids"] = self.decode_input_ids
        decode_inputs["position_ids"] = self.decode_pos_ids
        if self.batch_index is not None:
            decode_inputs["batch_index"] = self.batch_index

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

    def run_decode(self, decode_inputs, generation_len, streamer: Optional[transformers.TextStreamer] = None):
        """
        Default method for running decode. Executes the decoding process for a given set of inputs and a specified generation length.

        Enters a loop that continues until all sequences are finished or the maximum generation length is reached. In each iteration, it runs the session with the decode inputs, prepares the inputs for the next iteration and checks if all sequences are finished.

        Args:
            decode_inputs (dict): The initial inputs for decoding. This should be a dictionary containing 'input_ids' and 'position_ids'.
            generation_len (int): Max allowed length for generating tokens. The decoding process will be terminated  when generation length is reached.
            streamer (transformers.TextStreamer): TextStreamer object to print decoded tokens to console.
        Returns:
            num_token (int): The number of tokens processed in the decoding process.
        """
        finished_sequences = decode_inputs["input_ids"] == self._tokenizer.eos_token_id
        num_token = 0
        for num_token in range(1, generation_len):
            if streamer:
                streamer.put(decode_inputs["input_ids"][0])
            outputs = self._session.run(decode_inputs)

            if self._write_io_dir is not None:
                write_io_files(decode_inputs, outputs, self._write_io_dir, "decode", "aic_batch_io", True, False)
                self._write_io_dir = None

            # Prepare inputs for next iteration
            decode_inputs["input_ids"] = outputs["logits"].argmax(2)
            decode_inputs["position_ids"][:, -1] += 1
            self.generated_ids[:, num_token] = decode_inputs["input_ids"][:, -1]
            finished_sequences |= decode_inputs["input_ids"] == self._tokenizer.eos_token_id

            if finished_sequences.all():
                break
        return num_token


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
