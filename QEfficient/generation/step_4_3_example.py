from QEfficient.generation.base_infer import BaseInferenceEngine
from QEfficient.generation.spec_prefill import KeepConfig, SpecPrefillEngine
from QEfficient.utils import load_hf_tokenizer

spec_qpc = "/abs/path/to/spec/qpc"  # 8B spec with prefill_queries
base_qpc = "/abs/path/to/base/qpc"  # 70B base
tok = load_hf_tokenizer("meta-llama/Meta-Llama-3-8B")  # same tokenizer for both

spec = SpecPrefillEngine(spec_qpc, tok, ctx_len=128, prefill_seq_len=16, device_ids=[0])
base = BaseInferenceEngine(base_qpc, tok, ctx_len=128, prefill_seq_len=16, device_ids=[0])

prompt = " ".join(["This is a longer prompt to force multiple chunks."] * 10)
keep_cfg = KeepConfig(strategy="percentage", percentage=0.1, chunk=True, chunk_size=32)

ret = spec.prune_and_base_prefill(base, prompt, pool_kernel_size=13, keep_cfg=keep_cfg)
print(ret)
