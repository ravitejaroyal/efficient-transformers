import torch
from QEfficient.base.common import QEFFCommonLoader
from QEfficient.utils import load_hf_tokenizer

# Try QEff cache first; fall back to HF if needed
try:
    from QEfficient.transformers.cache_utils import DynamicCache
except Exception:
    from transformers.cache_utils import DynamicCache  # HF fallback

MODEL = "meta-llama/Llama-3.2-1B-Instruct"

# 1) Load model + tokenizer
qeff = QEFFCommonLoader.from_pretrained(MODEL)
tok  = load_hf_tokenizer(MODEL)

qeff.model.eval()

# 2) Tiny prompt (B=1)
text = "Hello world"
ids  = tok(text, return_tensors="pt").input_ids
S    = ids.shape[1]
pos  = torch.arange(S, dtype=torch.long).unsqueeze(0)      # [1, S]
cache_pos = torch.arange(S, dtype=torch.long).unsqueeze(0) # [1, S]

# 3) Empty cache like runtime does
pkv = DynamicCache()

# 4) Forward – **use_cache=True** and pass cache_position + cache object
with torch.no_grad():
    out = qeff.model(
        input_ids=ids,
        position_ids=pos,
        past_key_values=pkv,
        cache_position=cache_pos,
        use_cache=True,
        return_dict=True,
    )

# 5) Read the captured queries
assert hasattr(out, "prefill_queries"), "prefill_queries missing on output"
q = out.prefill_queries
assert q is not None, "prefill_queries is None"

# Accept [L,H,D] (B=1 squeezed) or [L,B,H,D] if you kept batch
if q.ndim == 3:
    L, H, D = q.shape
elif q.ndim == 4:
    L, B, H, D = q.shape
else:
    raise AssertionError(f"Unexpected ndim for prefill_queries: {q.ndim}, shape={tuple(q.shape)}")

cfg = qeff.model.config
print(f"[OK] prefill_queries shape={tuple(q.shape)}  "
      f"(layers={getattr(cfg, 'num_hidden_layers', None)}, "
      f"heads={getattr(cfg, 'num_attention_heads', None)})")
