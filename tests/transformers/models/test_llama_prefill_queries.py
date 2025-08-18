import torch
from QEfficient.base.common import QEFFCommonLoader
from QEfficient.utils import load_hf_tokenizer

MODEL = "meta-llama/Llama-3.2-1B-Instruct"  # use any small Llama you have access to

# 1) Load model + tokenizer
qeff = QEFFCommonLoader.from_pretrained(MODEL)
tok  = load_hf_tokenizer(MODEL)

# 2) Eval mode; (optional) enable your flag if you added one
qeff.model.eval()
if hasattr(qeff.model, "return_prefill_queries"):
    qeff.model.return_prefill_queries = True

# 3) Prepare a tiny prompt (B=1, short)
text = "Hello world"
ids  = tok(text, return_tensors="pt").input_ids
pos  = torch.arange(ids.shape[1], dtype=torch.long).unsqueeze(0)

# 4) Forward (no ONNX/export—just PyTorch)
with torch.no_grad():
    out = qeff.model(input_ids=ids, position_ids=pos, use_cache=True, return_dict=True)

# 5) Assertions / shape checks
assert hasattr(out, "prefill_queries"), "prefill_queries missing on output"
q = out.prefill_queries
assert q is not None, "prefill_queries is None"

# Accept [L,H,D] (B=1 squeezed) or [L,B,H,D] (if you kept batch in the stack)
if q.ndim == 3:
    L, H, D = q.shape
    B = 1
elif q.ndim == 4:
    L, B, H, D = q.shape
else:
    raise AssertionError(f"Unexpected ndim for prefill_queries: {q.ndim}, shape={tuple(q.shape)}")

# Pull config for sanity checks
cfg = qeff.model.config  # inner LlamaModel config
num_layers = getattr(cfg, "num_hidden_layers", None)
num_heads  = getattr(cfg, "num_attention_heads", None)
head_dim   = getattr(cfg, "hidden_size", None) // num_heads if num_heads and getattr(cfg, "hidden_size", None) else None

print(f"[check] prefill_queries shape={tuple(q.shape)}  (expect ~ [L,H,D] or [L,B,H,D])")
print(f"[check] layers={L} vs cfg.num_layers={num_layers}, heads={H} vs cfg.num_heads={num_heads}, head_dim={D} vs {head_dim}")

assert num_layers is None or L == num_layers, f"L mismatch: got {L}, cfg {num_layers}"
assert num_heads  is None or H == num_heads,  f"H mismatch: got {H}, cfg {num_heads}"
if head_dim is not None:
    assert D == head_dim, f"D mismatch: got {D}, cfg {head_dim}"

# 6) Quick non-zero sanity (shouldn’t be all zeros)
nz = (q.abs() > 0).float().mean().item()
print(f"[check] nonzero fraction ~ {nz:.4f}")
assert nz > 0.0, "prefill_queries appears all zeros"

print("[OK] Step 1 PyTorch verification passed.")

