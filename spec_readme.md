# Implementation vs. algorithm (look‑ahead = 0)

- Skip unused past-state I/O – the speculator session drops all `past_*` inputs and `past_value.*` outputs so that only `prefill_queries` and retained keys are fetched later

## Chunked prefill & query capture

- The prompt is padded to a multiple of the model’s prefill length, split into `num_chunks`, and buffers for `logits` and `prefill_queries` are bound once
- Retained-state key bindings are skipped for every chunk, then re-enabled only on the last chunk; positions and input IDs are cached per chunk; the final chunk yields `Q_final` (last-token queries) and per-layer retained keys

## Assemble global keys

- After prefill, per-chunk keys and positions are merged: pads (`position_ids == -1`) are dropped, valid token IDs concatenated, and each layer’s retained-state is sliced to `[H_kv, S_total, D]` to form `K_global`

## Host-side scoring

- For layers selected (`all`, `last4`, `last1`), derive the GQA group size and map each query head `h` to its KV head `g = h // (H/H_kv)`
- Compute attention logits with a matrix–vector product `z = K_l[g,:,:] @ q`, scale by `1/√D`, and apply a numerically-stable softmax over the sequence
- Aggregate importance by taking the max across heads and the mean across layers, yielding one score per token

## Top-k selection

- Choose a fraction of tokens with highest scores, always forcing retention of the final token so decode remains valid

## Scoring orchestration

- `prefill_and_score` glues the steps above (run prefill if needed → assemble keys → score → select) and reports invariants for debugging

## Faithfulness to the paper

With look‑ahead `N=0`, the implementation mirrors Algorithm 1 by using the last‑token query vectors (`prefill_queries`) as surrogates for importance, multiplying them against global keys, applying softmax per head, aggregating (head‑wise max, layer mean), and selecting the highest‑importance tokens. Chunk smoothing is performed via `pool_kernel_size`, but chunk‑level top‑k is not applied—the code keeps tokens individually rather than blockwise.

## Why each step matters

- **Assembling global keys:** removes padding and reconciles chunked prefill into a contiguous `[H_kv,S_total,D]` buffer so scoring covers the entire prompt.
- **Scoring global importance:** measures how strongly the final query attends to each past token; higher softmax probabilities indicate greater contextual relevance.
- **GQA mapping:** ensures query heads align with the correct KV heads when the model uses grouped-query attention.
- **Selecting global top‑k:** prunes low-importance tokens, reducing context length for the base model while forcing inclusion of the last token to preserve causality.

---

## Notes

- Chunk-based block selection (`keep_cfg.chunk`) exists but is bypassed in the current host-scoring path, which selects tokens individually after smoothing.

---

**Are you saying, in our repo, we are assembling all the past keys of the prompt and multiply them with last query of input. Let’s say if prompt length is 100 , then we are multiplying all the 100 keys with last query of input.**

---

## Step‑by‑step toy example of host‑side scoring

Let the final‑token queries and retained keys for a single layer be tiny:

- Query heads `H = 4`, KV heads `H_kv = 2` ⇒ group size `H/H_kv = 2`
- Sequence length `S = 3`, head dim `D = 2`
- Scale `1 / √D = 1/√2 ≈ 0.7071`

### 1. Data

`K_l[g, s, d]` (shape `[H_kv, S, D]`):

| g | s | K |
|---|---|---|
|0|0|[1,0]|
|0|1|[0,1]|
|0|2|[1,1]|
|1|0|[1,2]|
|1|1|[0,2]|
|1|2|[2,2]|

`Q_l[h, d]` (shape `[H, D]`):

| h | Q |
|---|---|
|0|[1,0]|
|1|[0,1]|
|2|[1,1]|
|3|[2,1]|

### 2. GQA mapping

```
group_size = H // H_kv = 4 // 2 = 2
g(h) = h // group_size = h // 2

h: 0 1 2 3
g: 0 0 1 1
```

### 3. Dot products `z = K_l[g,:,:] @ Q_l[h,:]`

| h | g | z (before scale) |
|---|---|------------------|
|0|0|[1,0,1]|
|1|0|[0,1,1]|
|2|1|[3,2,4]|
|3|1|[4,2,6]|

Apply scale `0.7071`:

| h | z·scale |
|---|---------|
|0|[0.707, 0, 0.707]|
|1|[0, 0.707, 0.707]|
|2|[2.121, 1.414, 2.828]|
|3|[2.828, 1.414, 4.243]|

### 4. Softmax over `S` (max‑trick)

Example for `h = 2`:

- Shift by max (2.828): `[−0.707, −1.414, 0]`
- Exponentiate: `[e^{-0.707}=0.493, e^{-1.414}=0.243, 1]`
- Normalize: `sum = 1.736 ⇒ softmax = [0.284, 0.140, 0.576]`

All heads:

| s | h0 | h1 | h2 | h3 |
|---|----|----|----|----|
|0|0.401|0.198|0.284|0.187|
|1|0.198|0.401|0.140|0.045|
|2|0.401|0.401|0.576|0.768|

### 5. Head‑wise max

For each position `s`, take `max` across heads:

```
head_max = [
  max(0.401, 0.198, 0.284, 0.187),  # s0 → 0.401
  max(0.198, 0.401, 0.140, 0.045),  # s1 → 0.401
  max(0.401, 0.401, 0.576, 0.768)   # s2 → 0.768
]
= [0.401, 0.401, 0.768]
```

### 6. Layer mean

Only one layer ⇒ `importance = head_max`:

```
importance = [0.401, 0.401, 0.768]
```

### 7. Top‑k selection (keep fraction ½)

```
S = 3, k = ceil(0.5 * 3) = 2
argpartition → indices {2 (0.768), 0 (0.401)}
sort → [0, 2]
force-last: last token index 2 already kept
keep_idx = [0, 2]
```

Token positions `[0, 2]` are retained; token `1` is pruned.

---

### Recap of steps

1. Map query head `h` to kv head `g = h // (H / H_kv)`
2. For each head:
   - `z = K_l[g,:,:] @ Q_l[h,:]`            # `[S]`
   - `z *= 1/√D`
   - `a = softmax(z)`                       # mask pads before softmax
3. `head_max[s] = max_h a_h[s]`             # `[S]`
4. `importance[s] = mean_layers head_max`   # here: single layer
5. `keep_idx = top_k(importance, k)`        # argpartition + sort
6. Force include last index `S-1`

This miniature example mirrors the implementation in `spec_prefill.py`: GQA grouping, dot products, numerically stable softmax, head-wise max, layer mean, and top-k with forced last token.
