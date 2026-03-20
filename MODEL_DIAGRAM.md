# Current Baseline Model Diagram

This diagram reflects the current default MLX baseline in `train_gpt_mlx.py`.

Run configuration used in the recent smoke test:
- `17,059,912` parameters
- `9` layers
- width `512`
- `8` attention heads
- `4` KV heads
- vocab size `1024`
- tied embeddings

```mermaid
flowchart TD
  A["Input token IDs"] --> B["Token embedding<br/>1024 vocab -> 512 dim"]
  B --> C["RMSNorm"]
  C --> X0["Save x0<br/>(original embedding stream)"]

  subgraph ENC["Encoder half: 4 blocks"]
    E0["Block 0<br/>resid_mix(x, x0)<br/>RMSNorm -> causal GQA attention<br/>RMSNorm -> ReLU^2 MLP"]
    E1["Block 1<br/>same"]
    E2["Block 2<br/>same"]
    E3["Block 3<br/>same"]
    E0 --> E1 --> E2 --> E3
  end

  subgraph DEC["Decoder half: 5 blocks"]
    D0["Add learned skip 3<br/>then Block 4"]
    D1["Add learned skip 2<br/>then Block 5"]
    D2["Add learned skip 1<br/>then Block 6"]
    D3["Add learned skip 0<br/>then Block 7"]
    D4["No skip left<br/>Block 8"]
    D0 --> D1 --> D2 --> D3 --> D4
  end

  X0 --> E0
  E3 --> D0
  D4 --> F["Final RMSNorm"]
  F --> G["Tied output projection<br/>512 -> 1024 vocab"]
  G --> H["Logit softcap<br/>30 * tanh(logits / 30)"]
  H --> I["Cross-entropy loss / val_bpb eval"]

  E0 -.store.-> S0["skip 0"]
  E1 -.store.-> S1["skip 1"]
  E2 -.store.-> S2["skip 2"]
  E3 -.store.-> S3["skip 3"]
  S3 -.reuse.-> D0
  S2 -.reuse.-> D1
  S1 -.reuse.-> D2
  S0 -.reuse.-> D3
```

Notes:
- Each block contains learned `resid_mix`, `attn_scale`, and `mlp_scale`.
- Attention uses separate Q/K/V projections, grouped-query attention, RMSNorm on Q and K, RoPE, and causal scaled dot-product attention.
- The MLP is `512 -> 1024 -> 512` with `ReLU^2`.
