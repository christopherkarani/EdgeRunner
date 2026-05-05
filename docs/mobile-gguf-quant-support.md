# Mobile GGUF Quant Support Matrix

First-wave mobile support covers K-quants only plus F16/BF16 loader compatibility.
Q1 and IQ-family formats remain separate rollout tracks.

| GGUF quant | Parse | Byte count | Dequant parity | Raw/fused GEMV | End-to-end acceptance |
| --- | --- | --- | --- | --- | --- |
| F16 | yes | yes | CPU conversion | float materialized | not targeted |
| BF16 | yes | yes | CPU conversion | float materialized | loader compatibility |
| Q2_K | yes | yes | Metal parity tests | float materialized | Qwen3 0.6B opt-in |
| Q3_K | yes | yes | Metal parity tests | float materialized | Qwen3 0.6B opt-in |
| Q4_K | yes | yes | Metal parity tests | float materialized | Qwen3 0.6B opt-in |
| Q5_K | yes | yes | Metal parity tests | float materialized | Qwen3 0.6B opt-in |
| Q6_K | yes | yes | Metal parity tests | float materialized; PLE gather supported | Qwen3 0.6B opt-in |
| Q8_0 | yes | yes | Metal parity tests | raw/fused decode path | Qwen3 0.6B pinned and opt-in |

Deferred upstream types are parsed by name where known and rejected explicitly:
`IQ*`, `TQ*`, `MXFP4`, and `NVFP4`.

`GGUF` raw type `41` is preserved as EdgeRunner's existing Bonsai
`Q1_0_G128` path in this wave to avoid changing Bonsai behavior.
