# Interpretation cost

Estimated cost to interpret the Qwen 32k features:

| Model | Input / 1M | Output / 1M | Estimated cost |
|---|---:|---:|---:|
| `gpt-5.6-sol` | $4 | $20 | **$367** |
| `gpt-5.6-terra` | $2 | $12 | **$185** |
| `gpt-5.6-luna` | $0.20 | $1.20 | **$18** |
| `claude-fable-5` | $10 | $50 | **$918** |
| `claude-opus-5` | $5 | $25 | **$459** |
| `claude-sonnet-5` | $2 | $10 | **$184** |
| `claude-haiku-4-5` | $1 | $5 | **$92** |

I used the Gemma 4 tokenizer for the estimates so they may differ from real costs.

> [!TIP]
> Too expensive? Do it locally! It works really well, follow the guide here: [interpret-locally.md](docs/training-guides/interpret-locally.md)

