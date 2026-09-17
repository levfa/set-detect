# setdetect

Prototyping framework for `setdetect`.

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/). All commands run from `py/`:

```
cd py
uv sync
```

## Commands

- `uv run texturecan download -o ~/data/huggingface/nyuuzyou/texturecan`: downloads the `nyuuzyou/texturecan` texture dataset (~3 GB) from Hugging Face into the given directory (`-o/--local-dir` is required).
