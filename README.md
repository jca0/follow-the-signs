# Follow the Signs

[**Paper (arXiv)**](https://arxiv.org/abs/2601.06652) · [**Video**](https://drive.google.com/file/d/1a3aqsFjCBe9wdSmbYcf_SGFyG7tKmRmZ/view?usp=sharing)

LLM-guided navigation in structured environments: an agent explores a partially observable building and reads signs and room numbers along the way to find a goal room. This branch is the simulation. The [`spot` branch](https://github.com/jca0/follow-the-signs/tree/spot) runs the same approach on a Boston Dynamics Spot robot.

## Repo overview

- [`ours.py`](ours.py) — our navigation agent
- [`navgpt.py`](navgpt.py) — NavGPT-style baseline
- [`llm_only.py`](llm_only.py) — LLM-only baseline

Prompts for each method live in [`prompts/`](prompts), and environments are defined in [`helpers/`](helpers).
