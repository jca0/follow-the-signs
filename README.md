# Follow the Signs

LLM-guided indoor navigation: an agent explores a partially observable building and reads signs and room numbers along the way to find a goal room. This branch is the grid-world simulation; the [`spot` branch](https://github.com/jca0/follow-the-signs/tree/spot) runs the same approach on a Boston Dynamics Spot robot.

## Methods

- [`ours.py`](ours.py) — our sign-following navigation agent
- [`navgpt.py`](navgpt.py) — NavGPT-style baseline
- [`llm_only.py`](llm_only.py) — LLM-only baseline

Prompts for each method live in [`prompts/`](prompts), and environments are defined in [`helpers/`](helpers).

## Setup

```bash
pip install -r requirements.txt
```

Add your API keys to [`config.yaml`](config.yaml) (placeholders are checked in).
