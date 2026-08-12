# Follow the Signs — Spot

Runs our sign-following, LLM-guided navigation agent on a Boston Dynamics Spot robot. The simulation version lives on the [`main` branch](https://github.com/jca0/follow-the-signs/tree/main).

## Methods

- [`spot/spot_ours.py`](spot/spot_ours.py) — our sign-following navigation agent on Spot
- [`spot/spot_frontier.py`](spot/spot_frontier.py) — frontier-exploration baseline

## Setup

```bash
pip install -r requirements.txt
export BOSDYN_CLIENT_USERNAME=<your spot username>
export BOSDYN_CLIENT_PASSWORD=<your spot password>
export HOSTNAME=<your spot ip>
```

Add your API keys to [`config.yaml`](config.yaml) (placeholders are checked in).
