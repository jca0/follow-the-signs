import ast
import re
import os
import sys

from openai import AzureOpenAI
from typing import Iterable, Union, Tuple, List
from pydantic import BaseModel, Field

from helpers.env_updated import *
from helpers.utils import *
from helpers.navgpt_agent_maps import *
from helpers.large_envs import *

endpoint = "https://llm-nav.openai.azure.com/"
deployment = "gpt-4o"

subscription_key = "YOUR_AZURE_OPENAI_API_KEY"
api_version = "2025-03-01-preview"

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)

FINISH_TOKENS = {"finish", "done", "stop"}
HistoryItem = Union[str, Tuple[str, str, Union[str, Tuple[int,int]]], List]
history = []
BUFFER_SIZE = 3  # recent items kept verbatim; older -> summarize
MAX_HISTORY_LEN = 100  # hard cap to avoid unbounded growth

def summarize_history(history: Iterable[HistoryItem], max_items: int = 10, max_chars: int = 800) -> str:
    """Return a compact summary of recent actions only, no LLM calls.
    - Keeps up to max_items from the provided history (assumed newest-first)
    - Extracts just the action field when tuple/list shaped like (obs, thought, action)
    - Truncates to max_chars
    """
    if history is None:
        return ""
    items = list(history)[:max_items]
    actions: List[str] = []
    for h in items:
        try:
            if isinstance(h, (list, tuple)) and len(h) >= 3:
                actions.append(str(h[2]))
            else:
                actions.append(str(h))
        except Exception:
            actions.append(str(h))
    summary = " -> ".join(actions)
    return summary[:max_chars]

def parse_action(s):
    s = str(s).strip()
    if s.strip('"\'').lower() in FINISH_TOKENS:
        return "Finish"
    try:
        v = ast.literal_eval(s)
    except Exception:
        m = re.search(r'\(?\s*(-?\d+)\s*,\s*(-?\d+)\s*\)?', s)
        if not m:
            return s
        v = (int(m.group(1)), int(m.group(2)))
    if isinstance(v, (list, tuple)) and len(v) == 2:
        return (int(v[0]), int(v[1]))
    if isinstance(v, str) and v.strip('"\'').lower() in FINISH_TOKENS:
        return "Finish"
    return v

class NavDecision(BaseModel):
    thought: str = Field(description="Reasoning for choosing the action")
    action: str = Field(description="Either 'Finish' or one of the candidate IDs")

def query_llm(O_text, H_text, candidates, valid, goal):
    """Build a compact prompt and parse a structured response.
    Clips large fields to avoid hitting model context limits.
    """
    def clip(s, n):
        try:
            s = str(s)
        except Exception:
            s = repr(s)
        return s if len(s) <= n else s[:n]

    # Normalize inputs
    O_text_c = clip(O_text, 4000)
    H_text_c = clip(H_text, 1000)

    if isinstance(candidates, str):
        candidates_c = clip(candidates, 4000)
    else:
        try:
            candidates_c = "\n".join(
                f"- {cid}: angle={ang}, dist~{dist}m" for cid, ang, dist in candidates
            )
        except Exception:
            candidates_c = str(candidates)
        candidates_c = clip(candidates_c, 4000)

    try:
        valid_list = sorted(list(valid))
    except Exception:
        valid_list = [valid]
    valid_text = str(valid_list[:64])

    with open("prompts/navgpt.txt", "r") as f:
        prompt_template = f.read()

    prompt = prompt_template.format(
        O_text=O_text_c,
        H_text=H_text_c,
        candidates=candidates_c,
        valid=valid_text,
        goal=str(goal),
    )

    response = client.responses.parse(
        model=deployment,
        input=prompt,
        text_format=NavDecision,
    )
    return response.output_parsed

def run_agent(seen_occupancy, seen_semantic, start, goal, timeout=250):
    steps = 0
    agent_pos = start
    heading = "N"
    goal_found = False

    while steps < timeout:
        steps += 1
        print(f"\nSTEP {steps}:")
        print("Agent position:", agent_pos)
        seen_occupancy.update_with_slice(agent_pos, k)
        seen_semantic.update_with_slice(agent_pos, k)
        O_text = seen_semantic.get_slice(agent_pos, k)
        candidates = seen_occupancy.get_candidates(agent_pos, heading)
        valid = {cid for cid, _, _ in candidates}

        if not candidates:
            print("No candidates found")
            return -1

        # Build bounded history text
        raw = history[:BUFFER_SIZE]
        older = history[BUFFER_SIZE:]
        if len(history) > MAX_HISTORY_LEN:
            del history[MAX_HISTORY_LEN:]
        H_text = summarize_history(older) if older else ""
        H_text = f"{H_text} {raw}".strip()

        candidates_str = "\n".join([f"- {cid}: angle={ang}, dist~{dist}m" for cid, ang, dist in candidates])
        llm_decision = query_llm(O_text, H_text, candidates_str, valid, goal)
        thought, action_raw = llm_decision.thought, llm_decision.action
        action = parse_action(action_raw)
        print("Thought:", thought)
        print('Action:', action)

        if action == "Finish":
            print("Found goal!")
            goal_found = True
            goal_pos = seen_semantic.find_label(goal)
            try:
                path = seen_occupancy.astar(agent_pos, goal_pos)
            except Exception as e:
                path = []
            if path:
                return steps + len(path)
            else:
                return steps

        if action not in valid:
            print("Invalid action, retrying...")
            llm_decision = query_llm(O_text, H_text, candidates_str, valid, goal)
            thought, action_raw = llm_decision.thought, llm_decision.action
            action = parse_action(action_raw)
            if action not in valid:
                print("Invalid action")
                return -1

        history.insert(0, (O_text, thought, action))
        pos_prev = agent_pos
        agent_pos = action

        dr, dc = agent_pos[0]-pos_prev[0], agent_pos[1]-pos_prev[1]
        DIR = {(-1,0):"N",(0,1):"E",(1,0):"S",(0,-1):"W"}
        if (dr,dc) not in DIR:
            print("Non-adjacent move")
            return -1
        heading = DIR[(dr,dc)]

    if not goal_found:
        return -1

if __name__ == "__main__":
    log_file = open('logs/navgpt.log', 'w')
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)

    seeds = []
    with open('seeds/large/bldg4_seeds.txt', 'r') as f:
        seeds = [ast.literal_eval(line) for line in f if line.strip()]

    env = Bldg4()
    occupancy_grid = env.occupancy_grid
    semantic_grid = env.semantic_grid
    seen_occupancy = SeenOccupancyGrid(occupancy_grid)
    seen_semantic = SeenSemanticGrid(semantic_grid)
    k = 30

    with open('seeds/large/bldg4_navgpt.txt', 'w') as f:
        for i in range(len(seeds)):
            try:
                print("SEED ", i)
                result = run_agent(seen_occupancy, seen_semantic, seeds[i]['start_pos'], seeds[i]['target_room'], timeout=250)
                f.write(str(result))
                f.write('\n')
                print("PATH LENGTH: ", result)
            except Exception as e:
                print(f"Error for seed {i}: {e}")
                f.write(str(-1))
                f.write('\n')
                continue