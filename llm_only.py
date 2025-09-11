import ast
import os
import sys

from openai import AzureOpenAI
from typing import Literal
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

class NavDecision(BaseModel):
    reasoning: str = Field(description="Why this action was chosen")
    action: Literal["left", "right", "up", "down"]

def query_llm(seen_occupancy_grid, seen_semantic_grid, agent_pos, goal):
    """
    """
    with open("prompts/llm_only.txt", "r") as f:
        prompt_template = f.read()

    safe = prompt_template.replace("{", "{{").replace("}", "}}")
    for key in ["goal", "agent_pos", "seen_semantic_grid"]:
        safe = safe.replace("{{" + key + "}}", "{" + key + "}")
    
    prompt = safe.format(goal=goal, agent_pos=agent_pos, seen_occupancy_grid=seen_occupancy_grid, seen_semantic_grid=seen_semantic_grid)

    response = client.responses.parse(
        model=deployment,
        input=[{"role": "user", "content": prompt}],
        text_format=NavDecision
    )

    return response.output_parsed

def run_agent(seen_occupancy, seen_semantic, start, goal, timeout=100):
    steps = 0
    found_goal = False
    agent_pos = start

    while steps < timeout:
        steps += 1
        print(f"\nSTEP {steps}:")
        print("Agent position:", agent_pos)
        seen_occupancy.update_with_slice(agent_pos, k)
        seen_semantic.update_with_slice(agent_pos, k)

        goal_pos = seen_semantic.find_label(goal)
        if agent_pos == goal_pos:
            found_goal = True
            break

        if seen_occupancy.is_fully_explored():
            print("No valid path to goal.")
            break

        # query LLM
        llm_output = query_llm(seen_occupancy.get_grid(), seen_semantic.get_grid(), agent_pos, goal)
        reasoning, action = llm_output.reasoning, llm_output.action
        print(f"Reasoning: {reasoning}")
        print(f"Action: {action}")

        # next step
        try:
            if action == "left":
                nr, nc = agent_pos[0], agent_pos[1] - 1
                if (nr, nc) == goal_pos or seen_occupancy.get_grid()[nr][nc] == 0:
                    agent_pos = (nr, nc)
            elif action == "right":
                nr, nc = agent_pos[0], agent_pos[1] + 1
                if (nr, nc) == goal_pos or seen_occupancy.get_grid()[nr][nc] == 0:
                    agent_pos = (nr, nc)
            elif action == "up":
                nr, nc = agent_pos[0] - 1, agent_pos[1]
                if (nr, nc) == goal_pos or seen_occupancy.get_grid()[nr][nc] == 0:
                    agent_pos = (nr, nc)
            elif action == "down":
                nr, nc = agent_pos[0] + 1, agent_pos[1]
                if (nr, nc) == goal_pos or seen_occupancy.get_grid()[nr][nc] == 0:
                    agent_pos = (nr, nc)
        except:
            print("Invalid action.")
            break

    if found_goal:
        print(f"Found goal in {steps} steps.")
        return steps
    else:
        print("No goal found")
        return -1 

if __name__ == "__main__":  
    log_file = open('logs/llm_only.log', 'w')
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

    with open('seeds/large/bldg4_llm.txt', 'w') as f:
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
                print("PATH LENGTH: ", result)
                continue