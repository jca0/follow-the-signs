import ast
import os
import sys

from openai import AzureOpenAI
from typing import Literal
from pydantic import BaseModel, Field

from helpers.env_updated import *
from helpers.utils import *
from helpers.agent_maps import *
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

PATH_STEPS = 3

class NavDecision(BaseModel):
    reasoning: str = Field(description="Why this region was chosen")
    region: Literal["left", "right", "up", "down"]
    pattern: str = Field(description="Patterns in room labels you see")

def query_llm(seen_occupancy_grid, seen_semantic_grid, agent_pos, goal):
    """
    """
    with open("prompts/ours_new.txt", "r") as f:
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

def run_agent(seen_occupancy, seen_semantic, start, goal, timeout=250):
    steps = 0
    found_goal = False
    agent_pos = start
    total_steps = 0

    while steps < timeout:
        steps += 1
        print(f"\nSTEP {steps}:")
        print("Agent position:", agent_pos)
        seen_occupancy.update_with_slice(agent_pos, k)
        seen_semantic.update_with_slice(agent_pos, k)

        goal_pos = seen_semantic.find_label(goal)
        if goal_pos:
            found_goal = True
            break

        if seen_occupancy.is_fully_explored():
            print("No valid path to goal.")
            break

        # query LLM
        llm_output = query_llm(seen_occupancy.get_grid(), seen_semantic.get_grid(), agent_pos, goal)
        reasoning, region, pattern = llm_output.reasoning, llm_output.region, llm_output.pattern
        print(f"Reasoning: {reasoning}")
        print(f"Region: {region}")
        print(f"Pattern: {pattern}")
        seen_semantic.update_key_with_pattern(agent_pos, "pattern", pattern)
        confidence_grid.update_frequency(agent_pos, region)
        confidence_grid.update_confidence(seen_semantic.get_grid(), goal)
        max_confidence_pos = confidence_grid.find_max_confidence_pos()

        # next step
        path_to_max_confidence = seen_occupancy.plan_towards(agent_pos, max_confidence_pos)
        if len(path_to_max_confidence) < PATH_STEPS:
            agent_pos = path_to_max_confidence[-1]
            total_steps += len(path_to_max_confidence)
        else:
            try:
                agent_pos = path_to_max_confidence[PATH_STEPS] # set agent's next position
                total_steps += PATH_STEPS
            except:
                print("No valid path to max confidence.")
                break


    if found_goal:
        goal_pos = seen_semantic.find_label(goal)
        path = seen_occupancy.astar(agent_pos, goal_pos)
        if path:
            print(path)
            total_steps += len(path) - 2
            print(f"Found goal in {total_steps} steps.")
            return total_steps
        else:
            return total_steps

    else:
        print("No goal found")
        return -1

if __name__ == "__main__":  
    log_file = open('logs/ours.log', 'w')
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
    confidence_grid = ConfidenceGrid(len(occupancy_grid), len(occupancy_grid[0]))
    k = 30

    with open('seeds/large/bldg4_ours.txt', 'w') as f:
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