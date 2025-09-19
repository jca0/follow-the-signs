import ast
import os
import sys
import math

from openai import AzureOpenAI
from typing import Literal
from pydantic import BaseModel, Field

from helpers.env_utils import *
from helpers.agent_maps import *
from helpers.spot_env import *
from helpers.spot_utils import *

endpoint = "https://llm-nav.openai.azure.com/"
deployment = "gpt-4o"

subscription_key = "YOUR_AZURE_OPENAI_API_KEY"
api_version = "2025-03-01-preview"

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)

CELLS_PER_STEP = 1
k = 3

class NavDecision(BaseModel):
    reasoning: str = Field(description="Why this region was chosen")
    region: Literal["left", "right", "up", "down"]
    pattern: str = Field(description="Patterns in room labels you see")

def query_llm(seen_occupancy_grid, seen_semantic_grid, agent_pos, goal):
    """
    """
    with open("../prompts/ours.txt", "r") as f:
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

def run_agent(env, command_client, grid_loc, seen_occupancy, seen_semantic, confidence_grid, start, goal, k, path_steps=1, timeout=250):
    steps = 0
    found_goal = False
    agent_pos = start
    total_steps = 0
    rows, cols = seen_occupancy.get_grid().shape

    while steps < timeout:
        steps += 1
        print(f"\nSTEP {steps}:")
        print("Agent position:", agent_pos)
        seen_occupancy.update_with_slice(agent_pos, k)
        seen_semantic.update_with_slice(agent_pos, k)
        # print(f"Seen occupancy: {seen_occupancy.mark_grid(agent_pos)}")

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
        print(f"Path to max confidence: {path_to_max_confidence}")
        if len(path_to_max_confidence) < path_steps:
            agent_pos = path_to_max_confidence[-1]
            # cr, cc = grid_loc.get_current_cell()
            # cx, cy = env.cell_center_xy(cr, cc)
            # tx, ty = env.cell_center_xy(agent_pos[0], agent_pos[1])
            # yaw = math.atan2(ty - cy, tx - cx)
            yaw = 0
            move_to_cell(command_client, grid_loc, agent_pos[0], agent_pos[1], yaw)
            total_steps += len(path_to_max_confidence)
        else:
            try:
                agent_pos = path_to_max_confidence[path_steps] # set agent's next position
                print(f"Moving to cell: {agent_pos}")
                # cr, cc = grid_loc.get_current_cell()
                # cx, cy = env.cell_center_xy(cr, cc)
                # tx, ty = env.cell_center_xy(agent_pos[0], agent_pos[1])
                # yaw = math.atan2(ty - cy, tx - cx)
                yaw = 0
                move_to_cell(command_client, grid_loc, agent_pos[0], agent_pos[1], yaw)
                total_steps += path_steps
            except Exception as e:
                print(f"Error: {e}")
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
    env = TestEnv()
    occupancy_grid = env.occupancy_grid
    semantic_grid = env.semantic_grid
    seen_occupancy = SeenOccupancyGrid(occupancy_grid)
    seen_semantic = SeenSemanticGrid(semantic_grid)
    confidence_grid = ConfidenceGrid(len(occupancy_grid), len(occupancy_grid[0]))
    result = run_agent(seen_occupancy, seen_semantic, confidence_grid, (0, 0), '1', 1)
    print(result)