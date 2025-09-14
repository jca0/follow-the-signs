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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

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

def save_confidence_heatmap(conf_grid, step, out_dir="logs/confidence"):
    os.makedirs(out_dir, exist_ok=True)
    grid = conf_grid.get_grid()
    plt.figure(figsize=(6, 6))
    plt.imshow(grid, cmap="viridis", origin="upper")
    plt.colorbar(label="confidence")
    plt.title(f"Confidence grid - step {step}")
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"step_{step:03d}.png")
    plt.savefig(out_path)
    plt.close()

def save_seen_grid(seen_occupancy, seen_semantic, agent_pos, k, step, out_dir="logs/seen"):
    os.makedirs(out_dir, exist_ok=True)
    occ = seen_occupancy.get_grid()
    sem = seen_semantic.get_grid()

    rows, cols = occ.shape
    rgb = np.ones((rows, cols, 3), dtype=float)

    # Unknown to gray
    unknown_mask = occ == -1
    rgb[unknown_mask] = np.array([0.8, 0.8, 0.8])

    # Free to white, walls to black
    wall_mask = occ == 1
    free_mask = occ == 0
    rgb[free_mask] = np.array([1.0, 1.0, 1.0])
    rgb[wall_mask] = np.array([0.0, 0.0, 0.0])

    # Signs in blue based on seen semantic
    for y in range(rows):
        for x in range(cols):
            cell = sem[y][x]
            if isinstance(cell, dict) and ("sign" in cell):
                rgb[y, x] = np.array([0.0, 0.45, 1.0])

    scale = 0.12
    fig_w = max(6.0, cols * scale)
    fig_h = max(6.0, rows * scale)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)
    ax.imshow(rgb, origin="upper", interpolation="nearest")

    # Grid lines
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5, alpha=0.4)
    ax.set_xticks([])
    ax.set_yticks([])

    # Overlay discovered room numbers from seen semantic
    for y in range(rows):
        for x in range(cols):
            cell = sem[y][x]
            if isinstance(cell, dict) and ("room_number" in cell) and not unknown_mask[y, x]:
                room = cell.get("room_number")
                if room is not None:
                    ax.text(x, y, str(room), ha="center", va="center", color="white", fontsize=6, fontweight="bold")

    # Mark agent and its current kxk FoV
    ax.scatter([agent_pos[1]], [agent_pos[0]], c="red", s=10)
    r = k // 2
    rect = Rectangle((agent_pos[1] - r - 0.5, agent_pos[0] - r - 0.5), 2 * r + 1, 2 * r + 1,
                     linewidth=1.0, edgecolor="red", facecolor="none", alpha=0.8)
    ax.add_patch(rect)

    ax.set_title(f"Seen grid - step {step}")
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"step_{step:03d}.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

class NavDecision(BaseModel):
    reasoning: str = Field(description="Why this region was chosen")
    region: Literal["left", "right", "up", "down"]
    pattern: str = Field(description="Patterns in room labels you see")

def query_llm(seen_occupancy_grid, seen_semantic_grid, agent_pos, goal):
    """
    """
    with open("prompts/ours.txt", "r") as f:
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

    # save initial seen grid
    save_seen_grid(seen_occupancy, seen_semantic, agent_pos, k, steps)

    while steps < timeout:
        steps += 1
        print(f"\nSTEP {steps}:")
        print("Agent position:", agent_pos)
        seen_occupancy.update_with_slice(agent_pos, k)
        seen_semantic.update_with_slice(agent_pos, k)

        # save seen grid after revealing slice
        save_seen_grid(seen_occupancy, seen_semantic, agent_pos, k, steps)

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
        save_confidence_heatmap(confidence_grid, steps)

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

    # seeds = []
    # with open('seeds/small/maseeh_seeds.txt', 'r') as f:
    #     seeds = [ast.literal_eval(line) for line in f if line.strip()]

    seed = {'start_pos': (64, 137), 'target_pos': (108, 197), 'target_room': '641G'}
    env = LargeSchwarz()
    occupancy_grid = env.occupancy_grid
    semantic_grid = env.semantic_grid
    seen_occupancy = SeenOccupancyGrid(occupancy_grid)
    seen_semantic = SeenSemanticGrid(semantic_grid)
    confidence_grid = ConfidenceGrid(len(occupancy_grid), len(occupancy_grid[0]))
    k = 30
    save_confidence_heatmap(confidence_grid, 0)
    result = run_agent(seen_occupancy, seen_semantic, seed['start_pos'], seed['target_room'], timeout=250)
    print("PATH LENGTH: ", result)


    # with open('seeds/large/bldg4_ours.txt', 'w') as f:
    #     # for i in range(len(seeds)):
    #     try:
    #         # print("SEED ", 2)
    #         seed = seeds[2]
    #         env = Maseeh()
    #         occupancy_grid = env.occupancy_grid
    #         semantic_grid = env.semantic_grid
    #         seen_occupancy = SeenOccupancyGrid(occupancy_grid)
    #         seen_semantic = SeenSemanticGrid(semantic_grid)
    #         confidence_grid = ConfidenceGrid(len(occupancy_grid), len(occupancy_grid[0]))
    #         k = 5
    #         result = run_agent(seen_occupancy, seen_semantic, seed['start_pos'], seed['target_room'], timeout=250)
    #         f.write(str(result))
    #         f.write('\n')
    #         print("PATH LENGTH: ", result)
    #     except Exception as e:
    #         print(f"Error for seed {2}: {e}")
    #         f.write(str(-1))
    #         f.write('\n')
    #         print("PATH LENGTH: ", -1)