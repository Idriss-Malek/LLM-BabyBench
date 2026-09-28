import re
import os
import yaml
from itertools import product

# Create output directory
output_dir = "configs"
os.makedirs(output_dir, exist_ok=True)

# =============================================================================
# MAIN CONFIGURATION: Just change these to control everything!
# =============================================================================

# Tasks to run - this is your main control switch!
tasks = ["predict", "plan", "decompose"]  # add "fpi" in a separate pass with formatters=["fpi_structured"]

# Formatters and Prompters (same across all tasks)
formatters = ["structured"]  # ["narrative", "structured", "json", "fpi_structured", "fpi_narrative", "fpi_json"] 
prompters = ["zero_shot"]  # ["zero_shot", "few_shot", "cot", "tot"]

# LLMs 
# Active set: non-reasoning DeepInfra-hosted instruct models only.
# Slugs verified against deepinfra.com model pages (May 2026).
llms = [
    {"name": "deepinfra", "model": "Qwen/Qwen2.5-72B-Instruct"},
    {"name": "deepinfra", "model": "moonshotai/Kimi-K2.5"},
    {"name": "deepinfra", "model": "meta-llama/Llama-4-Scout-17B-16E-Instruct"},
    {"name": "deepinfra", "model": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"},

    # --- Disabled for now ---
    # Llama 3.1 instruct (re-enable for the Llama scaling-ladder condition):
    # {"name": "deepinfra", "model": "meta-llama/Meta-Llama-3.1-8B-Instruct"},
    # {"name": "deepinfra", "model": "meta-llama/Meta-Llama-3.1-70B-Instruct"},

    # Frontier non-DeepInfra models (re-enable when running that condition):
    # {"name": "openai", "model": "gpt-5-chat-latest"},          # superseded by GPT-5.5 later
    # {"name": "anthropic", "model": "claude-sonnet-4-20250514"}, # superseded by Claude 4.7 later
    # {"name": "google", "model": "gemini-2.5-flash"},            # no llms/google.py provider implemented

    # Reasoning-leaning models (belong to the later reasoning condition, not this baseline):
    # {"name": "deepinfra", "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"},
    # {"name": "deepinfra", "model": "Qwen/Qwen3-32B"},

    # Deprecated / not hosted on DeepInfra (404 as of May 2026 — would hard-error):
    # {"name": "deepinfra", "model": "meta-llama/Meta-Llama-3.1-405B-Instruct"},
    # {"name": "deepinfra", "model": "Qwen/Qwen2.5-7B-Instruct"},
]

# Seeds 
seeds = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]

# =============================================================================
# TASK-SPECIFIC CONFIGURATIONS
# =============================================================================

task_configs = {
    "predict": {
        "levels": [
            "GoToObj", "GoToRedBallGrey", "GoToRedBall", "GoToLocal", "PutNextLocal", 
            "PickupLoc", "GoToObjMaze", "GoTo", "Pickup", "UnblockPickup", 
            "Open", "Synth", "SynthLoc", "GoToSeq", "SynthSeq", "BossLevel"
        ],
        "env_name_template": "BabyAI-{level}-v0",
        "agent_view_sizes": [],  # No agent view size variation for predict
        "llm_config": {
            "temperature": 0.7,
            "max_tokens": 8192,
            "system_prompt": ""
        },
        "eval_config": {
            "max_steps": None  # No max_steps limit for predict
        },
        "filename_includes_agent_view": False
    },
    
    "plan": {
        "levels": [
            'CustomBabyAI-GoToRedBall-Small-4Dists-v0',
            'CustomBabyAI-GoToRedBall-Small-5Dists-v0',
            'CustomBabyAI-GoToRedBall-Small-6Dists-v0',
            'CustomBabyAI-GoToRedBall-Small-7Dists-v0',
            'CustomBabyAI-GoToRedBall-Medium-20Dists-v0',
            'CustomBabyAI-GoToRedBall-Medium-40Dists-v0',
            'CustomBabyAI-GoToRedBall-Medium-50Dists-v0',
            'CustomBabyAI-GoToRedBall-Medium-60Dists-v0',
            'CustomBabyAI-GoToRedBall-Large-60Dists-v0',
            'CustomBabyAI-GoToRedBall-Large-80Dists-v0',
            'CustomBabyAI-GoToRedBall-Large-100Dists-v0',
            'CustomBabyAI-GoToRedBall-Large-120Dists-v0',
            'CustomBabyAI-GoToRedBall-Ultra-120Dists-v0',
            'CustomBabyAI-GoToRedBall-Ultra-140Dists-v0',
            'CustomBabyAI-GoToRedBall-Ultra-160Dists-v0',
            'CustomBabyAI-GoToRedBall-Ultra-180Dists-v0'
        ],
        "env_name_template": "{level}",  # Custom envs already have full names
        "agent_view_sizes": [],  # No agent view size variation for plan
        "llm_config": {
            "temperature": 0.7,
            "max_tokens": 8192,
            "system_prompt": ""
        },
        "eval_config": {
            "max_steps": None
        },
        "filename_includes_agent_view": False
    },
    
    "decompose": {
        "levels": [
            "GoToObj", "GoToRedBallGrey", "GoToRedBall", "GoToLocal", "PutNextLocal", 
            "PickupLoc", "GoToObjMaze", "GoTo", "Pickup", "UnblockPickup", 
            "Open", "Synth", "SynthLoc", "GoToSeq", "SynthSeq", "BossLevel"
        ],
        "env_name_template": "BabyAI-{level}-v0",
        "agent_view_sizes": [],
        "llm_config": {
            "temperature": 0.7,
            "max_tokens": 8192,
            "system_prompt": ""
        },
        "eval_config": {
            "max_steps": None
        },
        "filename_includes_agent_view": False
    },
    
    "fpi": {
        "levels": [
            "GoToObj", "GoToRedBallGrey", "GoToRedBall", "GoToLocal", "PutNextLocal", 
            "PickupLoc", "GoToObjMaze", "GoTo", "Pickup", "UnblockPickup", 
            "Open", "Synth", "SynthLoc", "GoToSeq", "SynthSeq", "BossLevel"
        ],
        "env_name_template": "BabyAI-{level}-v0",
        "agent_view_sizes": [3, 5, 7, 9],  # FPI has agent view size variations
        "llm_config": {
            "temperature": 0.7,
            "max_tokens": 8192,
            "system_prompt": "An agent is in a grid world consisting of one or more rooms. All rooms in the same grid world are squares of identical size and are organized in a square grid layout. Rooms are separated by walls and might contain objects such as keys, balls, and boxes of different colors. Some walls, connecting two adjacent rooms, have doors. Some doors are unlocked, whereas others need to be unlocked with keys of the same color. The agent can perform 6 actions: left (turn left), right (turn right), forward (move forward), pickup (pickup an object), drop (drop an object), and toggle (open/close a door or a box). Only the forward action changes the agent's position in the grid world. Turning left or right changes the agent's orientation only but not the position. The agent cannot move into a cell that is already occupied by an object, even if the object is one it is trying to interact with. Using a coordinate system where the (0, 0) position is the top-left corner of the grid world, necessarily corresponding to a wall, the coordinates follow the format (x, y), with x denoting the horizontal position in the grid and y denoting the vertical position in the grid."
        },
        "eval_config": {
            "max_steps": 50
        },
        "filename_includes_agent_view": True
    }
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def clean_for_filename(s):
    """Clean a string to be safe for use in filenames"""
    # Replace slashes, spaces and other problematic characters
    s = s.replace('/', '_').replace('\\', '_').replace(' ', '_')
    s = s.replace(':', '_').replace('*', '_').replace('?', '_')
    s = s.replace('"', '_').replace('<', '_').replace('>', '_')
    s = s.replace('|', '_').replace(';', '_').replace('=', '_')
    return s

def create_config(task, task_config, llm, level, prompter, formatter, seed, agent_view_size=None):
    """Create a configuration dictionary for a specific combination"""
    
    # Base environment config
    env_config = {
        "env_name": task_config["env_name_template"].format(level=level),
        "seed": seed
    }
    
    # Add agent_view_size if specified
    if agent_view_size is not None:
        env_config["agent_view_size"] = agent_view_size
    
    # Build eval config
    eval_config = {
        "task": task,
        "evaluator": task
    }
    if task_config["eval_config"]["max_steps"] is not None:
        eval_config["max_steps"] = task_config["eval_config"]["max_steps"]
    
    # Build filename
    clean_model = clean_for_filename(llm["model"])
    filename_parts = [llm['name'], clean_model, level, task, prompter, formatter, f"seed{seed}"]
    
    if task_config["filename_includes_agent_view"] and agent_view_size is not None:
        filename_parts.append(f"agent_view_size{agent_view_size}")
    
    output_path = f"results/{'_'.join(filename_parts)}.json"
    
    # Create full config
    config = {
        "env": env_config,
        "inputs": {
            "formatter": formatter,
            "processor": "omniscient_babyai_bot",
            "prompter": prompter
        },
        "eval": eval_config,
        "llm": {
            "name": llm["name"],
            "model": llm["model"],
            **task_config["llm_config"]
        },
        "output": {
            "path": output_path,
            "save_prompts": True,
            "save_raw_llm_output": True,
            "log_every": 10
        }
    }
    
    return config, '_'.join(filename_parts) + ".yaml"

# =============================================================================
# MAIN GENERATION LOOP
# =============================================================================

config_count = 0

for task in tasks:
    if task not in task_configs:
        print(f"Warning: Task '{task}' not found in task_configs. Skipping...")
        continue
    
    task_config = task_configs[task]
    selected_levels = task_config["levels"]
    agent_view_sizes = task_config["agent_view_sizes"] if task_config["agent_view_sizes"] else [None]
    
    print(f"\nGenerating configs for task: {task}")
    print(f"  - Levels: {len(selected_levels)}")
    print(f"  - Agent view sizes: {agent_view_sizes}")
    print(f"  - LLMs: {len(llms)}")
    print(f"  - Seeds: {len(seeds)}")
    print(f"  - Expected configs: {len(selected_levels) * len(agent_view_sizes) * len(llms) * len(seeds) * len(prompters) * len(formatters)}")
    
    for llm, level, prompter, formatter, seed, agent_view_size in product(
        llms, selected_levels, prompters, formatters, seeds, agent_view_sizes
    ):
        # Create configuration
        config, filename = create_config(
            task, task_config, llm, level, prompter, formatter, seed, agent_view_size
        )
        
        # Write to file
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        config_count += 1
        
        # Print progress every 100 configs
        if config_count % 100 == 0:
            print(f"Generated {config_count} configuration files so far...")

print(f"\nGenerated total of {config_count} configuration files in the '{output_dir}' directory.")
print(f"Tasks included: {', '.join(tasks)}")
