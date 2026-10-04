"""Generate the BabyBench run configs.

Reproduces, file for file, the configs used in the reported experiments:

  python create_yamls.py
      -> configs/            11,520  claude-sonnet-4-6 / claude-opus-4-6 / gpt-5.4
                                     x reasoning effort off/low/medium/high
                                     x 3 tasks x 16 levels x 20 seeds, T=0.0
  python create_yamls.py --pass3
      -> configs_pass3_off/   8,640  effort off, T=0.7, 3 samples per config (_t0.7_s0.._s2);
                                     the set behind the pass@1 / pass@3 tables
  python create_yamls.py --models claude-sonnet-4-6 --output_dir configs_sonnet                      (3,840)
  python create_yamls.py --models gpt-5.4 --output_dir configs_gpt5                                  (3,840)
  python create_yamls.py --models claude-sonnet-4-6 --efforts off --output_dir configs_sonnet_off2   (960)
  python create_yamls.py --models claude-opus-4-6 --efforts off --output_dir configs_opus_off        (960)
  python create_yamls.py --models gpt-5.4 --efforts off --output_dir configs_gpt5_off                (960)

Seeds, levels, per-level max_tokens and filenames are fixed below; edit them
there to define a new experiment.
"""
import argparse
import re
import os
import yaml
from itertools import product

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--output_dir", default=None,
                    help="Output directory (default: configs, or configs_pass3_off with --pass3)")
parser.add_argument("--models", nargs="+", default=None,
                    help="Only generate configs for these model ids, e.g. claude-sonnet-4-6 gpt-5.4")
parser.add_argument("--efforts", nargs="+", default=None,
                    help="Only these reasoning efforts for frontier models (off low medium high)")
parser.add_argument("--pass3", action="store_true",
                    help="pass@3 sampling set: effort off, T=0.7, 3 samples per config")
parser.add_argument("--temperature", type=float, default=None,
                    help="Override sampling temperature (default 0.0; 0.7 with --pass3)")
parser.add_argument("--samples", type=int, default=None,
                    help="Samples per config; >1 appends _t{T}_s{i} to names (default 1; 3 with --pass3)")
args = parser.parse_args()

PASS3_TEMPERATURE = 0.7
PASS3_SAMPLES = 3
n_samples = args.samples if args.samples is not None else (PASS3_SAMPLES if args.pass3 else 1)
temperature_override = args.temperature if args.temperature is not None else (PASS3_TEMPERATURE if args.pass3 else None)

# Create output directory
output_dir = args.output_dir or ("configs_pass3_off" if args.pass3 else "configs")
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
# Non-frontier baseline set (disabled — user only wants frontier reasoning models).
llms = [
    # {"name": "deepinfra", "model": "Qwen/Qwen2.5-72B-Instruct"},
    # {"name": "deepinfra", "model": "moonshotai/Kimi-K2.5"},
    # {"name": "deepinfra", "model": "meta-llama/Llama-4-Scout-17B-16E-Instruct"},
    # {"name": "deepinfra", "model": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"},

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

# Frontier reasoning models — swept across reasoning_efforts below.
# Kept separate from `llms` so non-reasoning instruct models don't get
# a spurious reasoning_effort key (which they wouldn't accept anyway).
frontier_llms = [
    {"name": "anthropic", "model": "claude-sonnet-4-6"},
    {"name": "anthropic", "model": "claude-opus-4-6"},
    {"name": "openai",    "model": "gpt-5.4"},
]

reasoning_efforts = ["off", "low", "medium", "high"]

# CLI selections (see module docstring)
if args.efforts:
    reasoning_efforts = list(args.efforts)
elif args.pass3:
    reasoning_efforts = ["off"]
if args.models:
    known = {m["model"] for m in llms + frontier_llms}
    unknown = set(args.models) - known
    if unknown:
        raise SystemExit(f"Unknown model id(s) {sorted(unknown)}; known: {sorted(known)}")
    llms = [m for m in llms if m["model"] in args.models]
    frontier_llms = [m for m in frontier_llms if m["model"] in args.models]

# Per-level max_tokens cap (overrides task_config["llm_config"]["max_tokens"]
# when the level appears below). Plan's CustomBabyAI-* levels are unaffected.
LEVEL_MAX_TOKENS = {
    "GoToObj": 2048, "GoToRedBallGrey": 2048, "GoToRedBall": 2048,
    "GoToLocal": 2048, "PutNextLocal": 2048,
    "PickupLoc": 4096, "GoToObjMaze": 4096, "GoTo": 4096,
    "Pickup": 4096, "UnblockPickup": 4096,
    "Open": 8192, "Synth": 8192, "SynthLoc": 8192,
    "GoToSeq": 8192, "SynthSeq": 8192, "BossLevel": 8192,
}

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
            "temperature": 0.0,
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
            "temperature": 0.0,
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
            "temperature": 0.0,
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
            "temperature": 0.0,
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

def create_config(task, task_config, llm, level, prompter, formatter, seed, agent_view_size=None, reasoning_effort=None,
                  temperature=None, sample=None):
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

    if reasoning_effort is not None:
        filename_parts.append(f"effort{reasoning_effort}")

    # pass@k sampling: one file per sample, e.g. ..._effortoff_t0.7_s0
    if sample is not None:
        t = temperature if temperature is not None else task_config["llm_config"]["temperature"]
        filename_parts.append(f"t{t}")
        filename_parts.append(f"s{sample}")

    output_path = f"results/{'_'.join(filename_parts)}.json"

    llm_block = {
        "name": llm["name"],
        "model": llm["model"],
        **task_config["llm_config"],
    }
    if level in LEVEL_MAX_TOKENS:
        llm_block["max_tokens"] = LEVEL_MAX_TOKENS[level]
    if reasoning_effort is not None:
        llm_block["reasoning_effort"] = reasoning_effort
    if temperature is not None:
        llm_block["temperature"] = temperature

    # Create full config
    config = {
        "env": env_config,
        "inputs": {
            "formatter": formatter,
            "processor": "omniscient_babyai_bot",
            "prompter": prompter
        },
        "eval": eval_config,
        "llm": llm_block,
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
    
    n_deepinfra = len(selected_levels) * len(agent_view_sizes) * len(llms) * len(seeds) * len(prompters) * len(formatters)
    n_deepinfra *= n_samples
    n_frontier = len(selected_levels) * len(agent_view_sizes) * len(frontier_llms) * len(reasoning_efforts) * len(seeds) * len(prompters) * len(formatters) * n_samples

    print(f"\nGenerating configs for task: {task}")
    print(f"  - Levels: {len(selected_levels)}")
    print(f"  - Agent view sizes: {agent_view_sizes}")
    print(f"  - DeepInfra LLMs: {len(llms)} (no effort axis)")
    print(f"  - Frontier LLMs: {len(frontier_llms)} x {len(reasoning_efforts)} effort levels")
    print(f"  - Seeds: {len(seeds)}")
    print(f"  - Expected configs: {n_deepinfra} (deepinfra) + {n_frontier} (frontier) = {n_deepinfra + n_frontier}")

    # DeepInfra (non-reasoning) loop — no effort axis
    for llm, level, prompter, formatter, seed, agent_view_size in product(
        llms, selected_levels, prompters, formatters, seeds, agent_view_sizes
    ):
        for sample in (range(n_samples) if n_samples > 1 else [None]):
            config, filename = create_config(
                task, task_config, llm, level, prompter, formatter, seed, agent_view_size,
                temperature=temperature_override, sample=sample,
            )
            filepath = os.path.join(output_dir, filename)
            with open(filepath, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            config_count += 1
            if config_count % 100 == 0:
                print(f"Generated {config_count} configuration files so far...")

    # Frontier reasoning models — sweep across reasoning_efforts
    for llm, level, prompter, formatter, seed, agent_view_size, effort in product(
        frontier_llms, selected_levels, prompters, formatters, seeds, agent_view_sizes, reasoning_efforts
    ):
        for sample in (range(n_samples) if n_samples > 1 else [None]):
            config, filename = create_config(
                task, task_config, llm, level, prompter, formatter, seed, agent_view_size,
                reasoning_effort=effort, temperature=temperature_override, sample=sample,
            )
            filepath = os.path.join(output_dir, filename)
            with open(filepath, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            config_count += 1
            if config_count % 100 == 0:
                print(f"Generated {config_count} configuration files so far...")

print(f"\nGenerated total of {config_count} configuration files in the '{output_dir}' directory.")
print(f"Tasks included: {', '.join(tasks)}")
