# runner/pipeline.py

import os
import json
import time
from pathlib import Path
from typing import Dict, Any

from runner.env_loader import load_envs
from runner.components import (
    get_formatter,
    get_processor,
    get_prompter,
    get_llm,
    get_evaluator,
)

def run_pipeline(config: Dict[str, Any]):
    start_time = time.time()
    
    envs = load_envs(config)
    
    # Get the base output path from the config
    base_output_path = Path(config.get("output", {}).get("path", "results/results.json"))
    
    # Create a unique filename by appending a timestamp
    timestamp = int(time.time())
    unique_output_path = base_output_path.with_name(f"{base_output_path.stem}_{timestamp}{base_output_path.suffix}")
    
    # Ensure the directory exists
    unique_output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load components
    agent_view_size = config["env"].get("agent_view_size", None)  # Default fallback
    formatter = get_formatter(config["inputs"]["formatter"], agent_view_size=agent_view_size)
    processor = get_processor(config["inputs"]["processor"])
    prompter = get_prompter(config["inputs"]["prompter"])
    evaluator = get_evaluator(config["eval"]["evaluator"])
    llm = get_llm(**config["llm"])

    # Extract config values
    env_name = config["env"]["env_name"]
    seed = config["env"]["seed"]
    task = config["eval"]["task"]
    
    # Output options
    save_prompts = config.get("output", {}).get("save_prompts", True)
    save_raw_llm_output = config.get("output", {}).get("save_raw_llm_output", True)

    results = []

    # Process the environment (should only be one env in the list)
    for idx, env in enumerate(envs):
        env_start_time = time.time()
        
        if task == 'fpi':
            # FPI: Multi-turn conversation evaluation
            # We need to modify the evaluator to return more detailed info
            eval_result = evaluator.evaluate(
                env_name=env_name,
                seed=seed,
                agent_view_size=agent_view_size,
                llm=llm,
                formatter=formatter,
                prompter=prompter,
                max_steps=config["eval"].get("max_steps", 100)
            )
            
            # Standard result structure
            result_entry = {
                "experiment_metadata": {
                    "config_file": config.get("_config_file", "unknown"),
                    "execution_duration_seconds": time.time() - env_start_time
                },
                "configuration": {
                    "task": task,
                    "env": {
                        "env_name": env_name,
                        "seed": seed,
                        "agent_view_size": agent_view_size
                    },
                    "llm": config["llm"],
                    "inputs": config["inputs"],
                    "eval": config["eval"]
                },
                "prompting_sequence": eval_result.get("prompts", []) if save_prompts else [],
                "llm_responses": eval_result.get("responses", []) if save_raw_llm_output else [],
                "evaluation_results": {
                    "evaluator_name": config["eval"]["evaluator"],
                    "metrics": eval_result.get("metrics", eval_result)  # Fallback to full eval_result
                },
                # "computational_cost": eval_result.get("cost", {})
            }
        
        else:
            # PPD tasks (predict/plan/decompose)
            env_description = formatter.format(env)
            process = processor.process(env_name, seed, task)

            if task == 'predict':
                prompt = prompter.prompt(env_description, task, process)
            else:
                prompt = prompter.prompt(env_description, task)

            # Get LLM response with metadata
            llm_start_time = time.time()
            from llms.utils import parser
            all_llm_output = llm.generate(prompt)
            llm_duration = time.time() - llm_start_time
            
            predicted_output = parser(all_llm_output, task)
            print(all_llm_output)

            # Evaluate based on task
            if task == 'predict':
                eval_result = evaluator.evaluate(env_name, seed, str_action_seq=process, predicted_output=predicted_output)
            elif task == 'plan':
                eval_result = evaluator.evaluate(env_name, seed, optimal_action_seq=process, llm_action_seq=predicted_output)
            elif task == 'decompose':
                eval_result = evaluator.evaluate(env, llm_output=predicted_output)

            # Build structured result
            result_entry = {
                "experiment_metadata": {
                    "config_file": config.get("_config_file", "unknown"),
                    "execution_duration_seconds": time.time() - env_start_time
                },
                "configuration": {
                    "task": task,
                    "env": {
                        "env_name": env_name,
                        "seed": seed,
                        "agent_view_size": agent_view_size
                    },
                    "llm": config["llm"],
                    "inputs": config["inputs"],
                    "eval": config["eval"]
                },
                "prompting_sequence": [
                    {
                        "step": 1,
                        "prompter_type": config["inputs"]["prompter"],
                        "formatter_type": config["inputs"]["formatter"],
                        "system_prompt": config["llm"].get("system_prompt", ""),
                        "user_prompt": prompt,
                        "full_prompt_tokens": len(prompt.split()) if prompt else 0  # Rough estimate
                    }
                ] if save_prompts else [],
                "llm_responses": [
                    {
                        "step": 1,
                        "raw_response": all_llm_output if save_raw_llm_output else "",
                        "parsed_output": predicted_output,
                        "parsing_successful": predicted_output is not None,
                        "response_time_seconds": llm_duration,
                        "model_metadata": {
                            "actual_model_used": getattr(llm, 'model_name', config["llm"]["model"]),
                            "provider": config["llm"]["name"]
                        }
                    }
                ],
                "evaluation_results": {
                    "evaluator_name": config["eval"]["evaluator"],
                    "metrics": eval_result
                },
                "computational_cost": {
                    "total_api_calls": 1,
                    "llm_response_time_seconds": llm_duration,
                    "total_runtime_seconds": time.time() - env_start_time
                }
            }

        results.append(result_entry)

    # Calculate total execution time
    total_duration = time.time() - start_time

    # Add total execution time to the first (and likely only) result
    if results:
        results[0]["experiment_metadata"]["total_execution_duration_seconds"] = total_duration

    # Save results to a unique file
    try:
        with open(unique_output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Evaluation complete. Results saved to: {unique_output_path}")
    except Exception as e:
        print(f"Error saving results to {unique_output_path}: {str(e)}")
        # Try saving to a fallback location
        fallback_path = Path(f"results/fallback_{os.getpid()}_{timestamp}.json")
        fallback_path.parent.mkdir(parents=True, exist_ok=True)
        with open(fallback_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to fallback location: {fallback_path}")

    return unique_output_path


def inject_config_filename(config: Dict[str, Any], config_filename: str) -> Dict[str, Any]:
    """Inject the original config filename for metadata tracking"""
    config["_config_file"] = config_filename
    return config
