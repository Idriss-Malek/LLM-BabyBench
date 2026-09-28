# runner/main.py

import argparse
import yaml
from pathlib import Path
from runner.pipeline import run_pipeline, inject_config_filename


def load_config(path: str):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    
    # Inject the original config filename for tracking
    config_filename = Path(path).name
    return inject_config_filename(config, config_filename)


def main():
    parser = argparse.ArgumentParser(description="Run LLM evaluation pipeline from config.")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config YAML file."
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging."
    )
    args = parser.parse_args()

    # Validate config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {args.config}")
        return 1
    
    try:
        config = load_config(args.config)
        
        if args.verbose:
            print(f"Running config: {config_path.name}")
            print(f"Task: {config.get('eval', {}).get('task', 'unknown')}")
            print(f"Model: {config.get('llm', {}).get('model', 'unknown')}")
            print(f"Environment: {config.get('env', {}).get('env_name', 'unknown')}")
        
        result_path = run_pipeline(config)
        
        if args.verbose:
            print(f"Results saved to: {result_path}")
        
        return 0
        
    except Exception as e:
        print(f"Error running pipeline: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
    