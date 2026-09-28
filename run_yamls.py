import os
import subprocess
import glob
import argparse
import yaml
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import json
from datetime import datetime

def get_output_path_from_config(config_path):
    """Extract the base output path from a YAML config file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Get the output path from the config
        if 'output' in config and 'path' in config['output']:
            return Path(config['output']['path'])
        else:
            # Default path from pipeline.py
            return Path("results/results.json")
    except Exception as e:
        print(f"Error parsing config file {config_path}: {str(e)}")
        return None

def config_has_results(config_path):
    """Check if results already exist for this configuration."""
    output_path = get_output_path_from_config(config_path)
    
    if not output_path:
        # If we can't determine the output path, assume it hasn't been run
        return False
    
    # Check if any files match the pattern {stem}_*.{suffix}
    result_pattern = f"{output_path.stem}_*{output_path.suffix}"
    matching_files = list(output_path.parent.glob(result_pattern))
    
    # If we find any matching result files, this config has been run
    return len(matching_files) > 0

def get_config_info(config_path):
    """Extract key info from config for better logging."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Extract key identifiers
        llm_name = config.get('llm', {}).get('name', 'unknown')
        llm_model = config.get('llm', {}).get('model', 'unknown')
        task = config.get('eval', {}).get('task', 'unknown')
        env_name = config.get('env', {}).get('env_name', 'unknown')
        
        return f"{llm_name}/{llm_model.split('/')[-1]}/{task}/{env_name.split('-')[-2] if '-' in env_name else env_name}"
    except:
        return config_path

def run_config(config_path, verbose=False):
    """Run a single configuration file using runner/main.py"""
    config_info = get_config_info(config_path)
    
    # First check if this config has already been run
    if config_has_results(config_path):
        if verbose:
            print(f"SKIP: {config_info}")
        return True, config_path, "Skipped (results exist)", 0
    
    try:
        start_time = time.time()
        if verbose:
            print(f"START: {config_info}")
        
        # Execute the runner/main.py script with the config file
        result = subprocess.run(
            ["python3", "-m", "runner.main", "--config", config_path],
            capture_output=True,
            text=True,
            check=False  # Don't raise exception on non-zero exit
        )
        
        # Calculate execution time
        duration = time.time() - start_time
        
        # Check if the run was successful
        if result.returncode == 0:
            if verbose:
                print(f"DONE: {config_info} ({duration:.1f}s)")
            return True, config_path, None, duration
        else:
            if verbose:
                print(f"FAIL: {config_info} ({duration:.1f}s)")
            error_message = result.stderr.strip()
            if verbose and error_message:
                print(f"Error: {error_message[:150]}...")
            return False, config_path, error_message, duration
            
    except Exception as e:
        duration = time.time() - start_time if 'start_time' in locals() else 0
        if verbose:
            print(f"ERROR: {config_info}: {str(e)}")
        return False, config_path, str(e), duration

def save_run_log(successful, failed, skipped, start_time, total_duration):
    """Save detailed run log as JSON."""
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "start_time": start_time,
        "total_duration_seconds": total_duration,
        "summary": {
            "total": len(successful) + len(failed) + len(skipped),
            "successful": len(successful),
            "failed": len(failed),
            "skipped": len(skipped)
        },
        "successful_configs": [{"config": path, "duration": dur} for path, dur in successful],
        "failed_configs": [{"config": path, "error": error, "duration": dur} for path, error, dur in failed],
        "skipped_configs": [{"config": path, "reason": reason} for path, reason in skipped]
    }
    
    log_filename = f"batch_run_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_filename, 'w') as f:
        json.dump(log_data, f, indent=2)
    
    return log_filename

def main():
    parser = argparse.ArgumentParser(description="Run all configuration files in the configs directory.")
    parser.add_argument(
        "--config_dir", type=str, default="configs", 
        help="Directory containing configuration YAML files (defaults to 'configs')."
    )
    parser.add_argument(
        "--max_workers", type=int, default=10,
        help="Maximum number of parallel worker threads."
    )
    parser.add_argument(
        "--force_rerun", action="store_true",
        help="Force rerun of all configs even if they have existing results."
    )
    parser.add_argument(
        "--filter", type=str, default=None,
        help="Only run configs whose filenames contain this string."
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Just show which configs would be run without actually running them."
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show detailed progress for each config."
    )
    args = parser.parse_args()
    
    # Find all YAML files in the configs directory
    config_paths = glob.glob(os.path.join(args.config_dir, "*.yaml"))
    
    # Apply filter if specified
    if args.filter:
        config_paths = [p for p in config_paths if args.filter in p]
        print(f"Applied filter '{args.filter}': {len(config_paths)} configs match")
    
    if not config_paths:
        print(f"No configuration files found in {args.config_dir}")
        return
    
    print(f"Found {len(config_paths)} configuration files to process")
    
    # If we're not forcing reruns, filter out configs that already have results
    skipped_configs = []
    if not args.force_rerun:
        pending_configs = []

        for config_path in config_paths:
            if config_has_results(config_path):
                skipped_configs.append(config_path)
            else:
                pending_configs.append(config_path)
        
        print(f"Skipping {len(skipped_configs)} configs with existing results")
        print(f"Processing {len(pending_configs)} configs without results")
        
        config_paths = pending_configs
    
    if not config_paths:
        print("No configurations to run (all have existing results)")
        return
    
    if args.dry_run:
        print("\n🔍 DRY RUN - Configs that would be executed:")
        for i, config_path in enumerate(config_paths, 1):
            config_info = get_config_info(config_path)
            print(f"  {i:3d}. {config_info}")
        print(f"\nTotal: {len(config_paths)} configs would be run")
        return
    
    # Statistics
    successful = []
    failed = []
    skipped = [(p, "Skipped (results exist)") for p in skipped_configs]
    
    start_time = time.time()
    
    print(f"\nStarting batch execution with {args.max_workers} workers...")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run configurations in parallel with progress tracking
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # Submit all tasks
        future_to_config = {
            executor.submit(run_config, config_path, args.verbose): config_path 
            for config_path in config_paths
        }
        
        completed = 0
        for future in as_completed(future_to_config):
            completed += 1
            config_path = future_to_config[future]
            
            try:
                success, config_path, error, duration = future.result()
                
                if error == "Skipped (results exist)":
                    skipped.append((config_path, error))
                elif success:
                    successful.append((config_path, duration))
                else:
                    failed.append((config_path, error, duration))
                
                # Progress update (only if not verbose to avoid spam)
                if not args.verbose:
                    if completed % 10 == 0 or completed == len(config_paths):
                        elapsed = time.time() - start_time
                        rate = completed / elapsed if elapsed > 0 else 0
                        eta = (len(config_paths) - completed) / rate if rate > 0 else 0
                        print(f"Progress: {completed}/{len(config_paths)} ({completed/len(config_paths)*100:.1f}%) | "
                              f"Rate: {rate:.1f}/s | ETA: {eta/60:.1f}m")
                        
            except Exception as e:
                failed.append((config_path, str(e), 0))
    
    total_duration = time.time() - start_time
    
    # Print summary
    print("\n" + "="*80)
    print(f"EXECUTION SUMMARY")
    print(f"Total time: {total_duration/60:.1f} minutes")
    print(f"Results:")
    print(f" • Total configurations: {len(config_paths)}")
    print(f" • Successfully ran: {len(successful)}")
    print(f" • Failed: {len(failed)}")
    print(f" • Skipped: {len(skipped)}")
    
    if successful:
        avg_duration = sum(dur for _, dur in successful) / len(successful)
        print(f"  • ⚡ Average runtime: {avg_duration:.1f}s")
    
    # Save detailed log
    log_filename = save_run_log(successful, failed, skipped, start_time, total_duration)
    print(f"Detailed log saved to: {log_filename}")
    
    # If there were failures, write them to a file
    if failed:
        failed_filename = f"failed_configs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(failed_filename, "w") as f:
            f.write(f"Failed configurations (Total: {len(failed)}):\n")
            f.write("="*50 + "\n")
            for config_path, error, duration in failed:
                f.write(f"\nConfig: {config_path}\n")
                f.write(f"Duration: {duration:.1f}s\n")
                f.write(f"Error: {error}\n")
                f.write("-" * 50 + "\n")
        print(f"Failed configurations written to: {failed_filename}")
    
    print("="*80)

if __name__ == "__main__":
    main()
    