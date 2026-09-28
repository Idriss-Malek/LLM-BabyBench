# evaluators/fpi.py

import time
from copy import deepcopy
from typing import Any, Dict, List

import gymnasium as gym

from evaluators.base import AbstractEvaluator
from evaluators.utils import str_action_seq_to_int
from processors import get_processor
from prompters.utils import Task

class FPIEvaluator(AbstractEvaluator):
    """
    Evaluator for FPI (Full mission, Partial observability, Interactive) task.
    Multi-turn conversation between LLM and environment to complete missions.
    """
    
    def evaluate(self, env_name: str, seed: int, agent_view_size: int, formatter, prompter, llm, max_steps: int = 100, **kwargs) -> Dict[str, Any]:
        """
        Conduct multi-turn interaction between LLM and partially observable environment.
        """
        
        # Initialize environment with partial observability
        env = gym.make(env_name, tile_size=32, render_mode='rgb_array', agent_view_size=agent_view_size)
        env.reset(seed=seed)
        
        # Initialize tracking
        steps_taken = 0
        total_actions_attempted = 0
        valid_actions_count = 0
        llm_terminated = False
        max_steps_reached = False
        
        # 🆕 NEW: Capture conversation data
        conversation_log = []
        
        # Conversation loop
        terminated = False
        truncated = False
        while steps_taken < max_steps and not terminated and not truncated:
            
            # Format current partial state
            state_description = formatter.format(env)
            
            # Generate prompt and get LLM response with history
            prompt = prompter.prompt(state_description, Task.FPI)
            
            print("="*50)
            print("Prompt: \n")
            print(prompt)
            print("="*50)
            print("LLM: \n")
            
            # 🆕 NEW: Time the LLM call
            llm_start_time = time.time()
            
            # LLM communication errors
            llm_response = llm.chat(prompt)  # Use chat for history
            
            # 🆕 NEW: Calculate response time
            llm_response_time = time.time() - llm_start_time
            
            print(llm_response)
            print("="*50)
            
            # Log this conversation turn
            conversation_log.append({
                "step": len(conversation_log) + 1,
                "prompt": prompt,
                "llm_response": llm_response,
                "response_time_seconds": llm_response_time,
                "environment_state": state_description
            })
            
            # Parsing errors
            from llms.utils import parser
            action_sequence_str = parser(llm_response, "fpi")
            print(action_sequence_str)
            
            # Ensuring non-empty action sequence
            if not action_sequence_str:
                raise ValueError("action_sequence_str cannot be None or empty.")
            
            # Action conversion errors
            actions = str_action_seq_to_int(action_sequence_str)
            print(actions)
            
            # Ensuring non-empty action list
            if not actions:
                raise ValueError("actions list cannot be None or empty.")
            
            # Execute action sequence
            for action in actions:
                total_actions_attempted += 1
                
                obs, reward, terminated, truncated, info = env.step(action)
                valid_actions_count += 1
                steps_taken += 1
                
                if terminated or truncated:
                    llm_terminated = terminated  # Only True if actual success
                    break
                
                if steps_taken >= max_steps:
                    max_steps_reached = True
                    break
        
        baseline_steps = self._get_baseline_performance(env_name, seed, agent_view_size)
        
        env.close()
        
        return {
            # Original metrics (unchanged)
            "metrics": {
                "success": 1.0 if llm_terminated else 0.0,
                "steps": steps_taken,
                "efficiency": baseline_steps / steps_taken if steps_taken > 0 and llm_terminated else 0.0,
                "action_validity_rate": valid_actions_count / total_actions_attempted if total_actions_attempted > 0 else 0.0,
                "llm_terminated": llm_terminated,
                "max_steps_reached": max_steps_reached,
                "baseline_steps": baseline_steps,
                "terminated": terminated,
                "truncated": truncated
            },
            # Conversation data for pipeline.py
            "prompts": [
                {
                    "step": turn["step"],
                    "prompter_type": "fpi",  # Could extract from prompter if needed
                    "formatter_type": "fpi_structured",  # Could extract from formatter if needed  
                    #"system_prompt": "",  # FPI uses chat history instead
                    "user_prompt": turn["prompt"],
                    "environment_state": turn["environment_state"]
                }
                for turn in conversation_log
            ],
            "responses": [
                {
                    "step": turn["step"],
                    "raw_response": turn["llm_response"],
                    "parsed_action": None,  # Could add parsed actions if needed
                    "parsing_successful": True,  # Could track parsing success
                    "response_time_seconds": turn["response_time_seconds"]
                }
                for turn in conversation_log
            ]
        }
    
    def _get_baseline_performance(self, env_name: str, seed: int, agent_view_size: int) -> int:
        """Get baseline performance using MiniGrid BabyAIBot under partial observability."""
        from minigrid.utils.baby_ai_bot import BabyAIBot
        
        # Environment creation errors
        baseline_env = gym.make(env_name, tile_size=32, render_mode='rgb_array', agent_view_size=agent_view_size)
        baseline_env.reset(seed=seed)
        
        # Bot creation errors
        bot = BabyAIBot(baseline_env)
        
        steps_taken = 0
        max_baseline_steps = 1000
        
        for i in range(max_baseline_steps):
            # Bot planning errors
            action = bot.replan()
            
            # Invalid actions
            obs, reward, done, truncated, info = baseline_env.step(action)
            steps_taken += 1
            
            if done:
                baseline_env.close()
                return steps_taken
            
            if truncated:
                baseline_env.close()
                raise RuntimeError(f"BabyAIBot was truncated after {steps_taken} steps without completing the task.")
        
        # Baseline bot failed to complete task within max steps
        baseline_env.close()
        raise RuntimeError(f"BabyAIBot failed to complete task within {max_baseline_steps} steps.")
    
