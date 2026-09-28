# BabyBench
## Overview
BabyBench is a suite designed to evaluate Large Language Models (LLMs) on grounded planning and reasoning tasks. Built upon a textual adaptation of the procedurally generated BabyAI grid world, this suite assesses the capabilities of LLMs to plan and reason within the constraints of interactive environments.
BabyBench specifically evaluates three fundamental aspects of grounded intelligence: (1) predicting the consequences of actions on the environment state, (2) generating sequences of low-level actions to achieve specified subgoals, and (3) decomposing high-level missions into coherent subgoal sequences.
## Dataset Structure
This dataset consists of three main parts, each targeting a specific aspect of grounded reasoning:
1. BabyBench-Predict (BABYBENCH_Predict.csv): Evaluates a model's ability to predict the final state of the environment after executing a sequence of actions. This task tests whether the model has learned a sufficiently detailed world model to reason about transitions deterministically.
2. BabyBench-Plan (BABYBENCH_Plan.csv): Tests a model's capability to generate valid action sequences that accomplish a specified subgoal, particularly navigation-oriented goals. This measures the model's grounded action planning ability and tests its capacity to integrate the environment's transition rules.
3. BabyBench-Decompose (BABYBENCH_Decompose.csv): Assesses a model's ability to break down a high-level mission goal into intermediate subgoals. This evaluates hierarchical reasoning and goal abstraction within a grounded context.

## Tasks
### 1. Predict Task
The Predict task evaluates a model's ability to simulate spatial transformations resulting from a sequence of discrete actions. The model is provided with a textual description of the initial environment, the agent's specific initial position and orientation, and a sequence of actions (e.g., left, forward, pick-up). The model must predict the final state after executing these actions in order, without any interaction with the environment at inference time. 
Success in this task requires robust spatial reasoning, such as understanding how orientation changes affect motion, maintaining an internal representation of agent positions over multiple steps, and accounting for environmental constraints.
### 2. Plan Task
The Plan task evaluates a model's capacity to synthesize valid action sequences that accomplish a specified subgoal, restricted specifically to navigation goals (GoNextToSubgoal). Given an initial state description and a target location, the model must generate a sequence of actions that lead to successful completion of the subgoal when executed in the environment.
This task specifically tests the model's ability to plan multi-step movement sequences, potentially involving turns, movement, and obstacle avoidance.

### 3. Decompose Task
In the Decompose task, the model is presented with an initial state and a final high-level goal description, and must generate a sequence of intermediate subgoals that guides the agent from the initial state to the final objective. The subgoals are drawn from a predefined vocabulary: GoNextToSubgoal, OpenSubgoal, DropSubgoal, and PickupSubgoal.
This task tests hierarchical reasoning and goal abstraction - the model must break down a long-horizon objective into a sequence of logically proximate tasks.
## Evaluation Metrics
The benchmark provides the following metrics to evaluate LLM performance on each task:
### Predict Task
- Success Rate: Proportion of correct final state predictions.
- Manhattan Distance: L1 distance between predicted and correct agent position (for incorrect predictions).
### Plan Task
- Success Rate: Proportion of tasks successfully completed by executing LLM-generated action sequences.
- Efficiency Ratio: Ratio of the length of OmniBot's optimal actions sequence to LLM's actions sequence for successful plans.
### Decompose Task
- Comprehension Rate: Success rate when OmniBot executes LLM subgoals, allowing additions.
- Precision Rate: Success rate when OmniBot executes *only* LLM subgoals (no additions).
- Assistance Curve Integral (ACI): Area under the curve of success rate vs. number of allowed additional bot subgoals (k).

## Evaluation Process
The benchmark provides standardized evaluation procedures:

1. For Predict task: Compare the model's predicted final state with the ground truth.
2. For Plan task: Execute the model-generated action sequence in the environment to verify if it achieves the target subgoal.
3. For Decompose task: Initialize the OmniBot with the model's generated subgoal sequence and execute in the environment, recording success and assistance metrics.
