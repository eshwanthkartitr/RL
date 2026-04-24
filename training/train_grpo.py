"""
Initial RL Training Loop using TRL and OpenEnv integration.
Ref: TRL GRPO Trainer with Tool Environments
"""
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig
from releaseops_arena.tool_env import ReleaseOpsToolEnv

def create_releaseops_environment():
    """Environment factory for TRL OpenEnv."""
    return ReleaseOpsToolEnv()

def reward_func(environments, **kwargs) -> list[float]:
    """
    Extracts the reward out of the `ReleaseOpsToolEnv` state 
    after the episode finishes or truncates.
    """
    return [env.reward for env in environments]

def main():
    # 1. Load Dataset
    # Each row needs 'prompt', and kwargs to pass to env.reset()
    dataset = load_dataset("json", data_files={"train": "training/data/train.jsonl"})
    
    # Optional: ensure we map kwargs properly if needed, but TRL 
    # OpenEnv handles extra columns as **kwargs to reset() automatically.

    # 2. Config setup
    # Make sure we use an LLM that supports function calling cleanly,
    # or define the prompt formatting properly.
    model_name = "Qwen/Qwen2.5-3B-Instruct"  # or relevant model
    
    # Default tiny configuration for early smoke testing
    training_args = GRPOConfig(
        output_dir="outputs/releaseops-grpo",
        learning_rate=1e-5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        max_prompt_length=512,
        max_completion_length=1024, # covers full episode + tool outputs
        num_generations=4, # Group size for GRPO
        max_steps=50,
        logging_steps=10,
        bf16=True,
        # OpenEnv / Tool config
        env_kwargs_keys=["family", "seed", "difficulty", "archetype_mix"] # Pass these DB columns to reset(**kwargs)
    )

    # 3. Initialize Trainer
    # Pass the environment factory to TRL.
    print("Initializing GRPO Trainer...")
    trainer = GRPOTrainer(
        model=model_name,
        reward_funcs=[reward_func],
        args=training_args,
        train_dataset=dataset["train"],
        environment_factory=create_releaseops_environment
    )

    print("Starting Training (smoke test)...")
    trainer.train()

if __name__ == "__main__":
    # Ensure dataset is generated
    if not os.path.exists("training/data/train.jsonl"):
        print("Dataset not found. Generating...")
        os.system("python3 training/make_dataset.py")
        
    main()
