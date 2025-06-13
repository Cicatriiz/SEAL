import argparse
import json
import os
import random
from typing import List, Dict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, LengthSampler

# Assuming arclib is in the python path or PYTHONPATH is set correctly
# You might need to adjust this depending on your project structure
try:
    from arclib.arc import Task, read_tasks_from_single_file
except ImportError:
    print("Warning: arclib.arc could not be imported. Make sure it's in the PYTHONPATH.")
    # Define dummy Task and read_tasks_from_single_file for the script to be parsable
    class Task:
        def __init__(self, task_name, task_path=None):
            self.task_name = task_name
            self.task_path = task_path
            self.data = {'train': [], 'test': []} # Simplified

        def serialize(self):
            return {'train': self.data['train']}

    def read_tasks_from_single_file(challenge_file, solution_file=None):
        # Dummy implementation
        print(f"Dummy read_tasks_from_single_file called with {challenge_file}, {solution_file}")
        tasks = []
        for i in range(20): # Create 20 dummy tasks
            task = Task(f"dummy_task_{i}", challenge_file)
            # Add some dummy examples
            task.data['train'].append({'input': [[0,1],[1,0]], 'output': [[1,0],[0,1]]})
            task.data['train'].append({'input': [[2,3],[3,2]], 'output': [[3,2],[2,3]]})
            tasks.append(task)
        return tasks

def get_prompt_for_rl(task: Task, system_message: str, self_edit_prompt_template: str) -> str:
    """
    Generates a prompt for the policy model to produce a self-edit configuration.
    Adapted from few_shot.self-edit.py.
    """
    train_examples = task.serialize()['train']
    formatted_examples = ""
    for example in train_examples:
        input_grid = example['input']
        input_str = "Input:\n"
        for row in input_grid:
            input_str += " ".join(map(str, row)) + "\n"
        output_grid = example['output']
        output_str = "\nOutput:\n"
        for row in output_grid:
            output_str += " ".join(map(str, row)) + "\n"
        formatted_examples += input_str + output_str + "\n"

    user_message = formatted_examples
    user_message = user_message + "------\n\n" + self_edit_prompt_template

    # Using a simple prompt format for now, can be adapted to specific model's chat template
    prompt = f"System: {system_message}\nUser: {user_message}\nAssistant:"
    return prompt

def load_arc_tasks(challenge_file: str, solution_file: str, tokenizer,
                     system_message: str, self_edit_prompt: str,
                     num_tasks_to_load: int = 10) -> List[Dict]:
    """
    Loads ARC tasks and prepares them for the PPO trainer.
    """
    print(f"Loading ARC tasks from {challenge_file} and {solution_file}...")
    try:
        tasks = read_tasks_from_single_file(challenge_file, solution_file)
    except Exception as e:
        print(f"Error reading ARC tasks: {e}. Using dummy tasks instead.")
        # Fallback to dummy tasks if actual reading fails (e.g. if files not found during testing)
        tasks = []
        for i in range(num_tasks_to_load * 2): # Ensure enough dummy tasks
            task = Task(f"fallback_dummy_task_{i}", challenge_file)
            task.data['train'].append({'input': [[i,1],[1,i]], 'output': [[1,i],[i,1]]})
            tasks.append(task)


    loaded_tasks_data = []
    for i, task in enumerate(tasks):
        if i >= num_tasks_to_load:
            break
        prompt_text = get_prompt_for_rl(task, system_message, self_edit_prompt)
        # Tokenization will be done per batch by PPOTrainer, but we can store text
        loaded_tasks_data.append({
            'task_name': task.task_name,
            'task_obj': task, # Keep the object for later use in run_arc_task_with_config
            'prompt_text': prompt_text,
            # 'tokenized_prompt': tokenizer.encode(prompt_text, return_tensors="pt").squeeze(0) # PPOTrainer handles this
        })
    print(f"Loaded {len(loaded_tasks_data)} tasks.")
    return loaded_tasks_data

def run_arc_task_with_config(base_arc_model_name: str, task_obj: Task, config_json_string: str, tokenizer_for_arc_tasks) -> float:
    """
    Placeholder for running an ARC task with a generated configuration.
    Eventually, this will involve training a LoRA model and evaluating it.
    """
    print(f"\n--- Running ARC Task: {task_obj.task_name} ---")
    print(f"Base ARC Model: {base_arc_model_name}")
    print(f"Generated Config: {config_json_string}")

    try:
        # TODO: 1. Parse config_json_string
        # config = json.loads(config_json_string)
        pass
    except json.JSONDecodeError:
        print("Error: Invalid JSON in config string. Returning 0.0 reward.")
        return 0.0

    # TODO: 2. Set up ARC task data using task_obj and config (data augmentation)
    #           - Use arclib.augmenters, arclib.representers, arclib.messagers

    # TODO: 3. Train a LoRA adapter on base_arc_model_name using the processed data and training params from config
    #           - Use arclib.update_model.TTT or similar
    #           - This will require a separate tokenizer for the base_arc_model if different from policy model

    # TODO: 4. Evaluate the performance of the adapted LoRA model on the task's test cases
    #           - Use arclib.eval
    #           - The reward should be based on "Competition Accuracy" or similar relevant metric.

    # Simplified reward: Check for valid JSON structure and key parameters
    # This is a step up from random, but far from full LoRA training evaluation.
    # Goal: Basic validation of the generated config.
    reward = 0.0
    try:
        config = json.loads(config_json_string)
        if not isinstance(config, dict):
            print("Error: Config is not a JSON object.")
            return -1.0 # Penalize non-dict JSON

        # Check for essential keys
        if "data_generation" not in config or "training" not in config:
            print("Error: Missing 'data_generation' or 'training' keys in config.")
            return -0.5 # Penalize missing main sections

        reward += 0.2 # Base reward for parsable JSON with main sections

        if isinstance(config["data_generation"], dict) and "augmenter_types" in config["data_generation"]:
            if isinstance(config["data_generation"]["augmenter_types"], list):
                 reward += 0.2 # Reward for correct augmenter_types structure
        else:
            print("Warning: 'data_generation' setup is not as expected.")

        if isinstance(config["training"], dict) and "learning_rate" in config["training"] and "num_train_epochs" in config["training"]:
            try:
                lr = float(config["training"]["learning_rate"])
                epochs = int(config["training"]["num_train_epochs"])
                if 1e-6 < lr < 1e-2 and 1 <= epochs <= 50: # Reasonable bounds
                    reward += 0.6 # Main reward for good training params
                else:
                    reward += 0.2 # Smaller reward for params present but out of typical range
            except ValueError:
                print("Warning: Could not parse learning_rate/num_train_epochs.")
                reward += 0.1 # Params present but not parsable to float/int
        else:
            print("Warning: 'training' parameters are not as expected.")

    except json.JSONDecodeError:
        print("Error: Invalid JSON in config string.")
        return -1.0 # Penalize invalid JSON heavily
    except Exception as e:
        print(f"An unexpected error occurred during config validation: {e}")
        return -1.0

    print(f"Calculated Reward: {reward}")
    print(f"--- Finished ARC Task (config validation): {task_obj.task_name} ---\n")
    return float(reward)

def main():
    parser = argparse.ArgumentParser(description="RL Self-Edit Refinement for ARC tasks using PPO.")
    parser.add_argument("--policy_model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Model name for the policy/generator.")
    parser.add_argument("--base_arc_model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Base model for ARC task LoRA training.")
    parser.add_argument("--challenge_file", type=str, required=True, help="Path to ARC challenge file (.json).")
    parser.add_argument("--solution_file", type=str, default=None, help="Path to ARC solution file (.json), optional.")
    parser.add_argument("--ppo_learning_rate", type=float, default=1.41e-5, help="Learning rate for PPO.")
    parser.add_argument("--ppo_batch_size", type=int, default=4, help="PPO batch size.") # Reduced for potential memory issues with large models
    parser.add_argument("--ppo_mini_batch_size", type=int, default=2, help="PPO mini batch size.") # Reduced
    parser.add_argument("--ppo_gradient_accumulation_steps", type=int, default=1, help="PPO gradient accumulation steps.")
    parser.add_argument("--num_rl_iterations", type=int, default=10, help="Number of RL training iterations.") # Reduced for quick testing
    parser.add_argument("--output_dir_for_policy_model", type=str, default="./RL_generator_model", help="Directory to save the trained policy model.")
    parser.add_argument("--self_edit_prompt_path", type=str, required=True, help="Path to the self-edit prompt text file.")
    parser.add_argument("--system_message_path", type=str, required=True, help="Path to the system message text file.")
    parser.add_argument("--num_tasks_to_load", type=int, default=5, help="Number of ARC tasks to load for RL training.") # Small number for testing
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Max new tokens for generation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 1. Initialize PPOConfig
    ppo_config = PPOConfig(
        model_name=args.policy_model_name,
        learning_rate=args.ppo_learning_rate,
        batch_size=args.ppo_batch_size,
        mini_batch_size=args.ppo_mini_batch_size,
        gradient_accumulation_steps=args.ppo_gradient_accumulation_steps,
        log_with="tensorboard", # or "wandb"
        tracker_project_name="rl_self_edit_arc",
        ppo_epochs=4, # Default in TRL
        seed=args.seed,
        # accelerator_kwargs = {"num_processes": 1} # Adjust if using multi-GPU
    )

    # 2. Initialize Tokenizer and Policy Model (with Value Head)
    #    The reference model is created by PPOTrainer internally if not passed.
    #    It's often initialized with the same weights as the policy model initially.
    tokenizer = AutoTokenizer.from_pretrained(args.policy_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token # Important for generation

    policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        args.policy_model_name,
        # load_in_8bit=True, # Optional: if memory is an issue
        # device_map="auto" # Optional: for multi-GPU
    )
    # Reference model will be created by PPOTrainer if not explicitly passed.
    # To pass one explicitly:
    # ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(args.policy_model_name)
    # ppo_trainer = PPOTrainer(ppo_config, model=policy_model, ref_model=ref_model, tokenizer=tokenizer, ...)

    # A separate tokenizer for ARC tasks if base_arc_model is different (or has different chat template needs)
    # For now, assuming it's the same or compatible.
    # tokenizer_for_arc_tasks = AutoTokenizer.from_pretrained(args.base_arc_model_name)
    # if tokenizer_for_arc_tasks.pad_token is None:
    #     tokenizer_for_arc_tasks.pad_token = tokenizer_for_arc_tasks.eos_token

    # 3. Initialize PPOTrainer
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=policy_model,
        ref_model=None, # Will be created internally
        tokenizer=tokenizer,
        # dataset=None, # We provide data manually in the loop
        # data_collator=None
    )

    # Load prompt templates
    with open(args.system_message_path, 'r') as f:
        system_message = f.read().strip()
    with open(args.self_edit_prompt_path, 'r') as f:
        self_edit_prompt_template = f.read().strip()

    # 4. Load ARC Tasks
    arc_tasks_data = load_arc_tasks(
        args.challenge_file,
        args.solution_file,
        tokenizer,
        system_message,
        self_edit_prompt_template,
        num_tasks_to_load=args.num_tasks_to_load
    )

    if not arc_tasks_data:
        print("No ARC tasks loaded. Exiting.")
        return

    # Define generation kwargs for policy model
    generation_kwargs = {
        "min_length": -1, # don't want to set this
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id, # Important!
        "max_new_tokens": args.max_new_tokens,
        # "eos_token_id": tokenizer.eos_token_id # Can also be set
    }
    # LengthSampler for dynamic output length during generation if needed
    # output_length_sampler = LengthSampler(min_length_samp=50, max_length_samp=args.max_new_tokens)


    # 5. Main RL Training Loop
    print("\n--- Starting PPO Training Loop ---")
    for iteration in range(args.num_rl_iterations):
        print(f"\nRL Iteration: {iteration + 1}/{args.num_rl_iterations}")

        batch_query_texts = []
        batch_task_objs = []

        # Simple batching: select tasks for the current PPO batch
        for i in range(ppo_config.batch_size):
            task_data = random.choice(arc_tasks_data) # Sample tasks (with replacement if batch_size > num_tasks)
            batch_query_texts.append(task_data['prompt_text'])
            batch_task_objs.append(task_data['task_obj'])

        # Tokenize prompts for the policy model
        query_tensors = [tokenizer.encode(text, return_tensors="pt").squeeze(0) for text in batch_query_texts]
        # query_tensors are List[torch.Tensor]

        # Generate responses (JSON config strings) from the policy model
        # PPOTrainer.generate expects a list of tokenized queries (torch.Tensor)
        # and returns a list of tokenized responses (torch.Tensor)
        response_tensors = ppo_trainer.generate(query_tensors, return_prompt=False, **generation_kwargs)
        # response_tensors are List[torch.Tensor]

        # Decode responses to text
        batch_response_texts = [tokenizer.decode(r.squeeze(), skip_special_tokens=True) for r in response_tensors]

        # Simulate rewards
        rewards = []
        for task_obj, query_text, response_text, response_ten in zip(batch_task_objs, batch_query_texts, batch_response_texts, response_tensors):
            # print(f"Query: {query_text[:100]}...") # Print snippet of query
            # print(f"Generated Response (config): {response_text}")

            # Check for valid JSON early (optional, good for debugging)
            try:
                json.loads(response_text)
            except json.JSONDecodeError:
                print(f"Warning: Generated response for task {task_obj.task_name} is not valid JSON: {response_text}")
                # Assign a penalty or skip? For now, run_arc_task_with_config will handle it.

            reward_value = run_arc_task_with_config(
                args.base_arc_model_name,
                task_obj,
                response_text,
                tokenizer # Placeholder for actual ARC model tokenizer
            )
            rewards.append(torch.tensor(reward_value, device=ppo_trainer.accelerator.device)) # Ensure reward is on the correct device

        # Perform PPO step
        # ppo_trainer.step expects:
        # query_tensors: List[torch.LongTensor] - tokenized queries
        # response_tensors: List[torch.LongTensor] - tokenized responses (model generation part only)
        # rewards: List[torch.FloatTensor] - rewards

        # Ensure query_tensors, response_tensors, and rewards are correctly formatted lists of tensors
        # The PPOTrainer will handle moving tensors to the correct device.
        try:
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            avg_reward = sum(r.item() for r in rewards) / len(rewards)
            print(f"PPO Step Stats: {stats}")
            print(f"Average Reward for iteration {iteration + 1}: {avg_reward:.3f}")
            if ppo_config.log_with == "tensorboard":
                 ppo_trainer.accelerator.log({"ppo/reward_mean": avg_reward}, step=iteration)
        except Exception as e:
            print(f"Error during PPO step: {e}")
            print("Query tensors length:", [len(q) for q in query_tensors])
            print("Response tensors length:", [len(r) for r in response_tensors])
            print("Rewards:", rewards)
            # Potentially skip this batch or handle error
            continue


    # 6. Save the trained policy model
    print("\n--- Finished PPO Training ---")
    if not os.path.exists(args.output_dir_for_policy_model):
        os.makedirs(args.output_dir_for_policy_model)

    print(f"Saving policy model to {args.output_dir_for_policy_model}...")
    ppo_trainer.save_model(args.output_dir_for_policy_model)
    # Also save tokenizer
    tokenizer.save_pretrained(args.output_dir_for_policy_model)
    print("Model and tokenizer saved.")

if __name__ == "__main__":
    # For this script to run, you need:
    # 1. A challenge file, e.g., from ARC dataset (even a dummy one for the dummy reader to work)
    #    The dummy_challenges.json file is created below if it doesn't exist.
    # 2. The dummy prompt files created earlier:
    #    few_shot/dummy_self_edit_prompt.txt
    #    few_shot/dummy_system_message.txt
    #
    # Example command (ensure dummy files are in correct relative paths if running from root):
    # python few_shot/RL-self-edit.py \
    #   --policy_model_name "gpt2" \
    #   --base_arc_model_name "gpt2" \
    #   --challenge_file "dummy_challenges.json" \
    #   --self_edit_prompt_path "few_shot/dummy_self_edit_prompt.txt" \
    #   --system_message_path "few_shot/dummy_system_message.txt" \
    #   --num_rl_iterations 3 \
    #   --ppo_batch_size 2 \
    #   --ppo_mini_batch_size 1 \
    #   --num_tasks_to_load 2 \
    #   --output_dir_for_policy_model "./rl_test_output"

    # Create a dummy challenge file in the script's directory if it doesn't exist for testing purposes
    # This path will be relative to where the script is run.
    # If running from repo root, this will be `./dummy_challenges.json`
    # If running from `few_shot/`, this will be `few_shot/dummy_challenges.json`
    # For consistency with example command, assume running from repo root.
    dummy_challenge_file_path = "dummy_challenges.json"
    if not os.path.exists(dummy_challenge_file_path):
        print(f"Creating {dummy_challenge_file_path} for testing...")
        dummy_task_data = []
        for i in range(5): # Create 5 dummy tasks in the file
            # Ensure dummy tasks have enough structure for get_prompt_for_rl
            dummy_task_data.append({
                "name": f"dummy_task_cli_{i}", # Added name for consistency
                "train": [
                    {"input": [[0,i],[i,0]], "output": [[i,0],[0,i]]},
                    {"input": [[1,i],[i,1]], "output": [[i,1],[1,i]]}
                ],
                "test": [ # Test key might be expected by some Task parsers
                    {"input": [[0,i],[i,0]], "output": [[i,0],[0,i]]}
                ]
            })
        with open(dummy_challenge_file_path, "w") as f:
            json.dump(dummy_task_data, f)
        print(f"{dummy_challenge_file_path} created. Use --challenge_file {dummy_challenge_file_path} for testing.")

    main()
