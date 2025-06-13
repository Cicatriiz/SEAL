# SEAL - few-shot

This document contains the commands to reproduce the SEAL Few Shot Experiments on ARC.
Code is adopted from: [Ekin's Repo](https://github.com/ekinakyurek/marc/tree/main)

## SEAL RL Iteration 1

### 1. Training on 12 Problems (Iteration 1)

Train the base model on 12 problems from ARC train set:

```bash
python self-edit.py     --experiment_name=training_set_iteration_1     --challenge_file=${DATA_DIR}/arc-agi_training_challenges_filtered_1B_training_set.json     --solution_file=${DATA_DIR}/arc-agi_training_solutions_filtered_1B_training_set.json     --model_name=meta-llama/Llama-3.2-1B-Instruct     --n_tasks=12     --n_self_edits_per_task=15
```

### 2. Evaluation on Iteration 1 LoRAs

Evaluate the trained LoRAs from iteration 1:

```bash
python eval-self-edits.py     --experiment_folder=${TTI_DIR}/training_set_iteration_1     --pretrained_checkpoint=meta-llama/Llama-3.2-1B-Instruct     --lora_checkpoints_folder=${LORA_DIR}/self-edit/training_set_iteration_1     --temperature=0     --n_sample=1     --data_file=${DATA_DIR}/arc-agi_training_challenges_filtered_1B_training_set.json     --solution_file=${DATA_DIR}/arc-agi_training_solutions_filtered_1B_training_set.json     --max_lora_rank=128     --include_n=1     --new_format     --num_examples=11     --n_self_edits=15
```

### 3. RestEM on Iteration 1 (8 Epochs)

Run RestEM training for 8 epochs:

```bash
python BC-self-edit.py     --configs_and_indices=${LORA_DIR}/self-edit/training_set_iteration_1/final_configs_and_indices.json     --results=${LORA_DIR}/self-edit/training_set_iteration_1/final_results.json     --model_name=meta-llama/Llama-3.2-1B-Instruct     --lora_rank=16     --lora_alpha=16     --num_train_epochs=8     --per_device_train_batch_size=5     --gradient_accumulation_steps=1     --learning_rate=5e-5
```

### 4. Create Self-Edits on Eval Set (RL Iteration 1, 8 Epochs)

Generate self-edits on evaluation set using the 8-epoch RL model:

```bash
python self-edit.py     --experiment_name=eval_RL_iteration_1_8_epoch     --challenge_file=${DATA_DIR}/arc-agi_evaluation_challenges_filtered_1B_eval_set.json     --solution_file=${DATA_DIR}/arc-agi_evaluation_solutions_filtered_1B_eval_set.json     --model_name=${LORA_DIR}/self-edit/training_set_iteration_1/RL_trained_model_iteration_1_8_epoch     --n_tasks=10     --n_self_edits_per_task=5
```

### 5. Evaluate RL Iteration 1 (8 Epochs) on Eval Set

Evaluate the 8-epoch RL model on the evaluation set:

```bash
python eval-self-edits.py     --experiment_folder=${TTI_DIR}/eval_set_RL_iteration_1_8_epoch     --pretrained_checkpoint=${LORA_DIR}/self-edit/training_set_iteration_1/RL_trained_model_iteration_1_8_epoch     --lora_checkpoints_folder=${LORA_DIR}/self-edit/eval_RL_iteration_1_8_epoch     --temperature=0     --n_sample=1     --data_file=${DATA_DIR}/arc-agi_evaluation_challenges_filtered_1B_eval_set.json     --solution_file=${DATA_DIR}/arc-agi_evaluation_solutions_filtered_1B_eval_set.json     --max_lora_rank=128     --include_n=1     --new_format     --num_examples=9     --n_self_edits=5
```

## Baseline Evaluation

### Evaluate Baseline Model

Evaluate the baseline model performance:

```bash
python eval-self-edits-baseline.py     --experiment_folder=${TTI_DIR}/eval_base_model     --pretrained_checkpoint=meta-llama/Llama-3.2-1B-Instruct     --lora_checkpoints_folder=${LORA_DIR}/self-edit/eval_RL_iteration_1_8_epoch     --temperature=0     --n_sample=1     --data_file=${DATA_DIR}/arc-agi_evaluation_challenges_filtered_1B_eval_set.json     --solution_file=${DATA_DIR}/arc-agi_evaluation_solutions_filtered_1B_eval_set.json     --max_lora_rank=128     --include_n=1     --new_format     --num_examples=9
```

## SEAL RL Iteration 0

### Create Self-Edits on Eval Set (RL Iteration 1)
Generate self-edits using the first RL iteration model:

```bash
python self-edit.py     --experiment_name=eval_RL_iteration_1     --challenge_file=${DATA_DIR}/arc-agi_evaluation_challenges_filtered_1B_eval_set.json     --solution_file=${DATA_DIR}/arc-agi_evaluation_solutions_filtered_1B_eval_set.json     --model_name=${LORA_DIR}/self-edit/training_set_iteration_1/RL_trained_model_iteration_1     --n_tasks=10     --n_self_edits_per_task=5
```

## Notes

- All experiments use the Llama-3.2-1B-Instruct base model
- The experiments are designed to iteratively improve performance through self-editing and reinforcement learning
- Evaluation is performed on filtered ARC-AGI datasets for both training and evaluation sets
- LoRA (Low-Rank Adaptation) is used for efficient fine-tuning with various rank configurations

## SEAL RL Iteration with PPO (Advanced)

Beyond the initial Behavior Cloning (`BC-self-edit.py`), this project includes an experimental approach to further refine the self-edit generation model using Reinforcement Learning (RL) with Proximal Policy Optimization (PPO). This is implemented in `RL-self-edit.py`.

### `RL-self-edit.py`: Refining Self-Edit Generation with RL

*   **Purpose:** To improve the capability of the language model to generate effective self-edit configurations (the JSON strings specifying data augmentation and training parameters) for ARC tasks.
*   **Methodology:**
    1.  **Policy Model:** The script uses a base language model (e.g., `meta-llama/Llama-3.2-1B-Instruct`, potentially after initial fine-tuning with `BC-self-edit.py`) as the "policy." This policy model generates self-edit JSON configurations.
    2.  **Environment & Reward:** For each generated configuration:
        *   *(Current Implementation)* The configuration's validity (correct JSON structure, presence of key parameters, sensible values) is checked, and a reward is assigned based on this validation.
        *   *(Intended Full Implementation)* The configuration would be used to train a LoRA adapter for the specific ARC task. This LoRA adapter's performance (e.g., "Competition Accuracy" on the ARC task) would serve as the reward signal. This part is currently a placeholder due to environment limitations for full LoRA training within the RL loop.
    3.  **PPO Update:** The policy model is updated using PPO, leveraging the TRL library. The goal is to teach the policy model to generate configurations that lead to higher rewards (i.e., better ARC task performance or, currently, better configuration validity).
*   **Difference from `BC-self-edit.py`:**
    *   `BC-self-edit.py` uses supervised fine-tuning (Behavior Cloning) on *known successful* self-edit configurations.
    *   `RL-self-edit.py` allows the model to explore a wider range of configurations and learn from the outcomes (rewards), even if those outcomes aren't perfect successes. This enables learning more nuanced or novel configurations.

### Running `RL-self-edit.py`

The script trains the self-edit generator model. Key parameters include:

*   `--policy_model_name`: The base model to be trained with PPO (e.g., "meta-llama/Llama-3.2-1B-Instruct" or the output of `BC-self-edit.py`).
*   `--base_arc_model_name`: The model on which ARC LoRAs will be (eventually) trained and evaluated within the RL loop's environment step.
*   `--challenge_file`, `--solution_file`: Paths to ARC task data.
*   `--self_edit_prompt_path`, `--system_message_path`: Paths to prompt component files.
*   `--ppo_learning_rate`, `--ppo_batch_size`, etc.: Standard PPO hyperparameters.
*   `--num_rl_iterations`: Number of PPO training iterations.
*   `--output_dir_for_policy_model`: Where to save the RL-trained self-edit generator model.

**Example Command (using dummy data):**

```bash
python few_shot/RL-self-edit.py   --policy_model_name "gpt2"   --base_arc_model_name "gpt2"   --challenge_file "dummy_challenges.json"   --self_edit_prompt_path "few_shot/dummy_self_edit_prompt.txt"   --system_message_path "few_shot/dummy_system_message.txt"   --num_rl_iterations 10   --ppo_batch_size 4   --ppo_mini_batch_size 2   --num_tasks_to_load 5   --output_dir_for_policy_model "./rl_generator_model_output"
```

**Note:** The full reward mechanism (based on actual ARC LoRA performance) in `RL-self-edit.py` is pending full implementation and testing, which is currently blocked by environment constraints. The script currently uses config validation for rewards.
