# this conf will work for H100

MAX_SEQ_LENGTH = 3200

PER_DEVICE_TRAIN_BATCH_SIZE = 1  
GRADIENT_ACCUMULATION_STEPS = 4  # INCREASE TO 4 FOR SMOOTHER TRAINING  
NUM_GENERATIONS = 7  # DECREASE IF OUT OF MEMORY  
MAX_PROMPT_LENGTH = 256  
MAX_COMPLETION_LENGTH = MAX_SEQ_LENGTH

# r1 specific, suggested by deepseek
TEMPERATURE=0.6 
TOP_P=0.95 


from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)


from unsloth import is_bfloat16_supported
max_seq_length = MAX_SEQ_LENGTH # Can increase for longer reasoning traces
lora_rank = 64 # Larger rank = smarter, but slower

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/DeepSeek-R1-Distill-Qwen-1.5B",
    max_seq_length = max_seq_length,
    load_in_4bit = False, # False for LoRA 16bit
    fast_inference = True, # Enable vLLM fast inference
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.5, # Reduce if out of memory
)

model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ], # Remove QKVO if out of memory
    lora_alpha = lora_rank,
    use_gradient_checkpointing = "unsloth", # Enable long context finetuning
    random_state = 3407,
)

import re
from datasets import load_dataset, Dataset
from typing import List, Dict, Tuple, Optional

# Format templates
XML_FORMAT = """\
<think>
{thinking}
</think>
\\boxed{{{answer}}}
"""

def extract_boxed_answer(text: str) -> str:
    """
    Extract content from within \boxed{} notation
    Args:
        text: Text containing LaTeX boxed content
    Returns:
        Content within boxed notation or empty string if not found
    """
    try:
        answer = text.split("</think>")[-1]
        # Look for content between \boxed{...}
        boxed = re.search(r"\\boxed{(.*?)}", answer)
        return boxed.group(1) if boxed else ""
    except:
        return ""

def calculate_length_reward(response_len: int, min_len: int, max_len: int, is_correct: bool) -> float:
    """
    Calculate length-based reward component
    Args:
        response_len: Length of current response
        min_len: Minimum length across all responses
        max_len: Maximum length across all responses
        is_correct: Whether the response is correct
    Returns:
        Length reward value
    """
    # If all responses have same length, return 0
    if max_len == min_len:
        return 0.0
    
    # Calculate lambda parameter
    lambda_val = 1.0 - (response_len - min_len)/(max_len - min_len)
    
    # Return reward based on correctness
    return lambda_val if is_correct else min(0, lambda_val)

def correctness_reward_func(prompts: List[Dict], completions: List[Dict],
                          answer: List[str], **kwargs) -> List[float]:
    """
    Calculate reward based on answer correctness and length
    Returns:
        List of reward scores incorporating both correctness and length penalties
    """
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_boxed_answer(r) for r in responses]
    response_lengths = [len(r) for r in responses]
    
    # Calculate min and max lengths
    min_len = min(response_lengths)
    max_len = max(response_lengths)
    
    # Calculate base correctness rewards
    correctness_rewards = [2.0 if r == a else 0.0 
                         for r, a in zip(extracted_responses, answer)]
    
    # Calculate length rewards
    length_rewards = [calculate_length_reward(length, min_len, max_len, r == a)
                     for length, r, a in zip(response_lengths, extracted_responses, answer)]
    
    # Combine rewards (you can adjust the weighting parameter as needed)
    weight = 1  # Length penalty weight
    final_rewards = [c + weight * l for c, l in zip(correctness_rewards, length_rewards)]
    
    # Debug printing (as in original function)
    q = prompts[0][-1]['content']
    print('-'*20, f"Question:\n{q}")
    for i in range(len(responses)):
        print(f"\nAnswer #{i+1}:\n{answer[i]}")
        print(f"Response #{i+1}:\n{responses[i]}")
        print(f"Extracted #{i+1}:\n{extracted_responses[i]}")
        print(f"Length reward #{i+1}:\n{length_rewards[i]}")
    
    return final_rewards

def strict_format_reward_func(completions: List[Dict], **kwargs) -> List[float]:
    """
    Reward function that checks if completion follows exact format template
    """
    pattern = r"^<think>\n.*?\n</think>\n\\boxed{.*?}$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r, re.DOTALL) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def count_format_elements(text: str) -> float:
    """
    Count format elements with detailed scoring, it will think anyways.
    """
    count = 0.0
    answer = text.split("</think>")[-1]
    if answer.count("\\boxed{") == 1:
        count += 0.125
    if answer.count("}") == 1:
        count += 0.125
    return count

def xmlcount_reward_func(completions: List[Dict], **kwargs) -> List[float]:
    """
    Calculate reward based on format element counting
    """
    contents = [completion[0]["content"] for completion in completions]
    return [count_format_elements(c) for c in contents]

# dataset specific

def get_gsm8k_questions(split: str = "train") -> Dataset:
    """
    Load and preprocess GSM8K dataset for R1 format
    Args:
        split: Dataset split to load ('train' or 'test')
    Returns:
        Processed dataset with R1-compatible prompts
    """
    data = load_dataset('openai/gsm8k', 'main')[split]
    data = data.map(lambda x: {
        'prompt': [
            # R1 spec: No system prompt, include instructions in user prompt
            {'role': 'user', 'content': preprocess_math_question(x['question'])}
        ],
        'answer': extract_hash_answer(x['answer'])
    })
    return data


def preprocess_math_question(question: str) -> str:
    """
    Prepare question for R1 model by adding mathematical reasoning directive
    Args:
        question: Original question text
    Returns:
        Formatted question with R1-specific instructions
    """
    return f"{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}."


def extract_hash_answer(text: str) -> Optional[str]:
    """
    Extract answer after #### marker (for GSM8K dataset compatibility)
    Args:
        text: Answer text containing hash marker
    Returns:
        Extracted answer or None if marker not found
    """
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


dataset = get_gsm8k_questions()

from trl import GRPOConfig, GRPOTrainer

training_args = GRPOConfig(
    temperature = TEMPERATURE,
    use_vllm = True, # use vLLM for fast inference!
    learning_rate = 5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    optim = "adamw_8bit",
    logging_steps = 1,
    bf16 = is_bfloat16_supported(),
    fp16 = not is_bfloat16_supported(),
    per_device_train_batch_size = PER_DEVICE_TRAIN_BATCH_SIZE,
    gradient_accumulation_steps = GRADIENT_ACCUMULATION_STEPS, # Increase to 4 for smoother training
    num_generations = NUM_GENERATIONS, # Decrease if out of memory
    max_prompt_length = MAX_PROMPT_LENGTH,
    max_completion_length = MAX_COMPLETION_LENGTH, # MAX STEPS IS BEING SET HERE!!!
    # num_train_epochs = 1, # Set to 1 for a full training run
    max_steps = 250,
    max_grad_norm = 0.1,
    report_to = "none", # Can use Weights & Biases
    output_dir = "outputs",
    save_strategy="steps",  # Explicitly set save strategy (default is "steps")
    save_steps=1,  # Changed from 250 to 10 steps
    save_total_limit=10,  # Optional: keep only last 5 checkpoints to save 
)

trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        xmlcount_reward_func,
        strict_format_reward_func,
        correctness_reward_func,
    ],
    args = training_args,
    train_dataset = dataset,
)

print(trainer.sampling_params)
trainer.sampling_params.top_p = TOP_P

# to verify if it changed
print(trainer.sampling_params)

trainer.train()