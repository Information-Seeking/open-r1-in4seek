import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import torch
import transformers
from datasets import load_dataset
from transformers import AutoTokenizer, set_seed
from transformers.trainer_utils import get_last_checkpoint

from math_equivalence import evaluate_answer
from open_r1.configs import GRPOConfig
from open_r1.utils.callbacks import get_callbacks
from src.chat_templates import LLAMA_3_TEMPLATE
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config
from grpo_trainer import GRPOTrainer
from vllm import LLM, SamplingParams
from unittest.mock import patch
import openai
import re
import numpy as np
logger = logging.getLogger(__name__)
# import wandb
# wandb.init(mode="offline")

@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["final", "info_gain"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )

    alpha: Optional[float] = field(
        default=None,
        metadata={"help": "Alpha parameter for info_gain_reward. Used only if 'info_gain_reward' is selected."},
    )

    final_reward_weight: Optional[str] = field(default="no", metadata={"help": "Weight for final reward"})

    dataset_start: Optional[int] = field(
        default=None,
    )

    dataset_end: Optional[int] = field(
        default=None,
    )


def final_weighted_reward(completions, current_reward, ground_truth, **kwargs):
    weight = kwargs.get("weight")
    if weight == "zero":
        rewards = [0 for _ in range(len(completions))]
        return rewards
    rewards = [float(evaluate_answer(gt, c)) for c, gt in zip(completions, ground_truth)]
    if weight == "no" or sum(rewards) == 0:
        print(f"completions: {completions}")
        print(f"ground_truth: {ground_truth}")
        print(f"rewards: {rewards}")
        return rewards
    tokenizer = kwargs.get("tokenizer")
    num_tokens = [len(tokenizer.encode(completion)) for completion in completions]
    if weight == "tts":
        total_tts = 0
        for i in range(len(rewards)):
            if rewards[i] == 1:
                total_tts += num_tokens[i]
        avg_tts = 1.0 * total_tts / sum(rewards)
        print(f"avg_tts: {avg_tts}")
        rewards = [avg_tts * rewards[i] / num_tokens[i] for i in range(len(num_tokens))]
    elif weight == "ttf":
        total_ttf = sum(num_tokens)
        avg_ttf = 1.0 * total_ttf / len(rewards)
        rewards = [avg_ttf * rewards[i] / num_tokens[i] for i in range(len(num_tokens))]
        print(f"avg_ttf: {avg_ttf}")
    print(f"num_tokens: {num_tokens}")
    print(f"ground_truth: {ground_truth}")
    print(f"rewards: {rewards}")
    return rewards


def make_final_weighted_reward(tokenizer, weight):
    """Factory function to create a reward function with a specific alpha."""

    def reward_wrapper(completions, current_reward, ground_truth, **kwargs):
        return final_weighted_reward(
            completions, current_reward, ground_truth, tokenizer=tokenizer, weight=weight, **kwargs
        )

    return reward_wrapper

def get_doctor_prompt():
    system_prompt = "You are an AI doctor."
    system_prompt += " Arrive at a diagnosis of a patient's medical condition."
    system_prompt += " Ask only one question at a time, and it should not be more than 1 line."
    system_prompt += " Continue asking questions until you're 100% confident of the diagnosis."
    system_prompt += " Do not ask the same question multiple times. Ask different questions to cover more information."
    system_prompt += " The questions should cover age and sex of the patient, current symptoms, medical history of illness and medications, and relevant family history if necessary."
    system_prompt += " Keep your questions short and brief to not confuse the patient. "
    system_prompt += " After you're done asking questions, give the final diagnosis as a short response. Do not explain, only give the diagnosis name."
    system_prompt += " You must state '**Final Diagnosis:**' at the beginning of your response, otherwise you will be penalized."
    system_prompt += " You must give only 1 diagnosis otherwise you will be penalized."
    return system_prompt

def get_patient_prompt(case_desc): 
    system_prompt = "You are a patient."
    system_prompt += " You do not have any medical knowledge."
    system_prompt += " You have to describe your symptoms from the given case vignette based on the questions asked."
    system_prompt += " Do not break character and reveal that you are describing symptoms from the case vignette."
    system_prompt += " Do not generate any new symptoms or knowledge, otherwise you will be penalized."
    system_prompt += " Do not reveal more information than what the question asks."
    system_prompt += " Keep your answer short, to only 1 sentence."
    system_prompt += " Simplify terminology used in the given paragraph to layman language."
    system_prompt += f"\n**Case Vignette**: {case_desc}"
    return system_prompt

def get_mcq_after_conversation_prompt(choices, question):
    # Prompt used for conversation + MCQ
    system_prompt = f" Stop asking questions now. {question}"
    system_prompt += " Choose the correct option based on the patient's above symptoms and a list of possible options."
    system_prompt += " Only one of the choices is correct."
    system_prompt += " Give the answer as a short response. Do not explain."
    system_prompt += "\nChoices: "+ choices
    return system_prompt

def get_frq_after_conversation_prompt(question):
    # Prompt used for conversation + FRQ
    system_prompt = f" Stop asking questions now. What is the most likely diagnosis?"
    system_prompt += " Give the answer as a short response based on the patient's above symptoms."
    system_prompt += " Do not explain."
    return system_prompt


##### Evaluation prompts
def get_extract_diagnosis_name_prompt(diagnosis_para):
    system_prompt = "Identify and return the dermatology diagnosis name from the given **Query Paragraph**."
    system_prompt += " If there are more than one concurrent diagnoses present (usually indicated by 'with' or 'and'), return the names of the concurrent diagnoses."
    system_prompt += " If there are more than one possible but unsure diagnosis present (usually indicated by presence of 'or' in the paragraph), return 'Multiple'."
    system_prompt += " If there are no diagnoses present, then return 'None'."
    system_prompt += " Do not explain."
    
    system_prompt += "\n**Example 1**: 'The final diagnosis is likely tinea manuum on the right hand and tinea pedis on both feet.' Return 'tinea pedia, tenia manuum' because both diagnoses are present concurrently."
    system_prompt += "\n**Example 2**: 'Impetigo with eczema herpeticum'. Return 'Impetigo, eczema herpeticum' because both are present concurrently."
    system_prompt += "\n**Example 3**: 'Possible diagnosis of regressed nevus or halo nevus.' Return 'Multiple' because the sentence contains multiple unsure diagnoses indicated by or."
    system_prompt += "\n**Example 4**: 'Genital herpes with concurrent lymphogranuloma venereum (LGV) or other sexually transmitted infection (STI) involving lymphatic swelling.' Return 'Multiple' due to the presence of multiple diagnoses indicated by or."
    system_prompt += "\n**Example 5**: '**Final Diagnosis:** Chronic bronchitis due to long-term smoking'. Return 'Chronic bronchitis'."
    system_prompt += "\n**Example 6**: 'I need more information to arrive at a diagnosis. Consult your medical provider.' Return 'None' because there is no diagnosis."
    system_prompt += f"\n\n**Query Paragraph** : {diagnosis_para}"
    return system_prompt

def get_diagnosis_evaluation_prompt(choice1, choice2):
    system_prompt = "Identify if **Query Diagnosis 1** and **Query Diagnosis 2** are equivalent or synonymous names of the disease."
    system_prompt += " Respond with a yes/no. Do not explain."
    system_prompt += " If **Query Diagnosis 2** contains more than 1 concurrent diagnoses separated by ',', identify if any of the diagnoses is equivalent or synonymous to **Query Diagnosis 1**."
    system_prompt += " Also, if **Diagnosis 1** is a subtype of **Diagnosis 2** respond with yes, but if **Diagnosis 2** is a subtype of **Diagnosis 1** respond with no."
    system_prompt += "\nExample 1: **Diagnosis 1**: eczema ; **Diagnosis 2**: eczema, onychomycosis. Eczema is same between the two, so respond Yes. "    
    system_prompt += "\nExample 2: **Diagnosis 1**: eczema ; **Diagnosis 2**: onychomycosis. They are different, so respond No. "    
    system_prompt += "\nExample 3: **Diagnosis 1**: toe nail fungus ; **Diagnosis 2**: onychomycosis. They are synonymous, so return Yes. "
    system_prompt += "\nExample 4: **Diagnosis 1**: wart ; **Diagnosis 2**: verruca vulgaris. They are synonymous, so return Yes. "
    system_prompt += "\nExample 5: **Diagnosis 1**: lymphoma ; **Diagnosis 2**: hodgkin's lymphoma. Diagnosis 2 is subtype of Diagnosis 1, so return No. "
    system_prompt += "\nExample 6: **Diagnosis 1**: hodgkin's lymphoma ; **Diagnosis 2**: lymphoma. Diagnosis 1 is subtype of Diagnosis 2, so return Yes. " 
    system_prompt += "\nExample 7: **Diagnosis 1**: melanoma ; **Diagnosis 2**: None. They are different, so respond No."
    system_prompt += "\nExample 8: **Diagnosis 1**: melanoma ; **Diagnosis 2**: Multiple. They are different, so respond No."
    
    system_prompt += f"\n\n**Query Diagnosis 1**: {choice1}"
    system_prompt += f"\n**Query Diagnosis 2**: {choice2}"
    
    return system_prompt

def helper_eval_responses(res):
    if "yes" in res.lower():
        return 1
    else:
        return 0

# def convert_to_dict_list(input_string):
#     """
#     Convert a string with special tags into a list of dictionaries with 'role' and 'content' keys.
#     Includes the initial prompt as a "system" role and maintains original roles for messages.
    
#     Args:
#         input_string (str): The input string with special formatting
        
#     Returns:
#         list: A list of dictionaries with 'role' and 'content' keys
#     """
#     # Extract the initial prompt (between begin_of_sentence and the first Assistant tag)
#     system_pattern = r'<｜begin▁of▁sentence｜>(.*?)(?=<｜Assistant｜>)'
#     system_match = re.search(system_pattern, input_string, re.DOTALL)
    
#     # Pattern to extract messages
#     pattern = r'<｜(User|Assistant)｜>(.*?)(?=<｜(?:end▁of▁sentence｜>|User|Assistant)｜>|$)'
    
#     # Find all matches for messages
#     matches = re.findall(pattern, input_string, re.DOTALL)
    
#     # Initialize result list with system message if found
#     result = []
#     if system_match:
#         system_content = system_match.group(1).strip()
#         result.append({
#             'role': 'system',
#             'content': system_content
#         })
    
#     # Add the rest of the messages with original roles
#     for role, content in matches:
#         # Convert role to lowercase
#         role_lower = role.lower()
#         # Remove any trailing <｜end▁of▁sentence｜> if present
#         content = content.replace('<｜end▁of▁sentence｜>', '').strip()
#         # Add to result list
#         result.append({
#             'role': role_lower,
#             'content': content
#         })
    
#     return result


def convert_to_dict_list(input_string):
    """
    Convert a string with Qwen-style header tags into a list of dictionaries with 'role' and 'content' keys.
    Handles formats with <|im_start|>role\ncontent<|im_end|> pattern.
    
    Args:
        input_string (str): The input string with Qwen header formatting
        
    Returns:
        list: A list of dictionaries with 'role' and 'content' keys
    """
    result = []
    
    # Pattern to match content between <|im_start|> and <|im_end|> tags
    # This captures the role (group 1) and the content (group 2)
    pattern = r'<\|im_start\|>(.*?)\n([\s\S]*?)<\|im_end\|>'
    
    # Find all matches using the pattern
    matches = re.findall(pattern, input_string, re.DOTALL)
    
    # If no matches found, print the input string and return empty list
    if not matches:
        print("No matches found. Input string:")
        print(input_string)
        return []
    
    # Process each match to create the dictionary entries
    for role, content in matches:
        role = role.strip().lower()
        content = content.strip()
        
        result.append({
            'role': role,
            'content': content
        })
    
    return result

    
def reverse_roles(message_list):
    """
    Takes a list of message dictionaries and reverses the 'user' and 'assistant' roles.
    Does not modify 'system' role messages.
    
    Args:
        message_list (list): List of dictionaries with 'role' and 'content' keys
        
    Returns:
        list: A new list with the 'user' and 'assistant' roles reversed
    """
    result = []
    
    for message in message_list:
        new_message = message.copy()  # Create a copy to avoid modifying the original
        
        # Only reverse user and assistant roles, leave system role unchanged
        if message['role'] == 'user':
            new_message['role'] = 'assistant'
        elif message['role'] == 'assistant':
            new_message['role'] = 'user'
        
        result.append(new_message)
    
    return result

def info_gain_reward(completions, prompts, ground_truth, case_vignette, choices, question, **kwargs):  #need ground_truth, choices, question, and case_vignette to be columns of the dataset
    """Reward function that adjusts rewards based on information gain."""
    num_samples = 10
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    llm = kwargs.get("llm")
    sampling_params = kwargs.get("sampling_params")
    label_sampling_params = kwargs.get("label_sampling_params")
    # prompts and completions should be a list of lists of dictionaries
    prompts = [convert_to_dict_list(prompt) for prompt in prompts]
    
    provider_prompt_list = [[{'role':'system', 'content':get_patient_prompt(case_vignette[i])}] + reverse_roles(prompts[i][1:]) + [{'role':'user', 'content':completions[i]}] for i in range(len(completions))] 
    # output = llm.chat(provider_prompt_list, sampling_params=sampling_params)
    # answers = [output[i].outputs[0].text for i in range(len(output))] 

    output = [
        llm.chat.completions.create(
                model=model_name,
                messages=provider_prompt
            ).choices[0].message.content
            for provider_prompt in provider_prompt_list
        ]
    answers = [output[i] for i in range(len(output))]
    
    mcq_prompts = [get_mcq_after_conversation_prompt(choices[i], question[i]) for i in range(len(choices))]
    frq_prompts = [get_frq_after_conversation_prompt(question[i]) for i in range(len(question))]
    #conv_snippet_list = [prompts[i]+[{'role':'user', 'content': answers[i]}] for i in range(len(answers))]
    
    # if agent outputs a final diagnosis, remove it from the prompt
    conv_snippet_list = []
    for i in range(len(answers)):
        if "final diagnosis" not in completions[i].lower():
            conv_snippet_list.append(prompts[i] + [{'role':'assistant', 'content':completions[i]}] + [{'role':'user', 'content': answers[i]}])
        else:
            conv_snippet_list.append(prompts[i])
    
    mcq_prompt_list = [conv_snippet_list[i] + [{"role": "system", "content": mcq_prompts[i]}] for i in range(len(conv_snippet_list))]
    mcq_prompt_list = [item for element in mcq_prompt_list for item in [element] * num_samples]
    # output = llm.chat(mcq_prompt_list, sampling_params=label_sampling_params)
    # mcq_response_list = [output[i].outputs[j].text for i in range(len(output)) for j in range(len(output[0].outputs))]

    output = [
        llm.chat.completions.create(
                model=model_name,
                messages=mcq_prompt
            ).choices[0].message.content
            for mcq_prompt in mcq_prompt_list
        ]
    mcq_response_list = [output[i] for i in range(len(output))]
    mcq_success_list = [ground_truth[int(i/num_samples)].lower() in mcq_response_list[i].lower() for i in range(len(mcq_response_list))]
    mcq_success_rate_list = [np.mean(mcq_success_list[i:i+num_samples]) for i in range(0, len(mcq_response_list), num_samples)]

    frq_prompt_list = [conv_snippet_list[i] + [{"role": "system", "content": frq_prompts[i]}] for i in range(len(conv_snippet_list))]
    frq_prompt_list = [item for element in frq_prompt_list for item in [element] * num_samples]
    
    # output = llm.chat(frq_prompt_list, sampling_params=label_sampling_params)
    # frq_response_list = [output[i].outputs[j].text for i in range(len(output)) for j in range(len(output[0].outputs))]
    output = [
        llm.chat.completions.create(
                model=model_name,
                messages=frq_prompt
            ).choices[0].message.content
            for frq_prompt in frq_prompt_list
        ]
    frq_response_list = [output[i] for i in range(len(output))]

    diagnosis_prompt_list = [[{"role":"system","content":get_extract_diagnosis_name_prompt(frq_response_list[i])}] for i in range(len(frq_response_list))]
    # output = llm.chat(diagnosis_prompt_list, sampling_params=sampling_params)
    # frq_diagnosis_list = [output[i].outputs[0].text for i in range(len(output))]

    output = [
        llm.chat.completions.create(
                model=model_name,
                messages=diagnosis_prompt
            ).choices[0].message.content
            for diagnosis_prompt in diagnosis_prompt_list
        ]
    frq_diagnosis_list = [output[i] for i in range(len(output))]

    eval_prompt_list = [[{"role":"system","content":get_diagnosis_evaluation_prompt(ground_truth[int(i/num_samples)].lower(), frq_diagnosis_list[i].lower())}] for i in range(len(frq_diagnosis_list))]
    # output = llm.chat(eval_prompt_list, sampling_params=sampling_params)
    # eval_list = [output[i].outputs[0].text for i in range(len(output))]
    
    output = [
        llm.chat.completions.create(
                model=model_name,
                messages=eval_prompt
            ).choices[0].message.content
            for eval_prompt in eval_prompt_list
        ]
    eval_list = [output[i] for i in range(len(output))]

    frq_success_list = [helper_eval_responses(eval_list[i]) or ground_truth[int(i/num_samples)].lower() in frq_response_list[i].lower() for i in range(len(eval_list))]
    frq_success_rate_list = [np.mean(frq_success_list[i:i+num_samples]) for i in range(0, len(frq_response_list), num_samples)]
    rewards_list = [0.5*mcq_success_rate_list[i] + 0.5*frq_success_rate_list[i] for i in range(len(mcq_success_rate_list))]

    print(f"completions: {completions}")
    print(f"answers: {answers}")
    print(f"ground_truth: {ground_truth[0]}")
    print(f"mcq_prompt example: {mcq_prompt_list[0]}")
    print(f"provider_prompt example: {provider_prompt_list[0]}")
    print(f"mcq success rates: {mcq_success_rate_list}")
    print(f"frq success rates: {frq_success_rate_list}")
    print(f"rewards: {rewards_list}")
    
    return rewards_list


def make_info_gain_reward(llm, sampling_params, label_sampling_params):
    """Factory function to create a reward function with a specific alpha."""
    client = openai.Client(base_url="http://localhost:8000/v1", api_key="dummy-key" )

    def reward_wrapper(completions, prompts, ground_truth, case_vignette, choices, question, **kwargs):
        return info_gain_reward(completions, prompts, ground_truth, case_vignette, choices, question, llm=client, sampling_params=sampling_params, label_sampling_params=label_sampling_params, **kwargs)

    return reward_wrapper


reward_funcs_registry = {"final": final_weighted_reward, "info_gain": info_gain_reward}


def main(script_args, training_args, model_args):
    print(script_args.reward_funcs)
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        revision=model_args.model_revision,
    )
    # vllm_device = f"cuda:7"
    # print(f"llm device: {vllm_device}")
    # if vllm_device.split(":")[0] == "cuda" and int(vllm_device.split(":")[1]) >= torch.cuda.device_count():
    #     raise ValueError(
    #         f"The requested device for vllm ({vllm_device}) is not available. You are likely using vLLM "
    #         "without restricting the number of GPUs for training. Set the `--num_processes` argument to a "
    #         "value lower than the number of GPUs available on your machine—typically, reducing it by one "
    #         f"is sufficient. In your case: `--num_processes {torch.cuda.device_count() - 1}`."
    #     )

    # world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    # profiling_patch = patch(
    #     "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling", return_value=None
    # )
    # with world_size_patch, profiling_patch:
    #     llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct", enable_prefix_caching=True, gpu_memory_utilization=0.65,  max_model_len=10000, device = vllm_device)
    
    # sampling_params = SamplingParams(temperature=0.6, top_p = 0.9, n=1, max_tokens=512)
    # label_sampling_params = SamplingParams(temperature=0.6, top_p = 0.9, n=10, max_tokens=512) 

    llm = None
    sampling_params = None
    label_sampling_params = None

    # if tokenizer.chat_template is None:
    #     tokenizer.chat_template = LLAMA_3_TEMPLATE
    #     tokenizer.eos_token = "<|eot_id|>"

    # tokenizer.pad_token = "<|end_of_text|>"

    reward_funcs = []
    for func in script_args.reward_funcs:
        if func == "info_gain" :
            print(f"Register <info_gain> reward ")
            reward_func = make_info_gain_reward(llm, sampling_params, label_sampling_params)
        elif func == "final" and script_args.final_reward_weight is not None:
            print(f"Register <final> reward with weight = {script_args.final_reward_weight}")
            reward_func = make_final_weighted_reward(tokenizer, script_args.final_reward_weight)
        else:
            print("Reward specification is wrong")
            exit()
        reward_funcs.append(reward_func)

    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Data parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    # Load the dataset
    dataset = load_dataset(script_args.dataset_name)
    if script_args.dataset_start is not None and script_args.dataset_end is not None:
        dataset["train"] = dataset["train"].select(range(script_args.dataset_start, script_args.dataset_end))

    def process_dataset(example):
        messages = example["messages"]
        # messages = [{"role": "user", "content": example["problem"]}]
        # text = tokenizer.apply_chat_template(messages, tokenize=False, return_tensors="pt", add_generation_prompt=True)
        # print(text)
        return {"prompt": messages, "current_reward": 0, "case_vignette": example["case_vignette"], "question": example["question"], "choices": example["choices"], "ground_truth": example["ground_truth"]}

    processed_dataset = dataset.map(process_dataset, remove_columns=["messages"])

    # # Get reward functions
    # reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    # # Format into conversation
    # def make_conversation(example):
    #     return {
    #         "prompt": [
    #             {"role": "system", "content": SYSTEM_PROMPT},
    #             {"role": "user", "content": example["problem"]},
    #         ],
    #     }

    # dataset = dataset.map(make_conversation)
    # for split in dataset:
    #     if "messages" in dataset[split].column_names:
    #         dataset[split] = dataset[split].remove_columns("messages")

    logger.info("*** Initializing model kwargs ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
    )
    training_args.model_init_kwargs = model_kwargs

    #############################
    # Initialize the GRPO trainer
    #############################
    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=processed_dataset[script_args.dataset_train_split],
        eval_dataset=(
            processed_dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None
        ),
        peft_config=get_peft_config(model_args),
        callbacks=get_callbacks(training_args, model_args),
    )

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "dataset": list(script_args.dataset_name),
        "dataset_tags": list(script_args.dataset_name),
        "tags": ["open-r1"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(dataset[script_args.dataset_test_split])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    #############
    # push to hub
    #############
    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
