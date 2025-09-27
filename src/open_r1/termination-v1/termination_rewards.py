#from termination_prompts import extract_answer, build_conv, build_solver_conv
import os
from vllm import LLM
import numpy as np
import random
import torch
import asyncio
from typing import List, Dict, Any
import re
from termination_prompts import get_extract_diagnosis_name_prompt, get_diagnosis_evaluation_prompt, get_frq_after_conversation_prompt, helper_eval_responses

def build_tokenized_conv(problem, hint, model_path):
    hint = hint.strip()
    hint = hint[0].lower() + hint[1:]
    if model_path == "agentica-org/DeepScaleR-1.5B-Preview":
        if "no hint" in hint:
            conv = [
                {"role": "user", "content": problem},
                {"role": "assistant", "content": f"<think>\n"},
            ]
        else:
            conv = [
                {"role": "user", "content": problem},
                {"role": "assistant", "content": f"<think>\nOkay, let me {hint}"},
            ]
    return conv

async def create_completion_async(llm, model_path, tokenized_conv):
    return llm.chat.completions.create(
        model=model_path,
        messages=tokenized_conv,
        n=16,
        temperature=0.65,
        max_tokens=16384,
        extra_body={
            "add_generation_prompt": False,
            "continue_final_message": True,
        },
    )

async def get_completions_async(llm, model_path, tokenized_convs):

    tasks = [create_completion_async(llm, model_path, conv) for conv in tokenized_convs]
    # tasks = [create_completion_async(llm, model_path, tokenized_convs)]
    return await asyncio.gather(*tasks)

def final_weighted_reward(completions, prompts, correct_ans, **kwargs):
    continue_phrase = "Need More Information"
    if torch.cuda.is_available():
        device_id = torch.cuda.current_device()
    else:
        # Fallback to device ID 0 if CUDA is not available
        device_id = 0

    index = device_id % 1
    num_samples = 10
    llm = kwargs.get("llm")[index]
    model_path = kwargs.get("model_path")[index]

    termination_list = [continue_phrase not in completion[0]["content"] for completion in completions]
    frq_prompts = [get_frq_after_conversation_prompt() for i in range(len(completions))]

    frq_prompt_list = [prompts[i][:-1] + [{"role": "system", "content": frq_prompts[i]}] for i in range(len(prompts))]

    output = [
        llm.chat.completions.create(
                model=model_path,
                messages=frq_prompt,
                n=num_samples,
                max_tokens=512
            ).choices
            for frq_prompt in frq_prompt_list
        ]
    frq_response_list = [choice.message.content for prompt_choices in output for choice in prompt_choices]

    diagnosis_prompt_list = [[{"role":"system","content":get_extract_diagnosis_name_prompt(frq_response_list[i])}] for i in range(len(frq_response_list))]

    output = [
        llm.chat.completions.create(
                model=model_path,
                messages=diagnosis_prompt,
                max_tokens=256
            ).choices[0].message.content
            for diagnosis_prompt in diagnosis_prompt_list
        ]
    frq_diagnosis_list = [output[i] for i in range(len(output))]

    eval_prompt_list = [[{"role":"system","content":get_diagnosis_evaluation_prompt(correct_ans[0].lower(), frq_diagnosis_list[i].lower())}] for i in range(len(frq_diagnosis_list))]
    
    output = [
        llm.chat.completions.create(
                model=model_path,
                messages=eval_prompt,
                max_tokens=256
            ).choices[0].message.content
            for eval_prompt in eval_prompt_list
        ]
    eval_list = [output[i] for i in range(len(output))]

    frq_success_list = [helper_eval_responses(eval_list[i]) or correct_ans[0].lower() in frq_response_list[i].lower() for i in range(len(eval_list))]
    frq_success_rate_list = [np.mean(frq_success_list[i:i+num_samples]) for i in range(0, len(frq_response_list), num_samples)]
    
    HIGH_CONFIDENCE_THRESHOLD = 0.5
    LOW_CONFIDENCE_THRESHOLD = 0.2
    POSITIVE_REWARD = 1
    NEGATIVE_REWARD = -1
    NEUTRAL_REWARD = 0

    rewards_list = []
    for i in range(len(frq_success_rate_list)):
        if (frq_success_rate_list[i] >= HIGH_CONFIDENCE_THRESHOLD and termination_list[i]) or (frq_success_rate_list[i] <= LOW_CONFIDENCE_THRESHOLD and not termination_list[i]):
            rewards_list.append(POSITIVE_REWARD)
        elif (frq_success_rate_list[i] >= HIGH_CONFIDENCE_THRESHOLD and not termination_list[i]) or (frq_success_rate_list[i] <= LOW_CONFIDENCE_THRESHOLD and termination_list[i]):
            rewards_list.append(NEGATIVE_REWARD)
        else:
            rewards_list.append(NEUTRAL_REWARD)
    
    
    print(f"prompts: {prompts[0]}")
    print(f"correct_ans: {correct_ans[0]}")
    print(f"completions: {completions[0]}")
    print(f"frq_response_list: {frq_response_list[0]}")
    print(f"frq diagnosis list: {frq_diagnosis_list}")
    print(f"frq success rates: {frq_success_rate_list}")
    print(f"termination list: {termination_list}")
    print(f"rewards: {rewards_list}")
    
    return rewards_list


def make_final_weighted_reward(llm, tokenizer, model_path):
    """Factory function to create a reward function with a specific alpha."""
    def reward_wrapper(completions, prompts, correct_ans, **kwargs):
        return final_weighted_reward(
            completions, prompts, correct_ans, llm=llm, model_path=model_path, tokenizer=tokenizer,**kwargs
        )
    return reward_wrapper


# completions = completions, problems = problem, ground_truth = ground_truth, has_full_info = has_full_info, query = query,