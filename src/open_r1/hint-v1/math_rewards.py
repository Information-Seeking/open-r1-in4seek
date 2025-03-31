from math_equivalence import evaluate_answer
import os
from vllm import LLM
import numpy as np

def build_tokenized_conv(problem, hint, model_path, tokenizer):
    if model_path == "agentica-org/DeepScaleR-1.5B-Preview":
        conv = [
            {"role": "user", "content": problem},
        ]
        tokenized_conv = [
            tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True, return_tensors="pt")
        ]
        if "<think>\n" not in tokenized_conv:
            tokenized_conv += "<think>\n"
        tokenized_conv += f"Okay, let me {hint}"
    return tokenized_conv

def final_weighted_reward(completions, problems, ground_truth, success_rate, **kwargs):
    rewards = []
    llm = kwargs.get("llm")
    model_path = kwargs.get("model_path")
    tokenizer = kwargs.get("tokenizer")
    sampling_params = kwargs.get("sampling_params")
    tokenized_convs = [
        build_tokenized_conv(problem, completion, model_path, tokenizer)
        for problem, completion in zip(problems, completions)
    ]
    outputs = llm.generate(tokenized_convs, sampling_params)
    for i in range(len(outputs)):
        output = outputs[i]
        ref_answer = ground_truth[i]
        base_reward = success_rate[i]
        hint_rewards = []
        for j in range(len(output.outputs)):
            gen_answer = output.outputs[j].text
            final_answer = gen_answer
            if "</think>" in gen_answer:
                final_answer = gen_answer.split("</think>")[-1]
            reward = evaluate_answer(ref_answer, final_answer)
            hint_rewards.append(reward)
        rewards.append(np.mean(hint_rewards))
    return rewards


def make_final_weighted_reward(llm, sampling_params, tokenizer, model_path):
    """Factory function to create a reward function with a specific alpha."""
    def reward_wrapper(completions, problem, ground_truth, success_rate, **kwargs):
        return final_weighted_reward(
            completions, problem, ground_truth, success_rate, llm=llm, sampling_params=sampling_params, model_path=model_path, tokenizer=tokenizer,**kwargs
        )
    return reward_wrapper


