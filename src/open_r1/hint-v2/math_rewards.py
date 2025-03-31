from math_equivalence import evaluate_answer
import os
from vllm import LLM
import numpy as np

def build_tokenized_conv(problem, hint, model_path):
    hint = hint.strip()
    hint = hint[0].lower() + hint[1:]
    if model_path == "agentica-org/DeepScaleR-1.5B-Preview":
        conv = [
            {"role": "user", "content": problem},
            {"role": "assistant", "content": f"<think>\nOkay, let me {hint}"},
        ]
    return conv

def final_weighted_reward(completions, problems, ground_truth, success_rate, **kwargs):
    hints = [
        completion[0]["content"] for completion in completions
    ]
    
    # print(f"problems: {problems}")
    # print(f"ground_truth: {ground_truth}")
    # print(f"hints: {hints}")
    
    rewards = []
    llm = kwargs.get("llm")
    model_path = kwargs.get("model_path")
    tokenized_convs = [
        build_tokenized_conv(problem, hint, model_path)
        for problem, hint in zip(problems, hints)
    ]

    output = [
        llm.chat.completions.create(
                model=model_path,
                messages=tokenized_conv,
                n=16,
                temperature=0.65,
                max_tokens=16384,
                extra_body={
                    "add_generation_prompt": False,
                    "continue_final_message": True,
                },
            ).choices
            for tokenized_conv in tokenized_convs
        ]
    
    outputs = [[choice.message.content for choice in prompt_choices] for prompt_choices in output]
    for i in range(len(outputs)):
        output = outputs[i]
    
    for i in range(len(outputs)):
        output = outputs[i]
        ref_answer = ground_truth[i]
        base_reward = success_rate[i]
        hint_rewards = []
        for j in range(len(output)):
            gen_answer = output[j]
            final_answer = gen_answer
            if "</think>" in gen_answer:
                final_answer = gen_answer.split("</think>")[-1]
            reward = evaluate_answer(ref_answer, final_answer)
            hint_rewards.append(reward)
        rewards.append(np.mean(hint_rewards))
        print(f"problem {i}: {problems[i]}\nhint {i}: {hints[i]}\nref_answer {i}: {ref_answer}\nbase_reward {i}: {base_reward}\nhint_rewards {i}: {hint_rewards}\nrewards {i}: {rewards[i]}")
    print(f"rewards: {rewards}")
    return rewards


def make_final_weighted_reward(llm, tokenizer, model_path):
    """Factory function to create a reward function with a specific alpha."""
    def reward_wrapper(completions, problem, ground_truth, success_rate, **kwargs):
        return final_weighted_reward(
            completions, problem, ground_truth, success_rate, llm=llm, model_path=model_path, tokenizer=tokenizer,**kwargs
        )
    return reward_wrapper


