import openai
from math_equivalence import evaluate_answer
from transformers import AutoTokenizer

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

def final_weighted_reward(problems, completions, ground_truth, **kwargs):
    client = kwargs.get("client")
    model_path = kwargs.get("model_path")
    tokenizer = kwargs.get("tokenizer")
    tokenized_convs = [
        build_tokenized_conv(problem, completion, model_path, tokenizer)
        for problem, completion in zip(problems, completions)
    ]
    output = [
        client.chat.completions.create(
                model=model_path,
                messages=provider_prompt
            ).choices[0].message.content
            for tokenized_convs in provider_prompt_list
        ]
    return rewards


def make_final_weighted_reward(model_path):
    """Factory function to create a reward function with a specific alpha."""
    client = openai.Client(base_url="http://localhost:8000/v1", api_key="dummy-key")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    def reward_wrapper(completions, ground_truth, **kwargs):
        return final_weighted_reward(
            completions, ground_truth, client=client, model_path=model_path, tokenizer=tokenizer,**kwargs
        )

    return reward_wrapper
