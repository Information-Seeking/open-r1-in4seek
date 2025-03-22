from math_equivalence import evaluate_answer

def final_weighted_reward(completions, ground_truth, **kwargs):
    weight = kwargs.get("weight")
    if weight == "zero":
        rewards = [0 for _ in range(len(completions))]
        return rewards
    rewards = [float(evaluate_answer(gt, c)) for c, gt in zip(completions, ground_truth)]
    if weight == "no" or sum(rewards) == 0:
        return rewards
    tokenizer = kwargs.get("tokenizer")
    num_tokens = [len(tokenizer.encode(completion)) for completion in completions]
    if weight == "tts":
        total_tts = 0
        for i in range(len(rewards)):
            if rewards[i] == 1:
                total_tts += num_tokens[i]
        avg_tts = 1.0 * total_tts / sum(rewards)
        rewards = [avg_tts * rewards[i] / num_tokens[i] for i in range(len(num_tokens))]
    elif weight == "ttf":
        total_ttf = sum(num_tokens)
        avg_ttf = 1.0 * total_ttf / len(rewards)
        rewards = [avg_ttf * rewards[i] / num_tokens[i] for i in range(len(num_tokens))]
    return rewards


def make_final_weighted_reward(tokenizer, weight):
    """Factory function to create a reward function with a specific alpha."""

    def reward_wrapper(completions, ground_truth, **kwargs):
        return final_weighted_reward(
            completions, ground_truth, tokenizer=tokenizer, weight=weight, **kwargs
        )

    return reward_wrapper