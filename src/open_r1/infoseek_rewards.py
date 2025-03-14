import openai
import numpy as np
import re

from infoseek_prompts import get_doctor_prompt, get_patient_prompt, get_mcq_after_conversation_prompt, get_frq_after_conversation_prompt, get_extract_diagnosis_name_prompt, get_diagnosis_evaluation_prompt

def helper_eval_responses(res):
    if "yes" in res.lower():
        return 1
    else:
        return 0
  
    
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
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    llm = kwargs.get("llm")
    sampling_params = kwargs.get("sampling_params")
    label_sampling_params = kwargs.get("label_sampling_params")
    
    # prompts and completions should be a list of lists of dictionaries
    print(f"length of prompt: {len(prompts)}")
    provider_prompt_list = [[{'role':'system', 'content':get_patient_prompt(case_vignette[i])}] + reverse_roles(prompts[i][1:]) + [{'role':'user', 'content':completions[i]}] for i in range(len(completions))] 
    print(f"info provider prompt: {provider_prompt_list[0]}")
    print("====================================================")
    # output = llm.chat(provider_prompt_list, sampling_params=sampling_params)
    # answers = [output[i].outputs[0].text for i in range(len(output))] 
    output = llm.chat.completions.create(
        model=model_name,
        messages=provider_prompt_list[0]
    )
    print(f"raw output: {output}")
    output = [
        llm.chat.completions.create(
                model=model_name,
                messages=provider_prompt
            ).choices[0].message.content
            for provider_prompt in provider_prompt_list
        ]
    print(f"output: {output}")
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
    print(f"frq_prompt example: {frq_prompt_list[0]}")
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