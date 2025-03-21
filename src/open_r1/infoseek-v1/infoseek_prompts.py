def build_conv(system_prompt, prompts, responses, role1 = "user", role2 = "assistant"):
        dialogue = []
        dialogue.append({
                "role": "system",
                "content": system_prompt
            })
        if (len(prompts) == 0):
            return dialogue
            
        if (len(responses) > len(prompts)):
            for i in range(len(prompts)):
                dialogue.append({
                    "role": role2,
                    "content": responses[i]
                })
                dialogue.append({
                    "role": role1,
                    "content": prompts[i]
                })
                        
            dialogue.append({
                "role": role2,
                "content": responses[-1]
            })
        elif (len(prompts) > len(responses)):
            for i in range(len(responses)):
                dialogue.append({
                    "role": role2,
                    "content": responses[i]
                })
                dialogue.append({
                    "role": role1,
                    "content": prompts[i]
                })  
            dialogue.append({
                "role": role1,
                "content": prompts[-1]
            })
        else:
            for i in range(len(responses)):
                dialogue.append({
                    "role": role2,
                    "content": responses[i]
                })
                dialogue.append({
                    "role": role1,
                    "content": prompts[i]
                })

        return dialogue

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