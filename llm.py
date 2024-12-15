import ollama
from huggingface_hub import InferenceClient

global gpt_call_count
gpt_call_count = 0

def gpt_model_call(prompt, model='ollama_llama3'):

    model_config = {
        'ollama_llama3': ("llama3:latest", 15000), # generaliza melhor em todos os agentes mas acaba por n ser o melhor pa nenhum
        'ollama_qwencoder': ("qwen2.5-coder:latest", 15000), #percebe melhor matriz e json format -> Obesrvation Agent
        'ollama_qwen': ("qwen:7b", 15000), #  é o mais inteligente unico problema era a verbose agora já n deve ter corrijido com função extraction
        'ollama_llama3_zero': ("llama3_zero:latest", 15000) # llama3 with zero temperature
    }                         
    

    model_name, max_tokens = model_config.get(model)

      # this global variable is used to keep track of the number of calls made to the GPT-4 model
    global gpt_call_count
    gpt_call_count += 1   # increment the call count by 1

    

    if model in ['ollama_llama3']:
        # Combine system message and user prompt for ollama
        print("Model Answearing:", model_name)
        combined_prompt = "System: You are a helpful assistant designed to output JSON. \nUser: " + prompt
        model_output = ollama.chat(model=model_name, messages=[
            {"role": "user", "content": combined_prompt}
        ])
        model_output = model_output["message"]["content"].strip()

    elif model in ['ollama_qwencoder']:
        # Combine system message and user prompt for ollama
        print("Model Answearing: ", model_name)
        combined_prompt = "System: You are a helpful assistant designed to output JSON. \nUser: " + prompt
        model_output = ollama.chat(model=model_name, messages=[
            {"role": "user", "content": combined_prompt}
        ])
        model_output = model_output["message"]["content"].strip()

    elif model in ['ollama_qwen']:


        print("Model Answearing:",model_name)
        # Combine system message and user prompt for ollama
        combined_prompt = "System: You are a helpful assistant designed to output JSON. \nUser: " + prompt
        model_output = ollama.chat(model=model_name, messages=[
            {"role": "user", "content": combined_prompt}
        ])
        model_output = model_output["message"]["content"].strip()

    elif model in ['ollama_llama3_zero']:


        print("Model Answearing:",model_name)
        # Combine system message and user prompt for ollama
        combined_prompt = "System: You are a helpful assistant designed to output JSON. \nUser: " + prompt
        model_output = ollama.chat(model=model_name, messages=[
            {"role": "user", "content": combined_prompt}
        ])
        model_output = model_output["message"]["content"].strip()

    else:
        print("model don't exist yet")

    return model_output
