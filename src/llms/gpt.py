import os
import openai


openai.api_type = "azure"
openai.api_base = "https://arabic-dialects-llm-translation.openai.azure.com/"
openai.api_version = "2023-09-15-preview"
openai.api_key = os.getenv("OPENAI_API_KEY") 

def request_gpt(prompt):
    response = openai.Completion.create(
        engine="gpt-35-turbo-model",
        prompt=prompt,
        temperature=0,
        max_tokens=100,
        top_p=0.5,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )
    return response["choices"][0]["text"]

# cut off after newline (may need to do something more involved with other prompts)
def postprocess(mt_text):
    first_line = mt_text.split('\n')[0] 
    return first_line.strip()


