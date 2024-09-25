import openai
from config import CONFIG

"""Codex completion"""

def codex_completion(prompt_text):
    openai.api_key = CONFIG['openai_api_key']
    return openai.Completion.create(
        engine='code-davinci-002',
        prompt=prompt_text,
        max_tokens=150,
        temperature=0,
        stop=['[eos]'],
    )["choices"][0]["text"]

def codex_completion_gpt3(prompt_text):
    openai.api_key = CONFIG['openai_api_key']
    return openai.Completion.create(
        model='text-davinci-003',
        prompt=prompt_text, 
        max_tokens=150,
        temperature=0,
        stop=['[eos]'],
    )["choices"][0]["text"]


def codex_completion_chatgpt(prompt_text):
    openai.api_key = CONFIG['openai_api_key']
    return openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[{"role": "system", "content": "You are a helpful assistant." + prompt_text}]
    )["choices"][0]["message"]["content"]
    