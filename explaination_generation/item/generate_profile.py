import json
from ollama import chat
from ollama import ChatResponse
import numpy as np



def get_gpt_response_w_system(prompt):
    global system_prompt
    response: ChatResponse = chat(model='deepseek-r1:14b', messages=[
      {'role': 'system', 'content': system_prompt},
      {"role": "user", "content": prompt}
    ])

    response_con = response['message']['content']
    return response_con


# read the system_prompt (Instruction) for item profile generation
system_prompt = ""
with open('./item_system_prompt.txt', 'r') as f:
    for line in f.readlines():
        system_prompt += line

# read the example prompts of items
class Colors:
    GREEN = '\033[92m'
    END = '\033[0m'

example_prompts = []
with open('./test_item_prompts.json', 'r') as f:
    u_prompt = json.load(f)
    for k, v in u_prompt.items():
        response = get_gpt_response_w_system(v)
        print(Colors.GREEN + "Generated Results:\n" + Colors.END)
        print(response)
        with open("./test_paper_profile_results/"+k.split("_")[1]+".json", 'w', encoding="utf8") as f1:
            json.dump(response, f1)
            f.close()
