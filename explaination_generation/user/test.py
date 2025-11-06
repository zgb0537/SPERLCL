from ollama import chat
from ollama import ChatResponse
import json
import numpy as np


def get_gpt_response_w_system(prompt):
    global system_prompt
    response: ChatResponse = chat(model='deepseek-r1:14b', messages=[
      {'role': 'system', 'content': system_prompt},
      {"role": "user", "content": prompt}
    ])

    response_con = response['message']['content']
    return response_con

# read the system_prompt (Instruction) for user profile generation
system_prompt = ""
with open('./user_system_prompt.txt', 'r') as f:
    for line in f.readlines():
        system_prompt += line

# read the example prompts of users
class Colors:
    GREEN = '\033[92m'
    END = '\033[0m'

print(Colors.GREEN + "Generating Profile for User" + Colors.END)
print("---------------------------------------------------\n")
print(Colors.GREEN + "The System Prompt (Instruction) is:\n" + Colors.END)
print(system_prompt)
print("---------------------------------------------------\n")
print(Colors.GREEN + "The Input Prompt is:\n" + Colors.END)

with open('./test_user_prompts.json', 'r') as f:
    u_prompt = json.load(f)
    print(type(u_prompt))
    for k, v in u_prompt.items():
        print(k)
        print(v)
        response = get_gpt_response_w_system(v)
        print(Colors.GREEN + "Generated Results:\n" + Colors.END)
        print(response)

        with open("./test_user_profile_results/"+k.split("_")[1]+".json", 'w', encoding="utf8") as f:
            json.dump(response, f)
            f.close()



