import json, csv
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
with open('./positive_explanation_system_prompt.txt', 'r',encoding="utf8") as f:
    for line in f.readlines():
        system_prompt += line

# read the example prompts of items
class Colors:
    GREEN = '\033[92m'
    END = '\033[0m'

csv_reader=csv.reader(open("../user/test.tsv"))
for row in csv_reader:
    #user_id, click, imp, label = row[0].split("\t")  #train
    record_id, user_id, click, imp = row[0].split("\t")  #test
    print(user_id)
    print(imp)
    imp_papers = imp.strip().split()
    print(imp_papers)
    with open("../user/test_user_profile_results/"+str(user_id)+".json", 'r') as user_prof:
        user_profile = json.load(user_prof)
        user_profile = user_profile.split("</think>")[1]
        if "json\n{\n    \"summarization\": \"" in user_profile:
            user_profile = user_profile.replace("user_profile",'')
        if "\",\n    \"reasoning\": \"" in user_profile:
            user_profile = user_profile.replace("\",\n    \"reasoning\": \"",' ')
        print(user_profile)


    for imp_paper in imp_papers:
        paper_id = imp_paper.split('-')[0]
        print(paper_id)
        with open("../item/test_paper_profile_results/"+paper_id+".json", 'r') as paper_prof:
            paper_profile = json.load(paper_prof)
            if "\"summarization\": \"" in paper_profile:
                paper_profile = paper_profile.split("\"summarization\": \"")[1].replace("\",\n    \"reasoning\": \"", ' ')
            else:
                paper_profile = paper_profile.split("</think>")[1].replace("\",\n    \"reasoning\": \"",' ')

        v="User Profile: "+user_profile + '  /n' + "Paper Profile: " + paper_profile
        response = get_gpt_response_w_system(v)
        print(Colors.GREEN + "Generated Results:\n" + Colors.END)
        print(response)

        with open("./test_positive_explanation_results/"+user_id+"_"+paper_id+".json", 'w', encoding="utf8") as f1:
            json.dump(response, f1)
            f1.close()

