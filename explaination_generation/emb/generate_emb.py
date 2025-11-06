import json
import pickle

import numpy as np
import ollama
import csv


results= {}
csv_reader=csv.reader(open("../user/test.tsv"))
for row in csv_reader:
    #user_id, click, imp, label = row[0].split("\t")  #train
    reword_id, user_id, click, imp = row[0].split("\t")  # train
    imp_papers = imp.strip().split()
    with open("../user/test_user_profile_results/"+str(user_id)+".json", 'r') as user_prof:
        user_profile = json.load(user_prof)
        user_profile = user_profile.split("</think>")[1]
        if "json\n{\n    \"summarization\": \"" in user_profile:
            user_profile = user_profile.replace("user_profile",'')
        if "\",\n    \"reasoning\": \"" in user_profile:
            user_profile = user_profile.replace("\",\n    \"reasoning\": \"",' ')
        print("user_profile:",user_profile)
        user_profile_response = ollama.embed(model="mxbai-embed-large", input=user_profile)
        user_profile_embeddings = user_profile_response["embeddings"][0]

        for imp_paper in imp_papers:
            print(imp_paper)
            if "-" in imp_paper:
                imp_paper=imp_paper.split('-')[0]
            with open("../item/test_paper_profile_results/" + imp_paper + ".json", 'r') as paper_prof:
                paper_profile = json.load(paper_prof)
                if "\"summarization\": \"" in paper_profile:
                    paper_profile = paper_profile.split("\"summarization\": \"")[1].replace(
                        "\",\n    \"reasoning\": \"", ' ')
                else:
                    paper_profile = paper_profile.split("</think>")[1].replace("\",\n    \"reasoning\": \"", ' ')

            print(imp_paper + "_paper_profile:", paper_profile)
            with open("../explanation/test_negtive_explanation_results/" + user_id+'_'+imp_paper+'.json', 'r') as negative_expl:
                negative_explanation = json.load(negative_expl)
                negative_explanation = negative_explanation.split("</think>")[1].replace("\",\n    \"reasoning\": \"", ' ')

            with open("../explanation/test_positive_explanation_results/" + user_id+'_'+imp_paper+'.json', 'r') as positive_expl:
                positive_explanation = json.load(positive_expl)
                positive_explanation = positive_explanation.split("</think>")[1].replace("\",\n    \"reasoning\": \"", ' ')

            print(imp_paper + "+positive_explanation:", positive_explanation)

            # store each document in a vector embedding database
            paper_profile_response = ollama.embed(model="mxbai-embed-large", input=paper_profile)
            paper_profile_embeddings = paper_profile_response["embeddings"][0]
            print(imp_paper + "+paper_profile_embeddings:", paper_profile_embeddings)
            negative_explanation_response = ollama.embed(model="mxbai-embed-large", input=negative_explanation)
            negative_explanation_embeddings = negative_explanation_response["embeddings"][0]
            print(imp_paper + "+negative_explanation_embeddings:", negative_explanation_embeddings)
            positive_explanation_response = ollama.embed(model="mxbai-embed-large", input=positive_explanation)
            positive_explanation_embeddings = positive_explanation_response["embeddings"][0]
            print(imp_paper + "+positive_explanation_embeddings:", positive_explanation_embeddings)

            results[user_id+'_'+imp_paper] = user_profile_embeddings+paper_profile_embeddings+negative_explanation_embeddings+positive_explanation_embeddings

with open('test_embedding.pkl', 'wb') as pickle_file:
    pickle.dump(results,pickle_file)

