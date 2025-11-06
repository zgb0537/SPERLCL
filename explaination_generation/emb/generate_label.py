import json
import pickle

import numpy as np
import csv


results= {}
csv_reader=csv.reader(open("../user/train.tsv"))
for row in csv_reader:
    user_id, click, imp, label = row[0].split("\t")  #train
    #reword_id, user_id, click, imp = row[0].split("\t")  # test
    imp_papers = imp.strip().split()
    labels = label.strip().split()
    for i in range(len(imp_papers)):
        imp_paper= imp_papers[i]
        lab = labels[i]
        if "-" in imp_paper:
            imp_paper1 = imp_paper.split('-')[0]
            results[user_id+'_'+imp_paper1]=imp_paper.split('-')[1]
        else:
            results[user_id + '_' + imp_paper] = lab


with open('train_label.pkl', 'wb') as pickle_file:
    pickle.dump(results,pickle_file)