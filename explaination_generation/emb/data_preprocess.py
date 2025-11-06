import pickle
import numpy as np

with open("train_embedding.pkl",'rb') as datafile:
    data=pickle.load(datafile)
with open("train_label.pkl",'rb') as labelfile:
    label=pickle.load(labelfile)

label_list=[]
data_list=[]
for k,v in label.items():
    print(k)
    label_list.append(v)
    data_list.append(data[k])

np.save('train_data.npy',np.array(data_list))
np.save('train_label.npy',np.array(label_list))
