import pandas as pd
import numpy as np
df_train=pd.read_csv("train_data.csv", header=0, usecols=[0,1])
df_val=pd.read_csv("validation_data.csv", header=0, usecols=[0,1])
#%%
sememe_dict = {}
for i in range(120000):
    sentence = df_train['Sememe'][i].split()
    for w in sentence:
        if w in sememe_dict:
           sememe_dict[w] += 1
        else:
           sememe_dict[w] = 1
#%%
words = []
counts = []
for key in sememe_dict.keys():
    words.append(key)
    counts.append(sememe_dict[key])
counts = np.array(counts)
total = np.sum(counts)
probs = counts/total
#%%
s_n_parameter = 0.7
import random
for i in range(120000):
    if i%1000 == 0:
        print(i)
    sentence = df_train['Sememe'][i].split()
    s_len = len(sentence)
    k_p = np.random.randn(s_len,1)
    for j in range(s_len):
        if k_p[j] > s_n_parameter:
            #sentence[j] = np.random.choice(words,probs)
            sentence[j] = random.choices(words, weights=probs,k=1)[0]
    sen_new = ''
    for s in sentence:
        sen_new += s
        sen_new += ' '
    df_train['Sememe'][i] = sen_new.strip()
for i in range(20000):
    sentence = df_val['Sememe'][i].split()
    s_len = len(sentence)
    k_p = np.random.randn(s_len,1)
    for j in range(s_len):
        if k_p[j] > s_n_parameter:
           #sentence[j] = np.random.choice(words,probs)
           sentence[j] = random.choices(words, weights=probs,k=1)[0]
    sen_new = ''
    for s in sentence:
        sen_new += s
        sen_new += ' '
    df_val['Sememe'][i] = sen_new.strip()
#%%
df_train.to_csv("./train_data_sn07.csv", sep=',',index=False)
df_val.to_csv("./validation_data_sn07.csv", sep=',',index=False)