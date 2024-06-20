import OpenHowNet
import scipy
import numpy as np
import nltk
import spacy
import os
path = "europarl/en/"
dir_list = os.listdir(path)
#%%
total_corp = []
#filename = "europarl/en/ep-00-01-17.txt"
for filename in dir_list[15:115]:
    with open(path+filename,encoding='utf8') as f:
        corp = f.readlines()
    total_corp += corp
#%%
cleaned_list = []
for s in total_corp:
    if s[0] != '<' and len(s) > 1:
        cleaned_list.append(s)
#%%
cleaned_txt = ''
for s in cleaned_list:
    cleaned_txt += s
    
#%%
#take unicode string  
#here u stands for unicode
nlp = spacy.load("en_core_web_sm")
nlp.max_length = 1000000
sentence_list = []
#%%
for i in range(35):
    doc = nlp(cleaned_txt[i*nlp.max_length:nlp.max_length*(i+1)])
    for sent in doc.sents:
      sentence_list.append(sent)
#%%
import random
sentence_list_son = []
for s in sentence_list:
    sentence_list_son.append(str(s))
random.shuffle(sentence_list_son)

#%% NAMED ENTITY LABELLING
deletetion_symbols = '''!(“%-[\,?']‘{^><)}"”;:’@#$./&*_~'''
nlp.pipe_labels['ner']
sentence_list_sonn = []
for s in range(len(sentence_list_son)):
#for s in range(10):
    doc = nlp(sentence_list_son[s])
    if s % 100 == 0:
        print(s)
    entities = ""
    new_txt = ""
    final_txt =""
    token_list = []
    for ent in doc.ents:
        if ent.label_ in ['PERSON', 'LANGUAGE', 'CARDINAL', 'LOC','GPE','FACILITY','DATE','TIME']:
            entities += str(ent).lower()
            entities += " "
            entity_list = entities.split()
    for x in str(doc):
        if x not in deletetion_symbols:
          a= x.lower()
          if a=="\n":
            a=" "
          new_txt += a
    tokens = new_txt.split()
    for t in range(len(tokens)):
        if tokens[t] in entities.split():
            tokens[t] += "ENTITY"
    for tok in tokens:
        final_txt += tok
        final_txt += " "
    sentence_list_sonn.append(final_txt)
    
#%%
sentence_list_sonnn = []
for w in range(len(sentence_list_sonn)):
    w_count = len(sentence_list_sonn[w].split())
    if w_count > 3 and w_count < 31:
        sentence_list_sonnn.append(sentence_list_sonn[w])
#%%
sentence_list_without_entity_tag = []
for w in range(len(sentence_list_sonnn)):
    word_list = sentence_list_sonnn[w].split()
    new_txt = ""
    for j in range(len(word_list)):
        if word_list[j][-6:] == 'ENTITY':
            new_txt += word_list[j][:-6]
        else:
            new_txt += word_list[j]
        new_txt += ' '
    sentence_list_without_entity_tag.append(new_txt.strip())
#%%
import random
#random.seed(42)
#random.shuffle(sentence_list_sonnn)
train_sentences = sentence_list_sonnn[:120000]
validation_sentences = sentence_list_sonnn[120000:140000]
test_sentences = sentence_list_sonnn[140000:]

train_raw = sentence_list_without_entity_tag[:120000]
validation_raw = sentence_list_without_entity_tag[120000:140000]
test_raw = sentence_list_without_entity_tag[140000:]
#%%
import pickle
with open('train_sentences.pickle', 'wb') as handle:
    pickle.dump(train_sentences, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('validation_sentences.pickle', 'wb') as handle:
    pickle.dump(validation_sentences, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('test_sentences.pickle', 'wb') as handle:
    pickle.dump(test_sentences, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('train_raw.pickle', 'wb') as handle:
    pickle.dump(train_raw, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('validation_raw.pickle', 'wb') as handle:
    pickle.dump(validation_raw, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('test_raw.pickle', 'wb') as handle:
    pickle.dump(test_raw, handle, protocol=pickle.HIGHEST_PROTOCOL)