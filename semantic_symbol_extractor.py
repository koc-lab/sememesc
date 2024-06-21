import OpenHowNet
import scipy
import numpy as np
import nltk
import spacy
hownet_dict = OpenHowNet.HowNetDict()
#%%
f_words = "f_words.txt"
with open(f_words,encoding='utf8') as f:
    s_words = f.readlines()
for w in range(len(s_words)):
    i = s_words[w].find(' ')
    s_words[w] = s_words[w][:i]
#%% Musa's Method
sememe_list_path = "SKB_mrd2skb/mrd2skb_sememes.txt"
sememe_dict_path = "SKB_mrd2skb/mrd2skb_skb.txt"
#sememe_dict_path = "SKB_mrd2skb/hownet_en.txt"
#lemmatization_path = "DictSKB-main/NLI/dataset/lemmatization.txt"
with open(sememe_dict_path,encoding='utf8') as f:
    sememe_dict = f.readlines()
with open(sememe_list_path,encoding='utf8') as f:
    sememe_list = f.readlines()
# with open(lemmatization_path,encoding='utf8') as f:
#     lemmatization = f.readlines()
# #%% New Sememe Generation Method
# sememe_dict_path = "DictSKB-main/NLI/dataset/sememe_dict.txt"
# sememe_list_path = "DictSKB-main/NLI/dataset/sememe.txt"
# lemmatization_path = "DictSKB-main/NLI/dataset/lemmatization.txt"
# with open(sememe_dict_path,encoding='utf8') as f:
#     sememe_dict = f.readlines()
# with open(sememe_list_path,encoding='utf8') as f:
#     sememe_list = f.readlines()
# with open(lemmatization_path,encoding='utf8') as f:
#     lemmatization = f.readlines()
#%%
sememe_dictionary = {}
for line in range(0,len(sememe_dict),2):
    word = sememe_dict[line].strip()
    if len(word.split()) == 1:
        sememes = sememe_dict[line+1].split()
        if len(sememes) > 0:
            sememe_dictionary[word] = sememes
#%% SPACY SIMILARITY
import gensim.downloader as api
from sklearn.metrics.pairwise import cosine_similarity
model = api.load("glove-twitter-200")
#v_1 = model.get_vector('phone').reshape(1,-1)
#v_2 = model.get_vector('computer').reshape(1,-1)
#sim = cosine_similarity(v_1,v_2)
#%%
sememe_glove_dict = {}
word_glove_dict = {}
for k in list(sememe_dictionary.keys()):
    sememes  = sememe_dictionary[k]
    for w in sememes:
        if w not in sememe_glove_dict:
            try:
                v_i = model.get_vector(w).reshape(1,-1)
                sememe_glove_dict[w] = v_i
            except:
                v_i = np.zeros((1,200))
                sememe_glove_dict[w] = v_i
    try:
        word_glove_dict[k] = model.get_vector(k).reshape(1,-1)
    except:
        word_glove_dict[k] = np.zeros((1,200))
# #%% ABLATION STUDY İÇİN RANDOM SEÇME SEMEMELERİ
# chosen_sememe_dictionary = {}
# for k in list(sememe_dictionary.keys()):
#     if len(sememe_dictionary[k]) < 4:
#         chosen_sememe_dictionary[k] = sememe_dictionary[k]
#     else:
#         chosen_sememe_dictionary[k] = sememe_dictionary[k][:3]

#%% SEMEME CHOSING WITH GLOVE
chosen_sememe_dictionary = {}
for key in word_glove_dict.keys():
    sememes = sememe_dictionary[key]
    top_sim = np.ones((1,(len(sememes))))*-1
    v_1 = word_glove_dict[key]
    top_k_sem = []
    for s in range(len(sememes)):
        v_2 = sememe_glove_dict[sememes[s]]
        top_sim[0,s] = cosine_similarity(v_1,v_2)
        try:
            ind = np.argsort(top_sim)[:,-3:]
        except:
            try:
                ind = np.argpartition(top_sim)[:-2]
            except:
                ind = np.argpartition(top_sim)[:-1]
    for idx in ind[0,:]:
        top_k_sem.append(sememes[idx])
    chosen_sememe_dictionary[key]  = top_k_sem
#%% 
# from itertools import combinations
# from scipy.spatial import distance
# chosen_sememe_dictionary = {}
# for key in word_glove_dict.keys():
#     sememes = sememe_dictionary[key]
#     if len(sememes) < 4 :
#         chosen_sememe_dictionary[key] = sememes
#     else:
#         comb = list(combinations(sememes,3))
#         distances = np.zeros(len(comb))
#         for i in range(len(comb)):
#             w_1 = sememe_glove_dict[comb[i][0]]
#             w_2 = sememe_glove_dict[comb[i][1]]
#             w_3 = sememe_glove_dict[comb[i][2]]
#             distances[i] += distance.euclidean(w_1,w_2)
#             distances[i] += distance.euclidean(w_1,w_3)
#             distances[i] += distance.euclidean(w_2,w_3)
#         chosen_sememe_dictionary[key] = list(comb[np.argmax(distances)])
#%%
# import pickle
# with open('new_algo_chosen_sememe_dictionary', 'wb') as handle:
#     pickle.dump(chosen_sememe_dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)
#%% RANDOM CHOSING
limited_sememe_dictionary = {}
for key in sememe_dictionary.keys():
    sememe_limited = sememe_dictionary[key][0:3]
    limited_sememe_dictionary[key] = sememe_limited
#%% Generate a list of English sememes in HowNet
key_list = list(hownet_dict.sememe_dic.keys())
key_list_en =[]
for word in key_list:
    for i in range(len(word)):
        if word[i] == '|':
            key_list_en.append(word[0:i])

#%% LOAD MUSA GROUNDTRUTH
import pandas as pd
import pickle
with open('train_sentences.pickle', 'rb') as handle:
    sentence_list_train = pickle.load(handle)
with open('validation_sentences.pickle', 'rb') as handle:
    sentence_list_val = pickle.load(handle)
with open('test_sentences.pickle', 'rb') as handle:
    sentence_list_test = pickle.load(handle)
# sentence_list_train =list(pd.read_csv('europarl_dataset_processed/train_data_ner_036.csv')['English'])
# sentence_list_val = list(pd.read_csv('europarl_dataset_processed/validation_data_ner_036.csv')['English'])
# sentence_list_testt= scipy.io.loadmat('europarl_dataset_processed/test_groundtruth_ner.mat')['sentence_list_sonnn']
#%%
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

class custom_lemmatizer:
    tag_dict = {"ADJ": wordnet.ADJ,
                "NOUN": wordnet.NOUN,
                "VERB": wordnet.VERB,
                "ADV" : wordnet.ADV,
                }
    lemmatizer = WordNetLemmatizer()

    def lemmatize(self, word_pos_tuple):
        word = word_pos_tuple[0]
        pos_tag = word_pos_tuple[1]
        if pos_tag in self.tag_dict:
            return self.lemmatizer.lemmatize(word,
                                             self.tag_dict[pos_tag]).lower()
        else:
            return word.lower()
cm = custom_lemmatizer()
#%%
def sememes_from_word_dict(word,s_words,sememe_dict):
    if word in s_words:
        return [word]
    elif word[-6:] =='entity':
        return [word[:-6]]
    else:
        try:
            word_sememes = sememe_dict[word]
            return word_sememes[:3]
        except:
            return [word]
#%% DATASET GENERATION
def sememes_from_word_limited(word,s_words,max_sem):
    if word in s_words:
        return [word]
    else:
        sememes = hownet_dict.get_sememes_by_word(word = word, display='list', merge=True, K=None)
        word_sememes = []
        for i in range(min(max_sem,len(sememes))):
            word_sememes.append(str(sememes[i]).split("|")[0])
        if len(word_sememes) > 0:
            return word_sememes
        else:
            # for ch in word:
            #     word_sememes.append(ch)
            # return word_sememes
            return [word]
#%%
def sememes_from_word(word,s_words):
    if word in s_words:
        return [word]
    else:
        sememes = hownet_dict.get_sememes_by_word(word = word, display='list', merge=True, K=None)
        word_sememes = []
        for i in range(len(sememes)):
            word_sememes.append(str(sememes[i]).split("|")[0])
        if len(word_sememes) > 0:
            return word_sememes
#%%
def sememes_from_word_char(word,s_words):
    if word in s_words:
        return [word]
    else:
        sememes = hownet_dict.get_sememes_by_word(word = word, display='list', merge=True, K=None)
        word_sememes = []
        for i in range(len(sememes)):
            word_sememes.append(str(sememes[i]).split("|")[0])
        if len(word_sememes) > 0:
            return word_sememes
        else:
            for ch in word:
                word_sememes.append(ch)
            return word_sememes
def extract_words_only(word, word_list):
    return[word]
def sememe_extraction(dataset):
    to_tokenize = dataset
    lemmatized_s_list = []
    for s in range(len(to_tokenize)):
        tokenized_s = nltk.tokenize.word_tokenize(to_tokenize[s])
        tagged_s = nltk.tag.pos_tag(tokenized_s, tagset='universal')
        lemm = []
        for t in tagged_s:
            lemmatized_s = cm.lemmatize(t)
            lemm.append(lemmatized_s)
        lemmatized_s_list.append(lemm)
    extracted_sememes_sent = []
    for sentence in lemmatized_s_list:
        ext_sem = []
        for word in sentence:
            #s_w = sememes_from_word_limited(word,s_words,3)
            s_w = sememes_from_word_dict(word,s_words,chosen_sememe_dictionary)
            #s_w = sememes_from_word_char(word,s_words)
            #s_w = extract_words_only(word,s_words)
            ext_sem.append(s_w)
        extracted_sememes_sent.append(ext_sem)
    extracted_sememes_one_list = []
    for sent in extracted_sememes_sent:
        list_sememes = []
        for semm in sent:
            if len(semm) > 0:
                for w in semm:
                    list_sememes.append(w)
                #list_sememes.append('spacee')
        extracted_sememes_one_list.append(list_sememes)

    extracted_sememes_text = []
    for semm in extracted_sememes_one_list:
        txt = ''
        for w in semm:
            txt += w
            txt += ' '
        extracted_sememes_text.append(txt)
    return extracted_sememes_text
#%%
train_data = sememe_extraction(sentence_list_train)
test_data = sememe_extraction(sentence_list_test)
val_data = sememe_extraction(sentence_list_val)
#%%
import csv

with open('train_raw.pickle', 'rb') as handle:
    train_raw = pickle.load(handle)
with open('validation_raw.pickle', 'rb') as handle:
    val_raw = pickle.load(handle)
with open('test_raw.pickle', 'rb') as handle:
    test_raw = pickle.load(handle)
new_list = []
for s in range(120000):
    new_list.append([train_data[s],train_raw[s]])
df = pd.DataFrame(new_list, columns =['Sememe', 'English'])
df.to_csv("./train_data.csv", sep=',',index=False)
new_list = []
for s in range(25677):
    new_list.append([test_data[s],test_raw[s]])
df = pd.DataFrame(new_list, columns =['Sememe', 'English'])
df.to_csv("./test_data.csv", sep=',',index=False)
new_list = []
for s in range(20000):
    new_list.append([val_data[s],val_raw[s]])
df = pd.DataFrame(new_list, columns =['Sememe', 'English'])
df.to_csv("./validation_data.csv", sep=',',index=False)