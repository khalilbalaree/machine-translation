import spacy, re, string
import numpy as np
def getChinese(context):
    filtrate = re.compile(u'[^\u4E00-\u9FA5]') # non-Chinese unicode range
    context = filtrate.sub(r'', context) # remove all non-Chinese characters
    return context

def zh_embeddings():
    zh_nlp = spacy.load("zh_core_web_md")
    all_stopwords = [l.strip() for l in open('../Data/zh_stopwords.txt','r',encoding='utf-8').readlines()]
    targ_lines = open('../Data/cleaned_data/my_en-zh.zh', 'r',encoding='utf-8').readlines()

    result = []
    counter = 0
    for targ in targ_lines:
        targ = zh_nlp(getChinese(targ))

        line = []
        for token in targ:
            if token.text not in all_stopwords:
                line.append(token.vector)

        result.append(np.array(line,dtype=float))
        if counter % 2000 == 0:
            print(counter, "/", len(targ_lines))
        counter += 1

    return result

def en_embeddings():
    en_nlp = spacy.load("en_core_web_md")
    all_stopwords = set(en_nlp.Defaults.stop_words)
    punc = set(string.punctuation)

    src_lines = open('../Data/cleaned_data/my_en-zh.en', 'r',encoding='utf-8').readlines()

    result = []
    counter = 0
    for src in src_lines:
        src = en_nlp(src.strip())

        line = []
        for token in src:
            if token.text not in all_stopwords and punc:  #
                #print(token.text)
                line.append(token.vector)
        if counter % 2000 == 0:
            print(counter, "/", len(src_lines))
        counter += 1
        result.append(np.array(line,dtype=float))

    return result
emb_zh = zh_embeddings()
emb_zh = np.asarray(emb_zh)

import torch
np.save('emb_zh.npy',emb_zh)
emb_en = en_embeddings()
emb_en = np.asarray(emb_en)
np.save('emb_en.npy',emb_en)