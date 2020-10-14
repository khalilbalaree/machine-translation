import re, os, string, jieba
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def getChinese(context):
    # context = context.decode("utf-8") # convert context from str to unicode
    filtrate = re.compile(u'[^\u4E00-\u9FA5]') # non-Chinese unicode range
    context = filtrate.sub(r'', context) # remove all non-Chinese characters
    # context = context.encode("utf-8") # convert unicode back to str
    return context


def clean_zh():
    if not os.path.exists('cleaned_data/my_en-zh.zh'):
        exit("run prepare.data first...")
    targ_lines = open('cleaned_data/my_en-zh.zh', 'r').readlines()
    zh_sw = [l.strip() for l in open('zh_stopwords.txt','r').readlines()]
    zh = []
    for targ in targ_lines:
        targ = jieba.lcut(getChinese(targ))
        clean = [t for t in targ if t not in zh_sw]
        zh.append(clean)

    return zh

def clean_en():
    if not os.path.exists('cleaned_data/my_en-zh.en'):
        exit("run prepare.data first...")
    src_lines = open('cleaned_data/my_en-zh.en','r').readlines()
    en_sw = set(stopwords.words('english'))
    punc = set(string.punctuation)
    en = []
    for src in src_lines:
        src = word_tokenize(src)
        clean = [t for t in src if t not in (en_sw and punc)]
        en.append(clean)

    return en
