#-*- coding:utf-8 -*-
import random, re

def filter_data(line, clean_mode):
    line = line.strip()
    if clean_mode == 'en':
        if (not bool(re.search(r'[0-9]|\"|\#|\$|\%|\&|\\|\'|\(|\)|\*|\+|\-|\/|\:|\;|\<|\=|\>|\@|\[|\]|\^|\_|\`|\{|\||\}|\~', line))) and line[-1] == '.' :
            return True
        else:
            return False
    elif clean_mode == 'zh':
        return not bool(re.search(r'[A-Za-z0-9]|\u300a|\u300b|\uff08|\uff09|\u201c|\u201d', line)) # 《》（） “”


def read_data():
    src_lines = open('en-zh/UNv1.0.en-zh.en').readlines()
    targ_lines = open('en-zh/UNv1.0.en-zh.zh', encoding='utf-8').readlines()

    temp_src_lines = []
    temp_targ_lines = []
    for i, src in enumerate(src_lines):
        targ = targ_lines[i]
        if filter_data(src, 'en') and filter_data(targ, 'zh'): # the order is important
            temp_src_lines.append(src)
            temp_targ_lines.append(targ)


    print("en: %d, zh: %d" % (len(temp_src_lines), len(temp_targ_lines)))

    random.seed(666)
    src_lines = random.sample(temp_src_lines, 200000)
    random.seed(666)
    targ_lines = random.sample(temp_targ_lines, 200000)
    

    with open('my_en-zh.en','w') as f:
        for src in src_lines:
            f.write(src)
    with open('my_en-zh.zh','w') as f:
        for targ in targ_lines:
            f.write(targ)

read_data()