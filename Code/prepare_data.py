import random, re, os

def filter_data(line, clean_mode):
    line = line.strip()
    if clean_mode == 'en':
        if (not bool(re.search(r'[0-9]|\"|\#|\$|\%|\&|\\|\'|\(|\)|\*|\+|\-|\/|\:|\;|\<|\=|\>|\@|\[|\]|\^|\_|\`|\{|\||\}|\~', line))) \
            and line[-1] == '.' \
            and len(line) > 10:
            return True
        else:
            return False
    elif clean_mode == 'zh':
        return not bool(re.search(r'[A-Za-z0-9]|《|》|「|」|【|】|（|）|“|‘|\*|\'|\-', line))

def read_data():
    src_lines = open('../Data/UNv1.0.en-zh/UNv1.0.en-zh.en', 'r', encoding='utf-8').readlines()
    targ_lines = open('../Data/UNv1.0.en-zh/UNv1.0.en-zh.zh', 'r', encoding='utf-8').readlines()

    temp_src_lines = []
    temp_targ_lines = []
    for i, src in enumerate(src_lines):
        targ = targ_lines[i]
        if filter_data(src, 'en') and filter_data(targ, 'zh'): # the order is important
            if len(src) > 100 or len(src) > 9 * len(targ) or len(targ) > 9 * len(src) or len(targ) <=10 or len(src) <=10:
                continue
            else:
                temp_src_lines.append(src)
                temp_targ_lines.append(targ)


    print("en: %d, zh: %d" % (len(temp_src_lines), len(temp_targ_lines)))

    random.seed(666)
    src_lines = random.sample(temp_src_lines, 50000)
    random.seed(666)
    targ_lines = random.sample(temp_targ_lines, 50000)
    

    with open('../Data/cleaned_data/my_en-zh.en','w', encoding='utf-8') as f:
        for src in src_lines:
            f.write(src)
    with open('../Data/cleaned_data/my_en-zh.zh','w', encoding='utf-8') as f:
        for targ in targ_lines:
            f.write(targ)


if __name__ == "__main__":
    if not os.path.exists('../Data/cleaned_data/'):
        os.makedirs('../Data/cleaned_data/')
    if not (os.path.exists('../Data/cleaned_data/my_en-zh.en') or os.path.exists('../Data/cleaned_data/my_en-zh.en')):
        print("Extracting 200K data from origin dataset...")
        read_data()
    print("Nothing to do...")
