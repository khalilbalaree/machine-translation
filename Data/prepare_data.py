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
    src_lines = open('en-zh/UNv1.0.en-zh.en', 'r').readlines()
    targ_lines = open('en-zh/UNv1.0.en-zh.zh', 'r', encoding='utf-8').readlines()

    temp_src_lines = []
    temp_targ_lines = []
    for i, src in enumerate(src_lines):
        targ = targ_lines[i]
        if filter_data(src, 'en') and filter_data(targ, 'zh'): # the order is important
            if len(src) > 9 * len(targ) or len(targ) > 9 * len(src):
                continue
            else:
                temp_src_lines.append(src)
                temp_targ_lines.append(targ)


    print("en: %d, zh: %d" % (len(temp_src_lines), len(temp_targ_lines)))

    random.seed(666)
    src_lines = random.sample(temp_src_lines, 200000)
    random.seed(666)
    targ_lines = random.sample(temp_targ_lines, 200000)
    

    with open('cleaned_data/my_en-zh.en','w') as f:
        for src in src_lines:
            f.write(src)
    with open('cleaned_data/my_en-zh.zh','w') as f:
        for targ in targ_lines:
            f.write(targ)


if __name__ == "__main__":
    if not os.path.exists('cleaned_data/'):
        os.makedirs('cleaned_data/')
    if not (os.path.exists('cleaned_data/my_en-zh.en') or os.path.exists('cleaned_data/my_en-zh.en')):
        print("Extracting 200K data from origin dataset...")
        read_data()
    print("Nothing to do...")
