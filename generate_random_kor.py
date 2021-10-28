import os
import string
from random import randint


def write_word_to_txt(word_list, output_path):
    with open(os.path.join(output_path,'random_rare_chars.txt'), 'w') as f:
        for word in word_list:
            f.write(f'{word}\n')

def make_words_list(gen_num, rare_chars, word_len_range):
    words = []
    for i in range(gen_num):
        words.append(make_random_word(rare_chars, word_len_range))
    return words

def make_random_word(rare_chars, word_len_range):
    word_len = randint(word_len_range[0], word_len_range[1])
    word = ''.join([rare_chars[x] for x in gen_random_int_list(word_len, len(rare_chars)-1)])
    return word

def gen_random_int_list(gen_num, max_len):
    int_list = []
    for i in range(gen_num):
        int_list.append(randint(0,max_len))
    return int_list

def extract_rare_char_list(class_num_txt_path):
    with open(class_num_txt_path, 'r', encoding='utf8') as f:
        text = f.readlines()
    rare_chars = []
    for line in text:
        char, num = line.replace('\n','').split('\t')
        if int(num) < 1000 and not char in string.printable[10:-38]:
            rare_chars.append(char)
    return rare_chars


if __name__ in '__main__':
    class_num_txt_path = '/home/gucheol/data/hc_recog_data/train/lmdb_char_class_num.txt'
    txt_output_path = '/home/gucheol/data/hc_recog_data/train'
    gen_num = 100000

    rare_chars = extract_rare_char_list(class_num_txt_path)
    # rare_chars = string.printable[10:-38]
    words = make_words_list(gen_num, rare_chars, (1,12))
    write_word_to_txt(words, txt_output_path)
