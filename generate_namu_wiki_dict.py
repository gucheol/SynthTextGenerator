import ijson 
from tqdm import tqdm 


chars = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ]

classes = {}
for i in range(ord('가'), ord('힣')+1):
    classes[chr(i)] = 0

for ch in chars:
    classes[ch] = 0 

dictionary = {} 
loop_count = 0 

def strip_valid_word(word):
    result = ''
    for ch in word:
        if classes.get(ch) is not None:
            result += ch 
    return result.upper()


i = 10 
with open('namuwiki_20190312.json') as json_file:
    # parser = ijson.parse(json_file)
    objects = ijson.items(json_file, 'item.text')
    for obj in tqdm(objects):
        words = obj.replace('[[', '').replace(']]', '').replace('==', '').replace('--', '').replace('...', '').replace("'''", '').replace('\\\\','').split()
        for word in words:
            item = strip_valid_word(word)
            if len(item) > 0 : 
                if dictionary.get(item) is not None:
                    dictionary[item] += 1
                else:
                    dictionary[item] = 1 


dictionary_reverse = sorted(dictionary.items(), reverse=True, key=lambda item: item[1])

print('------------------------------')
with open('namuwiki_dictionary.txt', 'w', encoding='utf8') as f: 
    for k, v in dictionary_reverse:
        f.write(f'{k}\t{v}\n')
