import ijson 
# from collections import OrderedDict 

# # with open ('amuwiki_20190312.json') as json_file: 

# #     json_data = ijson.items(json_file) 
# #     for json_object in json_data : 
# #         print (json_object)
# #         break
# class TopNDict():
#     def __init__(self, max_item):
#         self.dict = OrderedDict()
#         self.max_item = max_item
#         self.count = 0 
    
#     def push(self, key):
#         if self.dict.get(key) is None: 
#             if self.count < self.max_item :
#                 self.dict[key] = 1                 
#             else:
#                 self.pop() 
#                 self.dict[key] = 1
#             self.count +=1 
#         else:
#             self.dict[key] +=1 
    
#     def pop(self):
#         min_value = 10240 
#         popped = False
#         for key, value in self.dict.items():
#             min_value = min(min_value, value)
        
#         for key, value in self.dict.items():
#             if value == min_value :
#                 del self.dict [key]
#                 popped = True
#                 break         
#         if popped == False : 
#             raise Exception('pop {} is failed '.format(min_value))
#         self.count -= 1 

#     def print(self):
#         for key, value in sorted(self.dict.items(), key=lambda item: item[1]):
#             print ("key:{}, value:{}".format(key,value))
        

# namu_dict = TopNDict(1000) 

chars = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '(', ')', '-', '.', ',', '?', '!', '@', '#', '%', '*', '<', '>', '+', '~', ':', 
         ';', '|', '^', '_', '=', '$', '/', '\\', '{', '}', '[', ']']

classes = {}
for i in range(ord('가'), ord('힣')+1):
    classes[chr(i)] = 0

for ch in chars:
    classes[ch] = 0 
    
from tqdm import tqdm 

loop_count = 0 
with open('namuwiki_20190312.json') as json_file:
    # parser = ijson.parse(json_file)
    objects = ijson.items(json_file, 'item.text')
    for obj in tqdm(objects):
        words = obj.replace('[[', '').replace(']]', '').replace('==', '').replace('--', '').replace('...', '').replace("'''", '').replace('\\\\','').split()
        for word in words:
            for ch in word:
                if classes.get(ch) is not None:
                    classes[ch] += 1

classes_reverse = sorted(classes.items(), reverse=True, key=lambda item: item[1])
print('------------------------------')
for k, v in classes_reverse:
    print (f'{k}, {v}')
