with open('char_stat.txt', 'r', encoding='utf8') as f :
    lines = f.readlines()

classes = {}
for index, line in enumerate(lines):    
    if index > 2000:
        break 
    items = line.strip().split(',')
    classes[items[0]] = 0

with open('ko_labels_refine.txt', 'r', encoding='utf8') as f:
    lines = f.readlines()

target_classes = {}
for index, line in enumerate(lines):    
    items = line.strip().split(' ')
    target_classes[items[1]] = 0

print ('-------------------------------')
for k in classes.keys():    
    if target_classes.get(k) is None:
        print(items[1])
    else:
        target_classes[k] = 1

print ('-------------------------------')

f = open('ko_labels_refine2.txt', 'w', encoding='utf8')

index = 0 
for k in sorted(target_classes.keys()):
    if target_classes[k] > 0 : 
        f.write(f'{index} {k}\n')
        index += 1

    

    

