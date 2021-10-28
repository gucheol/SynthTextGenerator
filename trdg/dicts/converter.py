with open('kr.txt', 'r', encoding='ms949') as in_f:
    with open('kr_utf8.txt', 'w', encoding='utf-8') as out_f:
        while True:
            line = in_f.readline()
            if not line:
                break 
            out_f.write(line)
