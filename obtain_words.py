import io

def load_words(filename, index):
    filein = io.open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore')
    words = []
    for line in filein:
        tokens = line.rstrip().split(',')
        words.append(tokens[index])
    return words

def write_file(filename, words):
    with open(filename, 'a') as f:
        for i in words:
            f.write(i)
            f.write('\n')

w = load_words('word_transform/eval.vocab', 1)
write_file('en_words', w)
