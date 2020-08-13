from obtain_words import load_words
from utils import load_vectors
from obtain_words import write_file

words = load_words('en_words',0)
vectors = load_vectors('en.vec')
print(vectors['word'])

not_present = []
for word in words:
    if word not in vectors.keys():
        print(word)
        not_present.append(word)

write_file('not_present', not_present)
