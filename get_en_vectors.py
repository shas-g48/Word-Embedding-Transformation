import utils

word_list = []
with open('word_transform/common.en.vocab') as f:
    for line in f:
        word_list.append(line[:-1])

word_embed = {}

#file_list = ['xaa','xab','xac','xad','xae','xaf']
#en = utils.load_vectors('xaa')

#list_dict = []
#for file in file_list:
"""    ind = utils.load_vectors(file)
    d = dict()
    for i in ind.keys():
        d[i] = 0
    list_dict.append(d)
    del ind
    del d
    del list_dict
    #del list_dict
"""
#del ind

ct = 0
#ld = 0
#nf = 0

en = utils.load_vectors('xaf')
for word in word_list:
    print(ct)
    ct+=1
    #for i in list_dict:
    #if word in i.keys() and ld == list_dict.index(i):
    #        word_embed[word] = en[word]
    if word in en.keys():
        word_embed[word] = en[word]

        #elif word in i.keys() and ld != list_dict.index(i):
        #    del en
        #    ld = list_dict.index(i)
        #    en = utils.load_vectors(file_list[ld])
        #    word_embed[word] = en[word]
del en
#del list_dict

with open('output.vec', 'a') as f:
    for i in word_embed.keys():
        f.write(i)
        for k in word_embed[i]:
            f.write(' ')
            f.write(str(k))
        f.write('\n')
