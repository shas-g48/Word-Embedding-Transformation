import io
import numpy as np
import tensorflow as tf
import os

def load_vectors(filename):
    """
    Creates vectors for lang in a dict
    of form {word:embedding}
    """

    filein = io.open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore')

    data = {}
    for line in filein:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = list(map(float, tokens[1:]))
    return data

def load_data(filename, en, es):
    """
    Creates the dataset
    of form (input embedding, output embedding)

    en, es: locations of premade embeddings
    """

    filein = io.open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore')
    # get input and output strings
    load_input = []
    load_output = []
    for line in filein:
        tokens = line.rstrip().split(',')
        load_input.append(tokens[0])
        load_output.append(tokens[1])

    # get input and output vectors
    input_vec = []
    output_vec = []
    for i in load_input:
        input_vec.append(es[i])
    for i in load_output:
        output_vec.append(en[i])

    return (np.asarray(input_vec, dtype=np.float32), np.asarray(output_vec, dtype=np.float32))

def get_data(filename, batch_size, en, es):
    """
    Creates a tensorflow dataset
    from tuples

    en, es: locations of premade embeddings
    """

    dataset = load_data(filename, en, es)

    tf_dataset = tf.data.Dataset.from_tensor_slices(dataset)
    tf_dataset = tf_dataset.shuffle(100)
    tf_dataset = tf_dataset.batch(batch_size)

    return tf_dataset

def write_translation(filename, batch_size, top1_words, top51_words, top52_words, top53_words, top54_words, top55_words):
    """
    Writes the translated words in form
    word1_trans (top1)
    word1_trans1 .. word1_trans5 (top5)
    .. for all words
    """
    with open(filename, 'w') as f:
        for i in range(batch_size):
            f.write(top1_words[i])
            f.write('\n')
            f.write(top51_words[i])
            f.write(' ')
            f.write(top52_words[i])
            f.write(' ')
            f.write(top53_words[i])
            f.write(' ')
            f.write(top54_words[i])
            f.write(' ')
            f.write(top55_words[i])
            f.write('\n')

def safe_mkdir(path):
    try:
        os.mkdir(path)
    except OSError:
        pass
