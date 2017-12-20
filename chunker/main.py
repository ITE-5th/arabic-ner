import pickle

import nltk

from chunker.chunkers.consecutive_np_chunker import ConsecutiveNPChunker
from chunker.features_extractors.composer import Composer
from chunker.features_extractors.locations import Locations
from chunker.features_extractors.npchunk import NPChunk
from chunker.features_extractors.organizations import Organizations
from chunker.features_extractors.people import People
from chunker.features_extractors.stop_words import StopWords
from chunker.taggers.pos_tagger import PosTagger


def to_sentences(lines, delimiters: str = ".?!"):
    """
    Convert lines to sentences
    :param delimiters: to break lines into sentences
    :param lines: lines of a file
    :return: sentences
    """
    puncs = set(delimiters)
    result = []
    acc = []
    for line in lines:
        line = line.split(" ")
        line[0], line[1] = line[0].strip(), line[1].strip()
        if line[1] == "\u200f":
            continue
        if line[0] in puncs or line[0] == "":
            if acc:
                result.append(acc)
            acc = []
        else:
            acc.append((line[0], line[1]))

    return result


def read(file_path: str, delimiters: str = ".?!", from_index: int = None, to_index: int = None):
    """
    Read File and return its sentences
    :param file_path: File path
    :param delimiters: to break files into sentences
    :param from_index:
    :param to_index:
    :return:
    """
    with open(file_path) as f:
        lines = f.readlines()

    if to_index is not None:
        lines = lines[:to_index]

    if from_index is not None:
        lines = lines[from_index:]

    return to_sentences(lines, delimiters)


def split_data(tagged_data, ratio: float = 0.8):
    length = len(tagged_data)
    train_sents = tagged_data[:int(length * ratio)]
    test_sents = tagged_data[len(train_sents):]
    return train_sents, test_sents


def convert_to_tree(data):
    sents = []
    try:
        for datum in data:
            sents.extend([[(w, t, c) for ((w, t), c) in datum]])
            sents[-1] = nltk.chunk.conlltags2tree(sents[-1])
    except:
        print(datum)

    return sents


def load_tagged(train_path, test_path):
    return pickle.load(open(train_path, 'rb')), pickle.load(open(test_path, 'rb'))


if __name__ == "__main__":

    load_from_pickles = True
    save_tagged = True

    if load_from_pickles:
        train_sents, test_sents = load_tagged("./taggers/tagged_train", "./taggers/tagged_test")
    else:

        # read sentences
        # sents = read("../data/ANERCorp", from_index=2000, to_index=4000)
        sents = read("../data/modified_ANERCorp")
        # sents = read("../data/ANERCorp", from_index=2000)

        pos_tagger = PosTagger()

        tagged_data = pos_tagger.tag(sents)
        train_sents, test_sents = split_data(tagged_data)

        if save_tagged:
            pickle.dump(train_sents, open("./taggers/tagged_train", 'wb'))
            pickle.dump(test_sents, open("./taggers/tagged_test", 'wb'))
            exit(0)

    tree = convert_to_tree(test_sents)
    # output[1].draw()

    composer = Composer([NPChunk(), Locations(), People(), Organizations(), StopWords()])
    chunker = ConsecutiveNPChunker(train_sents=train_sents, extractor=composer)
    print(chunker.evaluate(tree))

    chunker.save("../pretrained/chunker")
