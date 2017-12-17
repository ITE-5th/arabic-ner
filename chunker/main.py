import os

import nltk

from chunker.chunkers.consecutive_np_chunker import ConsecutiveNPChunker
from chunker.features_extractors.npchunk_extractor import NPChunkExtractor


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


def convert_from_boi_to_sent(sent):
    """
    convert from BOI to sentence

    :param sent: the sentence
    :return:
    """
    untagged_sent = ""
    # remove BOI tags and make sentence to tag
    for word, _ in sent:
        untagged_sent += " " + word

    return untagged_sent


def pos_tag_sent(sent, pos_tagger):
    """
    Tag sentence
    convert sentence from BOI-form to String-form and tag it using the provided tagger
    :param sent: boi sentence
    :param pos_tagger: tagger to tag the sentence
    :return:
    """
    untagged_sent = convert_from_boi_to_sent(sent)

    pos_tagged_sent = pos_tagger.tag(nltk.regexp_tokenize(untagged_sent, pattern=" ", gaps=True))

    result = []
    for i, (_, word) in enumerate(pos_tagged_sent):
        result.append((tuple(word.split('/')), sent[i][1]))

    return result


def tag(sents, pos_tagger):
    """
    tag each sentence in sentences using pos_tag_sent
    :param sents: list of sentences
    :param pos_tagger: the tagger
    :return:
    """
    pos_tagged_sents = []
    for sent in sents:
        pos_tagged_sents.append(pos_tag_sent(sent, pos_tagger))

    return pos_tagged_sents


def split_data(tagged_data, ratio: float = 0.8):
    length = len(tagged_data)
    train_sents = tagged_data[:int(length * ratio)]
    test_sents = tagged_data[len(train_sents):]

    sents = []
    for test_sent in test_sents:
        sents.extend([[(w, t, c) for ((w, t), c) in test_sent]])
        sents[-1] = nltk.chunk.conlltags2tree(sents[-1])

    test_sents = sents

    return train_sents, test_sents


def load_stanford_tagger():
    # support for stanford-tagger
    base_path = os.path.dirname(os.path.realpath(__file__)) + "/taggers/stanford-postagger/"
    path_to_model = base_path + "/models/arabic.tagger"
    path_to_jar = base_path + "stanford-postagger.jar"

    return nltk.StanfordPOSTagger(path_to_model, path_to_jar, encoding='utf-8')


if __name__ == "__main__":
    # read sentences
    sents = read("../data/ANERCorp", from_index=149000)

    pos_tagger = load_stanford_tagger()

    tagged_data = tag(sents, pos_tagger)
    train_sents, test_sents = split_data(tagged_data)

    chunker = ConsecutiveNPChunker(train_sents=train_sents, extractor=NPChunkExtractor())
    print(chunker.evaluate(test_sents))
