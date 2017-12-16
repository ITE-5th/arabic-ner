import random

import spacy


def to_sentences(lines):
    puncs = set(".?!")
    result = []
    acc = []
    tags = set()
    for line in lines:
        line = line.split(" ")
        line[0], line[1] = line[0].strip(), line[1].lower().strip()
        tags.add(line[1])
        if line[0] in puncs:
            result.append(acc)
            acc = []
        else:
            acc.append((line[0], line[1]))
    return result, tags


def read(file_path: str):
    with open(file_path) as f:
        lines = f.readlines()
    sentences, tags = to_sentences(lines)
    result = []
    for i in range(len(sentences)):
        offset = 0
        acc = ""
        ner = []
        for j in range(len(sentences[i])):
            acc += sentences[i][j][0] + " "
            old_offset = offset
            offset += len(sentences[i][j][0]) + 1
            if sentences[i][j][1] != "o":
                ner.append((old_offset, offset - 2, sentences[i][j][1]))
        acc = acc.strip()
        result.append((acc, {"entities": ner}))
    return result, tags


if __name__ == '__main__':
    train_data, tags = read("data/ANERCorp")
    # there is \u200f tag in the data -____-
    tags.remove("\u200f")
    tags = list(tags)
    nlp = spacy.blank("xx")
    optimizer = nlp.begin_training()
    print("Begin Training")
    for i in range(20):
        random.shuffle(train_data)
        losses = {}
        for text, annotations in train_data:
            nlp.update([text], [annotations], drop=0.5, sgd=optimizer, losses=losses)
        print("Epoch Finished")
        print("losses = {}".format(losses))
    nlp.to_disk('./model')
