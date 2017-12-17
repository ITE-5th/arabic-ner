from net.util import *


def to_sentences(lines):
    lines = [line.strip() for line in lines]
    puncs = set(".?!")
    result = []
    acc = []
    tags = {"PAD": PADDING_INDEX, "EOS": EOS_INDEX, "SOS": SOS_INDEX}
    for line in lines:
        line = line.split(" ")
        line[0], line[1] = line[0].strip(), line[1].lower().strip()
        if line[1] == "\u200f":
            continue
        if line[0] in puncs or line[0] == "":
            if acc:
                result.append(acc)
            acc = []
        else:
            acc.append((line[0], line[1]))
        if line[1] not in tags:
            tags[line[1]] = len(tags)
    return result, tags


if __name__ == '__main__':
    with open("data/ANERCorp") as f:
        lines = f.readlines()
    sentences, _ = to_sentences(lines)
    vcs, tgs = [], []
    for i in range(len(sentences)):
        temp1, temp2 = [], []
        for j in range(len(sentences[i])):
            temp1.append(sentences[i][j][0])
            temp2.append(sentences[i][j][1])
        vcs.append(temp1)
        tgs.append(temp2)
    with open("data/sents.txt", "w") as f:
        f.write("\n".join([" ".join(sent) for sent in vcs]))
    with open("data/tags.txt", "w") as f:
        f.write("\n".join([" ".join(tag) for tag in tgs]))

# def spacy_train():
#     iterations = 20
#     nlp = spacy.blank("xx")
#     print("Begin Training")
#     other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
#     with nlp.disable_pipes(*other_pipes):  # only train NER
#         optimizer = nlp.begin_training()
#         for itn in range(iterations):
#             random.shuffle(train_data)
#             losses = {}
#             for text, annotations in train_data:
#                 nlp.update(
#                     [text],  # batch of texts
#                     [annotations],  # batch of annotations
#                     drop=0.5,  # dropout - make it harder to memorise data
#                     sgd=optimizer,  # callable to update weights
#                     losses=losses)
#             print(losses)
#             print("epoch finished")
#
#     nlp.to_disk('./model')
#     nlp = spacy.load("./model")
#     test = "أعلن أحمد من لبنان."
#     doc = nlp(test)
#     for ent in doc.ents:
#         print(ent.label_, ent.text)
