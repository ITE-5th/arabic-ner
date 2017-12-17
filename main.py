import random

import sys

from bilstmcrf.util import *
from PyQt5 import QtWidgets


# def to_sentences(lines):
#     lines = [line.strip() for line in lines]
#     puncs = set(".?!-")
#     result = []
#     acc = []
#     tags = {"EOS": EOS_INDEX, "SOS": SOS_INDEX}
#     for line in lines:
#         if line == "":
#             continue
#         line = line.split(" ")
#         if len(line) == 1 or line[0] in puncs:
#             if acc:
#                 result.append(acc)
#             acc = []
#             continue
#         else:
#             acc.append((line[0], line[1]))
#         line[1] = line[1].lower().strip()
#         if line[1] not in tags:
#             tags[line[1]] = len(tags)
#     return result, tags
#
#
# if __name__ == '__main__':
#     with open("data/ANERCorp") as f:
#         lines = f.readlines()
#     sentences, _ = to_sentences(lines)
#     sentences = [sent for sent in sentences if len(sent) <= 500]
#     random.shuffle(sentences)
#     train, dev, test = sentences[:int(len(sentences) * 7 // 10)], sentences[int(len(sentences) * 7 // 10):int(len(
#         sentences) * 8.5 // 10)], sentences[int(len(sentences) * 8.5 // 10):]
#     with open("data/train_data.txt", "w") as f:
#         f.write("-DOCSTART- -X- -X- -X- O\n\n")
#         for i in range(len(train)):
#             for j in range(len(train[i])):
#                 f.write("{} {}\n".format(train[i][j][0], train[i][j][1]))
#             f.write("\n")
#     with open("data/dev_data.txt", "w") as f:
#         f.write("-DOCSTART- -X- -X- -X- O\n\n")
#         for i in range(len(dev)):
#             for j in range(len(dev[i])):
#                 f.write("{} {}\n".format(dev[i][j][0], dev[i][j][1]))
#             f.write("\n")
#     with open("data/test_data.txt", "w") as f:
#         f.write("-DOCSTART- -X- -X- -X- O\n\n")
#         for i in range(len(test)):
#             for j in range(len(test[i])):
#                 f.write("{} {}\n".format(test[i][j][0], test[i][j][1]))
#             f.write("\n")
from ui import Ui_MainWindow

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    with open("qdarkstyle/style.qss") as f:
        app.setStyleSheet(f.read())
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setWindowTitle("Arabic Ner")
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

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
