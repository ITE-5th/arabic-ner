


# def read(file_path: str):
#     with open(file_path) as f:
#         lines = f.readlines()
#     sentences, tags = to_sentences(lines)
#     result = []
#     for i in range(len(sentences)):
#         offset = 0
#         acc = ""
#         ner = []
#         for j in range(len(sentences[i])):
#             acc += sentences[i][j][0] + " "
#             old_offset = offset
#             offset += len(sentences[i][j][0]) + 1
#             if sentences[i][j][1] != "o":
#                 ner.append((old_offset, offset - 2, sentences[i][j][1]))
#         acc = acc.strip()
#         result.append((acc, {"entities": ner}))
#     return result, tags


if __name__ == '__main__':
    pass


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
