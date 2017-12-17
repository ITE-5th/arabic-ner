import torch
from torch.utils.data import Dataset

from net.util import PADDING_INDEX, EOS_INDEX, SOS_INDEX


class NerDataset(Dataset):
    def __init__(self, corpus_path: str, max_sentence_length: int = 120):
        self.max_sentence_length = max_sentence_length
        with open(corpus_path) as f:
            lines = f.readlines()
        self.sentences, self.tags = self.to_sentences(lines)
        self.vocabs = self.extract_vocabs()
        self.sentences = self.process_sentences()
        self.length = len(self.sentences)

    def __getitem__(self, index):
        return self.sentences[index]

    def __len__(self):
        return self.length

    def extract_vocabs(self):
        vocabs = {"PAD": PADDING_INDEX, "EOS": EOS_INDEX}
        sentences = self.sentences
        for i in range(len(sentences)):
            for j in range(len(sentences[i])):
                word = sentences[i][j][0]
                if word not in vocabs:
                    vocabs[word] = len(vocabs)
        return vocabs

    def process_sentences(self):
        result = []
        sentences, tags, vocabs = self.sentences, self.tags, self.vocabs
        for i in range(len(sentences)):
            vcs, tgs = [], []
            if len(sentences[i]) > self.max_sentence_length:
                continue
            for j in range(len(sentences[i])):
                vcs.append(vocabs[sentences[i][j][0]])
                tgs.append(tags[sentences[i][j][1]])
            padding = [PADDING_INDEX] * (self.max_sentence_length - len(vcs))
            old_length = len(vcs)
            vcs += [EOS_INDEX] + padding
            tgs += [EOS_INDEX] + padding
            result.append((torch.LongTensor(vcs), torch.LongTensor(tgs), old_length))
        result.sort(key=lambda x: x[2], reverse=True)
        result = [(x, y) for x, y, _ in result]
        return result

    def to_sentences(self, lines):
        lines = [line.strip() for line in lines]
        puncs = set(".?!")
        result = []
        acc = []
        tags = {"PAD": PADDING_INDEX, "EOS": EOS_INDEX, "SOS": SOS_INDEX}
        for line in lines:
            if line == "":
                continue
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
