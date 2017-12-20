import nltk

from chunker.features_extractors.feature_extractor import FeatureExtractor
from chunker.features_extractors.features import features
from chunker.taggers.consecutive_np_chunk_tagger import ConsecutiveNPChunkTagger


class ConsecutiveNPChunker(nltk.ChunkParserI):

    def __init__(self, train_sents, extractor: FeatureExtractor):
        # tagged_sents = [[((w, t), c) for (w, t, c) in nltk.chunk.tree2conlltags(sent)] for sent in train_sents]

        self.tagger = ConsecutiveNPChunkTagger(train_sents, extractor)
        # self.tagger = nltk.ClassifierBasedTagger(train=train_sents, feature_detector=features)

    def parse(self, sentence):
        tagged_sents = self.tagger.tag(sentence)
        conlltags = [(w, t, c) for ((w, t), c) in tagged_sents]
        return nltk.chunk.conlltags2tree(conlltags)

    def save(self, path):
        self.tagger.save(path)

    def load(self, path):
        self.tagger.load(path)