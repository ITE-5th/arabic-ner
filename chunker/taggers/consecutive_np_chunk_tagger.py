import os
import pickle

import nltk

from chunker.features_extractors.feature_extractor import FeatureExtractor
from chunker.features_extractors.npchunk import NPChunk


class ConsecutiveNPChunkTagger(nltk.ParserI):
    def __init__(self, train_sents, extractor: FeatureExtractor = NPChunk(), megam_path: str = None):
        train_set = []
        self.feature_extractor = extractor

        if megam_path is None:
            dir_path = os.path.dirname(os.path.realpath(__file__))
            nltk.config_megam(dir_path + "/megam_0.92/megam.opt")
        else:
            nltk.config_megam(megam_path)

        for tagged_sent in train_sents:
            untagged_sent = nltk.tag.untag(tagged_sent)
            history = []

            for i, (word, tag) in enumerate(tagged_sent):
                featureset = self.feature_extractor.extract(untagged_sent, i, history)
                train_set.append((featureset, tag))
                history.append(tag)

        self.classifier = nltk.MaxentClassifier.train(train_set)

    def tag(self, sentence):
        history = []

        for i, word in enumerate(sentence):
            featureset = self.feature_extractor.extract(sentence, i, history)
            tag = self.classifier.classify(featureset)
            history.append(tag)

        return zip(sentence, history)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.classifier, f)

    def load(self, path):
        with open(path, 'rb') as f:
            self.classifier = pickle.load(f)
