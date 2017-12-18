from chunker.features_extractors.feature_extractor import FeatureExtractor


class Composer(FeatureExtractor):
    def __init__(self, extractors):
        self.extractos = extractors

    def extract(self, sentence, i, history):
        features = {}

        for extractor in self.extractos:
            features.update(extractor.extract(sentence, i, history))

        return features
