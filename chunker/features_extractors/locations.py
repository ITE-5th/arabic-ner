from chunker.features_extractors.feature_extractor import FeatureExtractor


class Locations(FeatureExtractor):
    def __init__(self, locations="../data/locations"):
        with open(locations) as f:
            lines = f.readlines()

        lines = map(str.strip, lines)

        self.locations = set(lines)

    def extract(self, sentence, i, history):
        word, pos = sentence[i]
        return {
            "loc": word in self.locations
        }
