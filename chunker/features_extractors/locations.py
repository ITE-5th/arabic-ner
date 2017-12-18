from chunker.features_extractors.feature_extractor import FeatureExtractor


class Locations(FeatureExtractor):
    def __init__(self, locations="../data/locations"):

        with open(locations) as f:
            lines = f.readlines()

        lines = map(str.strip, lines)

        self.locations = set(lines)

    def extract(self, sentence, i, history):
        word, pos = sentence[i]
        if i == 0:
            prevword, prevpos = "<START>", "<START>"
        else:
            prevword, prevpos = sentence[i - 1]

        if i == len(sentence) - 1:
            nextword, nextpos = "<END>", "<END>"
        else:
            nextword, nextpos = sentence[i + 1]

        return {
            "loc": word in self.locations
        }
