from chunker.features_extractors.feature_extractor import FeatureExtractor


class People(FeatureExtractor):
    def __init__(self, people="../data/people"):
        with open(people) as f:
            lines = f.readlines()

        lines = map(str.strip, lines)
        self.people = set(lines)

    def extract(self, sentence, i, history):
        word, pos = sentence[i]

        return {
            "person": word in self.people
        }
