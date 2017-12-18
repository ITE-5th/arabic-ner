from chunker.features_extractors.feature_extractor import FeatureExtractor


class People(FeatureExtractor):
    def __init__(self, people="../data/people"):
        with open(people) as f:
            lines = f.readlines()

        lines = map(str.strip, lines)
        self.people = set(lines)

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

        person = word in self.people
        next_person = nextword in self.people
        prev_person = prevword in self.people

        return {
            "person": person,
            "prevperson": prev_person,
            "nextperson": next_person,
            "person+next": "%s+%s" % (person, next_person),
            "person+prev": "%s+%s" % (prev_person, person),
        }
