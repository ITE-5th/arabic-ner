from chunker.features_extractors.feature_extractor import FeatureExtractor


class NPChunkExtractor(FeatureExtractor):
    def extract(self, sentence, i, history):
        word, pos = sentence[i]
        if i == 0:
            prevword, prevpos = "<START>", "<START>"
        else:
            prevword, prevpos = sentence[i - 1]

        return {"pos": pos, "word": word, "prevpos": prevpos}
