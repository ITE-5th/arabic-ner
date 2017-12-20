from chunker.features_extractors.feature_extractor import FeatureExtractor


class StopWords(FeatureExtractor):
    def __init__(self, stop_words="../data/stop_words", stop_words_tags="../data/stop_words_tags"):
        with open(stop_words) as f:
            lines = map(str.strip, f.readlines())
        self.stop_words = set(lines)

        with open(stop_words_tags) as f:
            lines = map(str.strip, f.readlines())
        self.stop_words_tags = set(lines)

    def extract(self, sentence, i, history):
        word, pos = sentence[i]
        return {
            "stop_word": word in self.stop_words,
            "stop_word_pos": pos in self.stop_words_tags
        }
