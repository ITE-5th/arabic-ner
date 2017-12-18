from chunker.features_extractors.feature_extractor import FeatureExtractor


class NPChunk(FeatureExtractor):
    def extract(self, sentence, i, history):
        try:
            word, pos = sentence[i]

        except:
            print("Sentence %s, Position: %i" % (sentence[i], i))
            print("Word %s, POS: %s" % (word, pos))

        try:
            if i == 0:
                prevword, prevpos = "<START>", "<START>"
            else:
                prevword, prevpos = sentence[i - 1]
        except:
            print("Prev word: %s, Position: %i, Pos: %s" % (prevword, i - 1, prevpos))

        try:
            if i == len(sentence) - 1:
                nextword, nextpos = "<END>", "<END>"
            else:
                nextword, nextpos = sentence[i + 1]
        except:
            print("Next word %s, Position: %i, Pos: %s" % (nextword, i, nextpos))

        return {"pos": pos,
                "word": word,
                # "prevword": prevword,
                "prevpos": prevpos,
                # "tags-since-dt": self.tag_since_dt(sentence, i),
                # "nextword": nextword,
                "nextpos": nextpos
                }

    def tag_since_dt(self, sentence, i):
        tags = set()
        for word, pos in sentence[:i]:
            if pos in ["DT", "PUNC"]:
                tags = set()
            else:
                tags.add(pos)

        return '+'.join(sorted(tags))
