from chunker.features_extractors.feature_extractor import FeatureExtractor


class NPChunk(FeatureExtractor):
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

        return {"pos": pos,
                "word": word,
                "prevword": prevword,
                "prevpos": prevpos,
                "tags-since-dt": self.tag_since_dt(sentence, i),
                "nextword": nextword,
                "nextpos": nextpos,
                "cur+next": "%s+%s" % (pos, nextpos),
                "cur+prev": "%s+%s" % (prevpos, pos),
                }

    def tag_since_dt(self, sentence, i):
        tags = set()
        for word, pos in sentence[:i]:
            if pos in ["DT", "PUNC"]:
                tags = set()
            else:
                tags.add(pos)

        return '+'.join(sorted(tags))
