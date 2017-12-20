import string

from nltk import SnowballStemmer

from chunker.features_extractors.feature_extractor import FeatureExtractor


class NPChunk(FeatureExtractor):
    def __init__(self, stemmer=SnowballStemmer("arabic")):
        self.stemmer = stemmer

    def extract(self, sentence, i, history):

        sentence, i, history = self.pad(sentence, i, history)

        word, pos = sentence[i]

        prevword, prevpos = sentence[i - 1]
        nextword, nextpos = sentence[i + 1]

        prevprevword, prevprevpos = sentence[i - 2]
        nextnextword, nextnextpos = sentence[i + 2]

        allascii = all([True for c in word if c in string.ascii_lowercase])
        previob = history[i - 1]
        prevpreviob = history[i - 1]

        return {"pos": pos,
                "pos[:2]": pos[:2],
                "all-ascii": allascii,

                "word": word,
                # 'lemma': self.stemmer.stem(word),
                "prev-iob": previob,

                "prev-word": prevword,
                "prev-pos": prevpos,
                # 'prev-lemma': self.stemmer.stem(prevword),

                "tags-since-dt": self.tag_since_dt(sentence, i),

                "next-word": nextword,
                "next-pos": nextpos,
                # 'next-lemma': self.stemmer.stem(nextword),

                "cur+next": "%s+%s" % (pos, nextpos),
                "cur+prev": "%s+%s" % (prevpos, pos),

                "next-next-word": nextnextword,
                "next-next-pos": nextnextpos,

                "prev-prev-word": prevprevword,
                "prev-prev-pos": prevprevpos,
                "prev-prev-iob": prevpreviob,
                }

    def tag_since_dt(self, sentence, i):
        tags = set()
        for word, pos in sentence[:i]:
            if pos in ["DT", "PUNC"]:
                tags = set()
            else:
                tags.add(pos)

        return '+'.join(sorted(tags))
