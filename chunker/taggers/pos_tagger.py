import nltk
import os


class PosTagger(object):

    def __init__(self):
        super().__init__()
        self.load_stanford_tagger()

    def load_stanford_tagger(self, base_path=None):

        if base_path is None:
            base_path = os.path.dirname(os.path.realpath(__file__)) + "/stanford-postagger/"

        path_to_model = base_path + "/models/arabic.tagger"
        path_to_jar = base_path + "stanford-postagger.jar"

        self.tagger = nltk.StanfordPOSTagger(path_to_model, path_to_jar, encoding='utf-8')

    def pos_tag_sent(self, sent, boi_form=True):
        """
        Tag sentence
        convert sentence from BOI-form to String-form and tag it using the provided tagger
        :param sent: boi sentence
        :param pos_tagger: tagger to tag the sentence
        :return:
        """
        if boi_form:
            untagged_sent = self.convert_from_boi_to_sent(sent)
        else:
            untagged_sent = sent

        tokens = nltk.regexp_tokenize(untagged_sent, pattern=" ", gaps=True)
        pos_tagged_sent = self.tagger.tag(tokens)

        result = []
        for i, (_, word) in enumerate(pos_tagged_sent):
            r = word.split('/')
            if len(r) == 3:
                r = ['/', r[-1]]
            result.append((tuple(r), sent[i][1]))

        return result

    def tag(self, sents, boi_form=True):
        """
        tag each sentence in sentences using pos_tag_sent
        :param boi_form:
        :param sents: list of sentences
        :param pos_tagger: the tagger
        :return:
        """
        pos_tagged_sents = []
        for sent in sents:
            pos_tagged_sents.append(self.pos_tag_sent(sent, boi_form))

        return pos_tagged_sents

    @staticmethod
    def convert_from_boi_to_sent(sent):
        """
        convert from BOI to sentence

        :param sent: the sentence
        :return:
        """
        untagged_sent = ""
        # remove BOI tags and make sentence to tag
        for word, _ in sent:
            untagged_sent += " " + word

        return untagged_sent
