import abc


class FeatureExtractor:

    @abc.abstractclassmethod
    def extract(self, sentence, i, history):
        pass

    def pad(self, sentence, i, history):
        sentence = [('<START2>', '<START2>'), ('<START1>', '<START1>')] + list(sentence) + [('<END1>', '<END1>'),
                                                                                            ('<END2>', '<END2>')]
        history = ['<START2>', '<START1>'] + list(history)

        return sentence, i + 2, history
