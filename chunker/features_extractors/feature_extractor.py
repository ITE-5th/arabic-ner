import abc


class FeatureExtractor:

    @abc.abstractclassmethod
    def extract(self, sentence, i ,history):
        pass
