from nltk.corpus import conll2000

from chunker.chunkers.consecutive_np_chunker import ConsecutiveNPChunker
from chunker.features_extractors.npchunk_extractor import NPChunkExtractor

if __name__ == "__main__":
    test_sents = conll2000.chunked_sents('test.txt', chunk_types=['NP'])
    train_sents = conll2000.chunked_sents('train.txt', chunk_types=['NP'])

    chunker = ConsecutiveNPChunker(train_sents=train_sents, extractor=NPChunkExtractor())
    print(chunker.evaluate(test_sents))