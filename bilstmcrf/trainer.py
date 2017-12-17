from gensim.models import FastText
from torch.cuda import manual_seed
from torch.optim import Adam

from bilstmcrf.network import *
from bilstmcrf.util import *


def load_data(root_dir: str):
    tags = {SOS: SOS_INDEX, EOS: EOS_INDEX}
    vocabs = {}
    with open("{}/sents.txt".format(root_dir)) as f1, open("{}/tags.txt".format(root_dir)) as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()
    lines1 = [line.strip() for line in lines1]
    lines2 = [line.strip() for line in lines2]
    result = []
    for f, s in zip(lines1, lines2):
        temp = s.split(" ")
        temp2 = f.split(" ")
        for t in temp:
            t = t.replace("\n", "").strip()
            if t not in tags:
                tags[t] = len(tags)
        for t in temp2:
            t = t.replace("\n", "").strip()
            if t not in vocabs:
                vocabs[t] = len(vocabs)
        result.append((temp2, temp))
    return result, vocabs, tags


def save_checkpoint(state, epoch: int, directory: str = '../models'):
    torch.save(state, "{}/epoch-{}-checkpoint.pth.tar".format(directory, epoch + 1))


def load_words_embed(pretrained_embed_model, vocab) -> torch.FloatTensor:
    l = len(vocab)
    embeds = torch.randn(l, 300)
    for i in range(0, l):
        try:
            embeds[i, :] = torch.from_numpy(pretrained_embed_model[vocab[i]]).view(1, 300)
        except:
            embeds[i, :] = torch.randn(1, 300)
    return embeds


if __name__ == '__main__':
    manual_seed(100)
    epochs = 1000
    data, vocab, tags = load_data("../data")
    fasttext = FastText.load("../data/wiki.ar.gensim")
    embeds = load_words_embed(fasttext, vocab)
    net = BiLSTMWithCRF(len(vocab), tags, 300, 8, preinit_embedding=embeds)
    # bilstmcrf = bilstmcrf.cuda()
    opt = Adam(net.parameters(), lr=0.01, weight_decay=1e-3)
    print("Begin training")
    for epoch in range(epochs):
        for sentence, tgs in data:
            opt.zero_grad()
            sentence_in = prepare_sequence(sentence, vocab)
            targets = torch.LongTensor([tags[t] for t in tgs])
            neg_log_likelihood = net.neg_log_likelihood(sentence_in, targets)
            neg_log_likelihood.backward()
            opt.step()
        print("Epoch finished")
