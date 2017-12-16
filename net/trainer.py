from multiprocessing import cpu_count

import torch
from gensim.models import FastText
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from net.ner_dataset import NerDataset
from net.network import BiLSTMWithCRF


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
    batch_size, epochs = 256, 1000
    models_dir = "../models"
    dataset = NerDataset("../data/ANERCorp")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=cpu_count())
    vocabs = dataset.vocabs
    embedding_model = FastText.load("../data/wiki.ar.gensim")
    preinit_words_embedding = load_words_embed(embedding_model, vocabs)
    net = BiLSTMWithCRF(vocabs_size=len(dataset.vocabs),
                        tags_size=len(dataset.tags),
                        batch_size=batch_size,
                        preinit_word_embedding=preinit_words_embedding).cuda()
    optimizer = Adam(filter(lambda x: x.requires_grad, net.parameters()), lr=0.01, weight_decay=1e-4)
    batches = len(dataset.sentences) / batch_size
    loss = CrossEntropyLoss().cuda()
    print("Begin training")
    for epoch in range(epochs):
        epoch_loss, epoch_correct = 0, 0
        for batch, (vcs, tgs) in enumerate(dataloader, 0):
            vcs = Variable(vcs).cuda()
            tgs = Variable(tgs).cuda()
            optimizer.zero_grad()
            outs = net(vcs)
            _, first = outs.data.max(1)
            second = tgs.data
            correct = torch.eq(first, second).sum()
            epoch_correct += correct
            epoch_loss += loss.data[0]
            loss.bacward()
            optimizer.step()
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, epoch, models_dir)
        print('Epoch {} done, average loss: {}, average accuracy: {}%'.format(
            epoch + 1, epoch_loss / batches, epoch_correct * 100 / (batches * batch_size)))
