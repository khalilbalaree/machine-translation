
import torch
from torch import nn
import torch.utils.data
import nltk
torch.manual_seed(11)
device = torch.device("cuda")

"""
Created on Fri Feb 15 06:20:25 2019

@author: Administrator
"""
import random
import argparse
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print("It's running by:", device)


class Args:
    lr = 0.001
    dropout = 0.1
    hidden_size = 300
    MAX_LENGTH = 30
    data_path = '%s-%s.txt'
    input_lang = 'eng'
    output_lang = 'spa'
    n_iters = 150000
    print_every = 10000


args = Args()

SOS_token = 0
EOS_token = 1

MAX_LENGTH = 30

# ------------------------------------------------------------------------------
class EncoderRNN(nn.Module):
    def __init__(self, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, embedded, hidden):
        output = embedded.view(1, 1, -1)

        output, hidden = self.gru(output, hidden)

        return output, hidden, embedded

    def initHidden(self):  # initial Hidden-layer
        return torch.zeros(1, 1, self.hidden_size, device=device)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden


class AttnDecoderRNN(nn.Module):  # 'Neural Machine Translation by Jointly Learning to Align and Translate'
    def __init__(self, hidden_size, output_size, dropout_p=args.dropout, max_length=args.MAX_LENGTH):  #
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size,
                                      self.hidden_size)  # A simple lookup table that stores embeddings of a fixed dictionary and size.
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)  #
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

        self.tanh = nn.Tanh()

    def forward(self, embedded, hidden, encoder_outputs):
        '''
        Intra attention goes here read from decoder input
        '''

        # TODO inter attention from here
        # print('embedded[0]',embedded[0].shape)  # 1, 256
        # print( ' hidden[0]', hidden[0].shape)  # 1, 256
        attn_weights = F.softmax(
            F.tanh(
                self.attn(torch.cat((embedded[0], hidden[0]), 1))
            )
            , dim=1)
        # print(attn_weights.shape)
        # print(encoder_outputs.shape)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)
        # print('output', output.shape)  # 1, 1, 256
        # TODO till here

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = self.out(output[0])  # FC
        return output, hidden, attn_weights

    def initHidden(self):  # Initial hidden-layer
        return torch.zeros(1, 1, self.hidden_size, device=device)


teacher_forcing_ratio = 0.5

criterion2 = torch.nn.SmoothL1Loss()
def train(input_tensor, target_tensor, encoder, decoder, criterion, max_length=args.MAX_LENGTH):
    encoder_hidden = encoder.initHidden()  # init hidden layer

    input_tensor = torch.tensor(input_tensor, device=device, dtype=torch.float)
    target_tensor = torch.tensor(target_tensor, device=device, dtype=torch.float)
    # print(input_tensor.shape)
    # print(target_tensor.shape)
    input_length = input_tensor.shape[0]
    target_length = target_tensor.shape[0]

    # encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0


    '''
    #test lstmn
    #test lstmn
    #test lstmn
    '''

    lstmn_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)  # TODO 原来是 max length
    # print('encodered_ embeddeds: ', encoder_embeddeds)
    # print('encodered_ embeddeds shape: ', encoder_embeddeds.shape)
    counter = 0
    acc=0
    for ei in range(input_length):
        encoder_output, encoder_hidden, encoder_embedded = encoder(input_tensor[ei], encoder_hidden)
        lstmn_outputs[ei] = encoder_output#outputs
        counter += 1

    lstmn_hidden = encoder_hidden

    decoder_hidden = lstmn_hidden
    encoder_outputs = lstmn_outputs

    counter = 0
    scale_up_factor = 1.05
    decoder_input = target_tensor[0]
    for di in range(target_length):
        counter += 1
        outputs = decoder_input
        decoder_output, decoder_hidden, decoder_attention = decoder(outputs.unsqueeze(0).unsqueeze(0), decoder_hidden,encoder_outputs)
        # print(decoder_output.shape)

        # print(decoder_output.shape)
        # print(decoder_input.shape)
        # print(target_tensor[di].shape)
        y = torch.ones(1).to(device)
        y_neg = torch.tensor(-1).to(device)
        loss += (scale_up_factor**counter) * criterion(decoder_output, target_tensor[di].unsqueeze(0), y)

        acc += criterion(decoder_output, target_tensor[di].unsqueeze(0),y_neg)


        # print(random.random())
        if random.random() < teacher_forcing_ratio:  #teaching force applied
          decoder_input = decoder_output.squeeze()
        else:
          decoder_input = target_tensor[di]
        # print(1 + '1')

    # print(loss)
    loss.backward()

    return loss.item() / target_length, acc.item()/target_length


# --------------------print time consume-------------------------------------------------------------
import time
import math


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


# --------------------------------------------------------------------------------------------

def trainIters(encoder, decoder, n_iters, print_every=100, plot_every=10, learning_rate=args.lr):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    print_acc_total = 0
    plot_acc_total = 0
    valid_every = 30000

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)


    training_pairs = []
    eng = np.load('emb_en.npy',allow_pickle=True)
    chn = np.load('emb_zh.npy',allow_pickle=True)
    print('length of all training data: ', len(eng))

    counter = 0
    while counter < n_iters:
        randomidx = random.randint(1, 49999)
        len_eng = eng[randomidx].shape
        len_chn = chn[randomidx].shape
        if len_eng[0] != 0 and len_chn[0] != 0:
            # print(eng[randomidx].shape, chn[randomidx].shape, len_eng[0], len_chn[0])
            emb_pair = (eng[randomidx], chn[randomidx])
            counter += 1
        training_pairs.append(emb_pair)

    criterion = torch.nn.CosineEmbeddingLoss(reduction='none')  # nn.MSELoss()#nn.NLLLoss()
    for iter in range(1, n_iters + 1):

        training_pair = training_pairs[iter - 1]
        # print(training_pair)
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]
        encoder.train()
        decoder.train()

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        loss,acc = train(input_tensor, target_tensor, encoder,
                     decoder, criterion)


        encoder_optimizer.step()
        decoder_optimizer.step()
        #print(1+'1')
        print_loss_total += loss
        plot_loss_total += loss

        print_acc_total +=acc
        plot_acc_total +=acc

        if iter % print_every == 0 :  # print results
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print_acc_avg = print_acc_total / print_every
            print_acc_total = 0
            print('%s (%d %d%%) %.4f  acc: %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg,print_acc_avg))
        if iter % valid_every == 0:
          evaluate(encoder1, attn_decoder1, max_length=args.MAX_LENGTH)

        if iter % plot_every == 0:  # plot results
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)
    torch.save(decoder.state_dict(), './lstmn/decoder.pt') #Returns a dictionary containing a whole state of the module.
    torch.save(encoder.state_dict(), './lstmn/encoder.pt')


# ------------------------------------------------------------------------------
# plot the result

import matplotlib.pyplot as plt

plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


# ------------------------------------------------------------------------------
# evaluate the result
from scipy.spatial import distance
import numpy as np
import spacy

eng = np.load('emb_en.npy',allow_pickle=True)
chn = np.load('emb_zh.npy',allow_pickle=True)
print("validation embedding eng sample shape: ",eng.shape)
print("validation embedding chn sample shape: ", chn.shape)
#TODO show first sample:
nlpchn = spacy.load("zh_core_web_md")
nlpeng = spacy.load("en_core_web_md")
# https://stackoverflow.com/questions/54717449/mapping-word-vector-to-the-most-similar-closest-word-using-spacy
def find_most_similar(nlp, p):
    # Format the vocabulary for use in the distance function
    ids = [x for x in nlp.vocab.vectors.keys()]
    vectors = [nlp.vocab.vectors[x] for x in ids]
    vectors = np.array(vectors)

    # *** Find the closest word below ***
    closest_index = distance.cdist(p, vectors).argmin()
    word_id = ids[closest_index]
    return nlp.vocab[word_id].text
def evaluate(encoder, attn_decoder1, max_length=args.MAX_LENGTH):
    encoder.eval()
    attn_decoder1.eval()
    encoder_hidden = encoder.initHidden()
    #TODO load all sample:
    print("load all samples: ... ")
    short_sentence = [11,15,21,65,76,
              77,82,93,95,105,
              108,115,119,147,177,
              206,208,224,243,248]
    
    long_sentence = [10,11,26,80,99,
              111,121,124,152,166,
              179,199,252,286,340,
              404,417,446,485,545]
    for which in [short_sentence,long_sentence]:
      total_acc = 0
      total_blue = 0
      for i in which:
        first_sentence_eng = eng[i-1]
        first_sentence_chn = chn[i-1]
        #print(first_sentence_eng.shape)
        #print(first_sentence_chn.shape)
        
        hypothesis = []
        reference = []


        for i in range(len(first_sentence_chn)):
            p = first_sentence_chn[i]
            p = np.array([p])
            #print(p.shape)
            output = find_most_similar(nlpchn, p)
            #print(output)
            reference.append(output)

        with torch.no_grad():
            encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
            lstmn_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)  # TODO 原来是 max length
            counter = 0
            for ei in range(len(first_sentence_eng)):
              encoder_output, encoder_hidden, encoder_embedded = encoder(torch.tensor(first_sentence_eng[ei],device=device,dtype=torch.float32), encoder_hidden,)
              lstmn_outputs[ei] = encoder_output#outputs
              counter += 1

            lstmn_hidden = encoder_hidden


            decoder_hidden = lstmn_hidden
            encoder_outputs = lstmn_outputs
            #print("start decode")
            decoder_input = torch.tensor(first_sentence_chn[0],device=device,dtype=torch.float32)  # SOS

            counter = 0
            acc = 0
            lossfunc = torch.nn.CosineEmbeddingLoss(reduction='none')
            for di in range(len(first_sentence_eng)):
                counter += 1
                outputs = decoder_input
                decoder_output, decoder_hidden, decoder_attention = attn_decoder1(outputs.unsqueeze(0).unsqueeze(0), decoder_hidden,encoder_outputs)
                decoder_input = decoder_output.squeeze()
                output = find_most_similar(nlpchn, decoder_output.detach().cpu().numpy())
                #print(output)
                hypothesis.append(output)
                if di >= len(first_sentence_chn):
                    break
                loss = lossfunc(decoder_output.to(device), torch.tensor(first_sentence_chn[di],device=device).unsqueeze(0), torch.tensor(-1,device=device))
                #print(loss.item())
                acc += loss.item()
            #print("@@@@@@@@@@@@@")
            acc /= len(first_sentence_chn)
            total_acc+=acc
            
            #print(hypothesis,reference)
            BLUEscore = nltk.translate.bleu_score.sentence_bleu([reference],hypothesis,weights = (1,0,0,0))
            
            total_blue += BLUEscore
      print(hypothesis)
      print(reference)
      print("Cosine_similarity_acc_20:", total_acc/20)
      print("BlUE_20:", total_blue*100/20)



# ------------------------------------------------------------------------------

# hidden_size = 256
max_len = 30
dim = args.hidden_size
batch_size = 1
encoder1 = EncoderRNN(args.hidden_size).to(device)  # Enconder
attn_decoder1 = AttnDecoderRNN(args.hidden_size, dim, dropout_p=args.dropout).to(
    device)  # original is output_lang.n_words

torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
random.seed(1)
np.random.seed(1)
torch.backends.cudnn.deterministic = True

trainIters(encoder1, attn_decoder1, args.n_iters, print_every=args.print_every)  # hyperparameters
#evaluate(encoder1, attn_decoder1,lstmn, lstmn2, max_length=args.MAX_LENGTH)