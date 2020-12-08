# -*- coding: utf-8 -*-


import torch
from torch import nn
import torch.utils.data
import nltk
torch.manual_seed(11)
device = torch.device("cuda")

class LSTMN(nn.Module):

    ### PARAMETERS
    # dim is number of hidden units
    # max_len is the maximum length of the input
    # batch_size is length of minibatch
    # output_size is dim of final output
    # init_hidden (optional) is the initial hidden state
    # init_cell (optional) is the initial cell state
    ### RETURNS
    # outputs, size (batch_size, output_size)
    # hidden_states, size (batch_size, max_length, dim)
    ###

    def __init__(self, dim, max_len, batch_size, output_size, hidden_size, encoder_or_decoder ,init_hidden=0, init_cell=0 ):
        super(LSTMN, self).__init__()
        self.which = encoder_or_decoder  #0 = encoder // 1 = decoder
        self.embedding = nn.Embedding(hidden_size,
                                      hidden_size).to(device)  #TODO 暂时这么写
        self.max_len = max_len
        self.dim = dim
        self.output_size = output_size
        self.batch_size = batch_size
        '''
        embedding from encoder
        '''
        self.hidden_size = hidden_size

        # Weight matrix for each of the 4 gates
        self.W = torch.randn(4 * dim, dim + dim, device=device,
                            dtype=torch.float)  # modifed....  TODO should be max_len + dim

        # Bias matrix
        self.b = torch.randn(4 * dim, device=device, dtype=torch.float)

        # Set forget bias to 1 per http://proceedings.mlr.press/v37/jozefowicz15.pdf
        self.b[:dim] = 1

        # initial hidden state
        self.h = torch.randn(dim, device=device, dtype=torch.float) if init_hidden == 0 else init_hidden

        # initial cell state
        self.c = torch.randn(dim, device=device, dtype=torch.float) if init_cell == 0 else init_cell

        # Attention weights
        self.W_h = torch.zeros(10, dim, device=device, dtype=torch.float)  # todo init weight in zeros
        self.W_x2 = torch.zeros(10, dim, device=device, dtype=torch.float)  # todo init weight in zeros
        self.W_x = torch.randn(10, dim, device=device, dtype=torch.float)  # todo should be max_len
        self.W_ht = torch.randn(10, dim, device=device, dtype=torch.float)

        self.v = torch.randn(10, device=device, dtype=torch.float)

        self.ht = torch.randn(dim, device=device, dtype=torch.float)
        self.ct = torch.randn(dim, device=device, dtype=torch.float)
        # for deep fusion weight calculated with encoder_ct_tape and encoder_ht_tape
        self.W_y = torch.zeros(10, dim, device=device, dtype=torch.float)  # todo init weight in zeros
        self.W_yt = torch.randn(10, dim, device=device, dtype=torch.float)
        self.u = torch.randn(10, device=device, dtype=torch.float)
        self.yt = torch.randn(dim, device=device, dtype=torch.float)
        self.at = torch.randn(dim, device=device, dtype=torch.float)
        self.W_r = torch.randn(dim, dim + dim, device=device,
                             dtype=torch.float)
        self.b_r = torch.randn(dim, device=device, dtype=torch.float)
        self.b_r[:dim] = 1
        # Projection layer
        self.repetiveProjection = torch.nn.Linear(dim * 4, dim * 4).to(device)
   
        self.reshape_bef = torch.nn.Linear(dim * 4, 1600).to(device)
        self.reshape_pos = torch.nn.Linear(1600, dim * 4).to(device)


        self.repetiveProjection3 = torch.nn.Linear(dim * 4, dim * 4).to(device)
        self.inter_projection_groups2_rep = nn.Sequential(
            torch.nn.Conv2d(1, 32, 3, 1, 1).to(device),
            nn.ReLU().to(device),
            nn.Dropout(0.1).to(device),
            nn.BatchNorm2d(32).to(device),
            torch.nn.Conv2d(32, 64, 3, 1, 1).to(device),
            nn.ReLU().to(device),
            nn.Dropout(0.1).to(device),
            nn.BatchNorm2d(64).to(device),
            torch.nn.Conv2d(64, 64, 3, 1, 1).to(device),
            nn.ReLU().to(device),
            nn.Dropout(0.1).to(device),
            nn.BatchNorm2d(64).to(device),
            torch.nn.Conv2d(64, 64, 3, 1, 1).to(device),
            nn.ReLU().to(device),
            nn.Dropout(0.1).to(device),
            nn.BatchNorm2d(64).to(device),
            torch.nn.Conv2d(64, 32, 3, 1, 1).to(device),
            nn.ReLU().to(device),
            nn.Dropout(0.1).to(device),
            nn.BatchNorm2d(32).to(device),
            torch.nn.Conv2d(32, 1, 3, 1, 1).to(device),
            nn.ReLU().to(device),
            nn.Dropout(0.1).to(device),
            nn.BatchNorm2d(1).to(device),
        )
        self.pos_cat = nn.Sequential(
                torch.nn.Conv2d(2, 1, 3, 1, 1).to(device),
                nn.BatchNorm2d(1).to(device)
        ) 
    # x is batch of input column vectors shape (batch_size, max_len)
    def forward(self, x, hidden_states, past_ht, past_x, past_c, counter, encoder_ct_tape=0,encoder_ht_tape = 0,past_yt = 0):
        if self.which == 0:
          x_t = x.to('cuda').squeeze(0).squeeze(0)  # 【300】
          attention_vector = []

          # Iterate through past hidden states and calculate attention vector
          for k, h in enumerate(hidden_states):
              a_t = self.v @ (nn.Tanh()(self.W_h @ h + self.W_x @ x_t +self.W_ht @ past_ht[
                  k]))  # todo add w_x @ x_t term
              attention_vector.append(a_t)

          attention_vector = torch.Tensor(attention_vector)
          attention_softmax = torch.nn.Softmax()(attention_vector)  # 同 torch.nn.functional.softmax(attention_vector)
          # print('hidden state: ', hidden_states) #accumulative ==> iteration x 64
          # print('attention vector; ', attention_vector)#accumulative
          # print('atten vector softmax: ',attention_softmax)#accumulative
          if len(hidden_states) > 0:
              ht = 0
              ct = 0
              for k, s in enumerate(attention_softmax):
                  ht += s * hidden_states[k]
                  ct += s * past_c[k]

              self.ht = torch.tensor(ht, device='cuda')
              self.ct = torch.tensor(ct, device='cuda')

          # print('self.ht: ',self.ht.shape )  # 64 x 1
          # print('self.ct:', self.ct) # 64 x 1

          concat_input = torch.cat((self.ht, x_t.to('cuda')), 0).view(-1, 1)  # 74 x 1

          whx = self.W.mm(concat_input).view(-1) + self.b
          '''inter conv block'''
          
          whx = self.reshape_bef(whx)
          whx_pre = whx.view(1, 1, 40, 40)
          whx = self.inter_projection_groups2_rep(whx_pre)  # TODO linear here ， 目前最好效果！！！！！！
          whx = torch.cat([whx_pre,whx],1)  #residual block
          whx = self.pos_cat(whx)
          whx = whx.view(-1)
          whx = self.reshape_pos(whx)
          whx = whx.view(1200)
          

          f_t = nn.Sigmoid()(whx[:self.dim])
          o_t = nn.Sigmoid()(whx[self.dim:self.dim * 2])
          i_t = nn.Sigmoid()(whx[self.dim * 2:self.dim * 3])
          ch_t = nn.Tanh()(whx[self.dim * 3:self.dim * 4])
          self.c = f_t * self.ct + i_t * ch_t  # + r_t * self.ht

          self.h = o_t * (nn.Tanh()(self.c))

          past_x.append(x_t)
          past_ht.append(self.ht)
          hidden_states.append(self.h)
          past_c.append(self.c)

          # print('self.h: ',self.h.shape)    # 64 x 1
          # print('self.c: ', self.c.shape)   #64 x 1
          # print('past_ht: ', past_ht)
          return hidden_states[-1], hidden_states, past_ht, past_x, past_c, counter
        elif self.which == 1:
          
          '''
          embedding batch input
          '''

          x_t = x.to('cuda').squeeze(0).squeeze(0)  # 【300】
          attention_vector = []

          # Iterate through past hidden states and calculate attention vector
          for k, h in enumerate(hidden_states):
              a_t = self.v @ (nn.Tanh()(self.W_h @ h + self.W_x @ x_t + self.W_ht @ past_ht[
                  k]))  # todo add w_x @ x_t term
              attention_vector.append(a_t)

          attention_vector = torch.Tensor(attention_vector)
          attention_softmax = torch.nn.Softmax()(attention_vector)  # 同 torch.nn.functional.softmax(attention_vector)
          # print('hidden state: ', hidden_states) #accumulative ==> iteration x 64
          # print('attention vector; ', attention_vector)#accumulative
          # print('atten vector softmax: ',attention_softmax)#accumulative
          if len(hidden_states) > 0:
              ht = 0
              ct = 0
              for k, s in enumerate(attention_softmax):
                  ht += s * hidden_states[k]
                  ct += s * past_c[k]

              self.ht = torch.tensor(ht, device='cuda')
              self.ct = torch.tensor(ct, device='cuda')

          # print('self.ht: ',self.ht.shape )  # 64 x 1
          # print('self.ct:', self.ct) # 64 x 1

          concat_input = torch.cat((self.ht, x_t.to('cuda')), 0).view(-1, 1)  # 74 x 
          whx = self.W.mm(concat_input).view(-1) + self.b
          '''add inter attention here:'''
          attention_vector_deep = []
          # Iterate through past hidden states and calculate attention vector
          #for i in range(counter):
          for k_deep, h_deep in enumerate(encoder_ht_tape):
            #a_t_deep = self.v @ (nn.Tanh()(self.W_y @ encoder_ht_tape[i] + self.W_x2 @ x_t + self.W_yt @ past_yt[i]))
            if k_deep >= len(past_yt):
              a = 0
            else:
              a = self.W_yt @ past_yt[k_deep]
            a_t_deep = self.v @ (nn.Tanh()(self.W_y @ h_deep + self.W_x2 @ x_t + a))  # todo add w_x @ x_t term
            attention_vector_deep.append(a_t_deep)
          attention_vector_deep = torch.Tensor(attention_vector_deep)
          attention_softmax_deep = torch.nn.Softmax()(attention_vector_deep)
          if len(encoder_ht_tape) > 0:  #todo should be encoder_ht_tape
              yt = 0
              at = 0
              for k, s in enumerate(attention_softmax_deep):
                  yt += s * encoder_ht_tape[k]
                  at += s * encoder_ct_tape[k]

              self.yt = torch.tensor(yt, device='cuda')
              self.at = torch.tensor(at, device='cuda')
          concat_input_deep = torch.cat((self.yt, x_t.to('cuda')), 0).view(-1, 1)
          past_yt.append(self.yt)


          #r_t = nn.Sigmoid() (self.W_r.mm(concat_input_deep).view(-1) + self.b_r)
          r_t_whx = self.W_r.mm(concat_input_deep).view(-1) + self.b_r

          r_t = nn.Sigmoid() (r_t_whx)
          #whx = torch.cat((whx, self.W_r.mm(concat_input_deep).view(-1) + self.b_r),0)
          #print(whx.shape)  #1500
          #print("concat_input_deep: ",concat_input_deep.shape)  # 600, 1

          '''inter conv block'''
          whx = self.reshape_bef(whx)
          whx_pre = whx.view(1, 1, 40, 40)
          whx = self.inter_projection_groups2_rep(whx_pre)  # TODO linear here ， 目前最好效果！！！！！！
          whx = torch.cat([whx_pre,whx],1)  #residual block
          whx = self.pos_cat(whx)
          whx = whx.view(-1)
          whx = self.reshape_pos(whx)
          whx = whx.view(1200)
          

          f_t = nn.Sigmoid()(whx[:self.dim])
          o_t = nn.Sigmoid()(whx[self.dim:self.dim * 2])
          i_t = nn.Sigmoid()(whx[self.dim * 2:self.dim * 3])
          ch_t = nn.Tanh()(whx[self.dim * 3:self.dim * 4])
          #r_t = nn.Sigmoid()(whx[self.dim * 4:])
          self.c = r_t * self.at + f_t * self.ct + i_t * ch_t  # + r_t * self.ht

          self.h = o_t * (nn.Tanh()(self.c))

          past_x.append(x_t)
          past_ht.append(self.ht)
          hidden_states.append(self.h)
          past_c.append(self.c)
          return hidden_states[-1], hidden_states, past_ht, past_x, past_c, past_yt, counter


import random
import argparse
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print("It's running by:", device)
# device = torch.device("cpu")




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
def train(input_tensor, target_tensor, decoder, lstmn, lstmn2, criterion, max_length=args.MAX_LENGTH):

    input_tensor = torch.tensor(input_tensor, device=device, dtype=torch.float)
    target_tensor = torch.tensor(target_tensor, device=device, dtype=torch.float)
    # print(input_tensor.shape)
    # print(target_tensor.shape)
    input_length = input_tensor.shape[0]
    target_length = target_tensor.shape[0]

    # encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0
    loss1= 0
    loss2 = 0
    # print('input tensor shape: ', input_tensor.shape)
    # print('input tensor: ', input_tensor)




    '''
    #test lstmn
    #test lstmn
    #test lstmn
    '''
    lstmn_outputs = torch.zeros(max_length, args.hidden_size, device=device)  # TODO 原来是 max length

    hidden_states = []
    past_ht = []
    past_x = []
    past_c = []
    counter = 0
    acc=0
    for ei in range(input_length):
        outputs, hidden_states, past_ht, past_x, past_c, counter = lstmn(input_tensor[ei], hidden_states, past_ht,past_x, past_c, counter, 0)
        lstmn_outputs[ei] = outputs#outputs
        counter += 1

    lstmn_hidden = outputs.unsqueeze(0).unsqueeze(0) 
    # print('output hidden: ', hidden_states[0])

    # print('lstmn_outputs: ', lstmn_outputs)
    # print('lstmn_hidden: ',lstmn_hidden.shape)
    # print(1+'1')
    # TODO 加了 很多 linear 后 效果提升至 7.89 - > 6.68
    decoder_hidden = lstmn_hidden
    encoder_outputs = lstmn_outputs

    hidden_states_2 = []
    past_ht_2 = []
    past_x_2 = []
    past_c_2 = []
    past_yt = []
    counter = 0
    decoder_input = target_tensor[0]
    for di in range(target_length):
        outputs, hidden_states_2, past_ht_2, past_x_2, past_c_2, past_yt, counter = lstmn2(decoder_input, hidden_states_2, past_ht_2, past_x_2, past_c_2, counter, past_c ,past_ht, past_yt)
        counter += 1
        decoder_output, decoder_hidden, decoder_attention = decoder(outputs.unsqueeze(0).unsqueeze(0), decoder_hidden,encoder_outputs)
        y = torch.ones(1).to(device)
        y_neg = torch.tensor(-1).to(device)
        loss += criterion(decoder_output, target_tensor[di].unsqueeze(0), y)


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

def trainIters( decoder, lstmn, lstmn2, n_iters, print_every=100, plot_every=10, learning_rate=args.lr):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    print_acc_total = 0
    plot_acc_total = 0
    valid_every = 30000

    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    # TODO 目前最好效果 SGD = 0.01， Adam = 0.001
    lstmn_optimizer = optim.SGD(lstmn.parameters(),lr=0.01)  # list(lstmn.parameters()) + list(decoder.parameters())
    lstmn_optimizer2 = optim.SGD(lstmn2.parameters(), lr=0.01)


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
        '''
        if 100000 > iter > 50000:
            for param_group in lstmn_optimizer.param_groups:
                param_group['lr'] = 0.001
        elif iter >= 100000:
            for param_group in lstmn_optimizer.param_groups:
                param_group['lr'] = 0.0001
        '''
        training_pair = training_pairs[iter - 1]
        # print(training_pair)
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]
        decoder.train()
        lstmn.train()
        lstmn2.train()
        lstmn_optimizer.zero_grad()
        lstmn_optimizer2.zero_grad()
        decoder_optimizer.zero_grad()

        loss,acc = train(input_tensor, target_tensor,
                     decoder, lstmn, lstmn2, criterion)

        lstmn_optimizer.step()
        lstmn_optimizer2.step()
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
          evaluate(attn_decoder1,lstmn, lstmn2, max_length=args.MAX_LENGTH)

        if iter % plot_every == 0:  # plot results
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)
    torch.save(decoder.state_dict(), './lstmn/decoder_final.pt') #Returns a dictionary containing a whole state of the module.
    torch.save(lstmn.state_dict(), './lstmn/lstmn_final.pt')
    torch.save(lstmn2.state_dict(),'./lstmn2_final.pt')


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
def evaluate(attn_decoder1, lstmn, lstmn2, max_length=args.MAX_LENGTH):
    attn_decoder1.eval()
    lstmn.eval()
    lstmn2.eval()
    
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
            lstmn_outputs = torch.zeros(max_length, args.hidden_size, device=device)  # TODO 原来是 max length
            hidden_states = []
            past_ht = []
            past_x = []
            past_c = []
            counter = 0
            for ei in range(len(first_sentence_eng)):
                outputs, hidden_states, past_ht, past_x, past_c, counter = lstmn(torch.tensor(first_sentence_eng[ei],device=device,dtype=torch.float32), hidden_states, past_ht,
                                                                                past_x, past_c, counter, 0)
                lstmn_outputs[ei] = outputs
                counter += 1
            lstmn_hidden = outputs.unsqueeze(0).unsqueeze(0)

            decoder_hidden = lstmn_hidden
            encoder_outputs = lstmn_outputs
            #print("start decode")
            decoder_input = torch.tensor(first_sentence_chn[0],device=device,dtype=torch.float32)  # SOS
            hidden_states_2 = []
            past_ht_2 = []
            past_x_2 = []
            past_c_2 = []
            past_yt = []
            counter = 0
            acc = 0
            lossfunc = torch.nn.CosineEmbeddingLoss(reduction='none')
            for di in range(len(first_sentence_eng)):
                outputs, hidden_states_2, past_ht_2, past_x_2, past_c_2, past_yt, counter = lstmn2(decoder_input,
                                                                                                  hidden_states_2,
                                                                                                  past_ht_2,
                                                                                                  past_x_2, past_c_2,
                                                                                                  counter,
                                                                                                  past_c, past_ht, past_yt)
                counter += 1
                decoder_output, decoder_hidden, decoder_attention = attn_decoder1(outputs.unsqueeze(0).unsqueeze(0),
                                                                            decoder_hidden,
                                                                            encoder_outputs)
                decoder_input = decoder_output.squeeze()
                output = find_most_similar(nlpchn, decoder_output.detach().cpu().numpy())
                #print(output)
                hypothesis.append(output)
                #print(torch.tensor(first_sentence_chn[di],device=device).shape,decoder_output.shape)
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
      print("Cosine_similarity_acc_20:", total_acc/20)
      print("BlUE_20:", total_blue*100/20)


# ------------------------------------------------------------------------------

# hidden_size = 256
max_len = 30
dim = args.hidden_size
batch_size = 1
attn_decoder1 = AttnDecoderRNN(args.hidden_size, dim, dropout_p=args.dropout).to(
    device)  # original is output_lang.n_words

lstmn = LSTMN(dim, max_len, batch_size, dim, dim,0)
lstmn2 = LSTMN(dim, max_len, batch_size, dim, dim,1)


torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
random.seed(1)
np.random.seed(1)
torch.backends.cudnn.deterministic = True

trainIters(attn_decoder1, lstmn, lstmn2, args.n_iters, print_every=args.print_every)  # hyperparameters
#evaluate(attn_decoder1,lstmn, lstmn2, max_length=args.MAX_LENGTH)