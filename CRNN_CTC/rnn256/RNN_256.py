import os
import time
import argparse
import numpy as np 
import pandas as pd 
import pickle
from copy import deepcopy
from ctcdecode import CTCBeamDecoder
import Levenshtein
import multiprocessing as mp

import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

##oc: one-channel; nc: n-channel
op_int = {"nor_conv":0, "3_oc_conv":1, "5_oc_conv":2, "d_3_oc_conv":3, "d_5_oc_conv":4, \
    "1_nc_conv":5, "pool":6, "skip":7, "fc":8, "interval":9}

use_cuda = torch.cuda.is_available()
# print(torch.cuda.device_count())
torch.manual_seed(7) # enable repeating the results
device = torch.device("cuda:0" if use_cuda else "cpu")

class ResCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, dropout):
        super(ResCNN, self).__init__()

        self.op = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(in_channels, out_channels, kernel, stride, padding=kernel//2),
            # nn.BatchNorm1d(out_channels),
            # nn.GELU(),
            # nn.Dropout(dropout),
            # nn.Conv1d(out_channels, out_channels, kernel, stride, padding=kernel//2),
        )

    def forward(self, x):
        # residual = x
        x = self.op(x)  ## number of feature equals to the channel size, which change from 3->32->64
        # x += residual
        return x  # (batch, channel, sequence_length)


class BiGRU(nn.Module):
    def __init__(self, rnn_dim, hidden_size, dropout, batch_first):
        super(BiGRU, self).__init__()

        self.bigru = nn.GRU(input_size=rnn_dim, hidden_size=hidden_size, \
            num_layers=1, batch_first=batch_first, bidirectional=True)
        self.layer_norm = nn.LayerNorm(rnn_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        x = F.gelu(x)
        x, _ = self.bigru(x)
        x = self.dropout(x)
        return x


class ToolModule(nn.Module):
    def __init__(self, cnn_layers, rnn_layers, feat_dim, rnn_dim, n_class, dropout): # feat_dim = 64, rnn_dim =128
        super(ToolModule, self).__init__()

        self.stemcnn = nn.Conv2d(1, 32, 3, stride=1, padding=1)
        self.rescnn = nn.Sequential(
            *[ResCNN(32, 32, 7, stride=1, dropout=dropout) for _ in range(cnn_layers)]
        )
        self.fc = nn.Linear(feat_dim * 3, rnn_dim)
        self.rnn = nn.Sequential(
            *[BiGRU(rnn_dim = rnn_dim if i == 0 else rnn_dim*2, \
                hidden_size = rnn_dim, dropout = dropout, batch_first=True) \
                    for i in range(rnn_layers)]
        )
        self.classifier = nn.Sequential(
            nn.Linear(rnn_dim*2, rnn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(rnn_dim, n_class)
        )

    def forward(self, x): 
        x = self.stemcnn(x)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1]*sizes[2], sizes[3])
        # x = self.rescnn(x) # x = [batch, channel (features), length]
        x = x.transpose(1, 2) # x = [batch, length, channel (features)]
        x = self.fc(x)
        x = self.rnn(x)
        x = self.classifier(x)
        return x

def GreedyDecoder(output, labels, label_lengths, blank_label=11, collapse_repeated=True):
    arg_maxes = torch.argmax(output, dim=2)
    # print(arg_maxes)
    decodes = []
    targets = []
    for i, args in enumerate(arg_maxes):
        # print("the first is {}".format(i))
        decode = []
        targets.append(labels[i][:label_lengths[i]].tolist())
        for j, index in enumerate(args):
            # print("the second is {}".format(j))
            if index != blank_label:
                if collapse_repeated and j != 0 and index == args[j -1]:
                    continue
                decode.append(index.item())
                # print("decode:", decode)
        decodes.append(decode)  #TODO: convert the int to text
    return decodes, targets

def levenshtein_distance(truth, hypo):

    m = len(truth)
    n = len(hypo)

    # basic cases
    if hypo == truth:
        return 0
    elif m == 0:
        return n
    elif n == 0:
        return m

    if m < n:
        truth, hypo = hypo, truth
        m, n = n, m

    distance = np.zeros((2, n + 1), dtype = np.int32)

    for j in range(0, n+1):
        distance[0][j] = j
    
    for i in range(1, m+1):
        prev_row_idx = (i-1) % 2 
        cur_row_idx = i % 2
        distance[cur_row_idx][0] = i
        for j in range(1, n+1):
            if truth[i-1] == hypo[j-1]:
                distance[cur_row_idx][j] = distance[prev_row_idx][j-1]
            else:
                s_num = distance[prev_row_idx][j-1] + 1
                i_num = distance[cur_row_idx][j-1] + 1
                d_num = distance[prev_row_idx][j] + 1
                distance[cur_row_idx][j] = min(s_num, i_num, d_num)

    return distance[m%2][n]

### Evaluation Metric: OER (Operation prediction Error Rate) = Edit_Distance (decodes, targets) / len(targets)
def OER(decode_targets, decode_results, truth_len):
    truth_len = truth_len.tolist()
    batch_size = len(decode_targets)
    # num_cores = int(mp.cpu_count())
    # corepool = mp.Pool(num_cores)
    truth, hypo = [], []
    edit_distance = []
    for i in range(batch_size):
        truth.append(''.join(str(e) if e <10 else 'a' for e in decode_targets[i]))
        hypo.append(''.join(str(e) if e <10 else 'a' for e in decode_results[i]))
        edit_distance.append(Levenshtein.distance(truth[i], hypo[i]))
        # print(Levenshtein.distance(truth[i], hypo[i]))    ##to gpu?
        # print(truth[i], hypo[i])
        # os._exit(0)
    # edit_distance = [corepool.apply_async(Levenshtein.distance, args=(truth[i], hypo[i])) for i in range(batch_size)]
    # edit_distance = [p.get() for p in edit_distance]
    # print(edit_distance)
    # print(truth_len)
    batch_oer = list(map(lambda x: x[0]/x[1], zip(edit_distance, truth_len)))

    return batch_oer

def train(model, train_dataset, criterion, optimizer, scheduler, epoch, batch_size):
    print("\nstart training!")
    model.train()
    data_len = len(train_dataset[0])
    # print(data_len)
    input_sequences, labels, input_lengths, label_lengths = train_dataset

    input_sequences = input_sequences.unsqueeze(1)

    shuffled_index=torch.randperm(data_len)
    num_batch, test_loss, oer = 0, 0, 0
    
    for count in range(0, data_len, batch_size):
    # for count in range(0, 100, batch_size):
        num_batch += 1
        index = shuffled_index[count : count + batch_size]
        # print(index)
        train_sequence=input_sequences[index].to(device)
        train_sequence.requires_grad_()
        train_label=labels[index].to(device=device, dtype=torch.int64)

        optimizer.zero_grad()
        output = model(train_sequence) # [batch, length, n_class]
        # print(output.size())
        # print(output)
        output = F.log_softmax(output, dim=2)
        # print(output.size())

        # if count > 7600 : 
        #     interval_results = torch.argmax(output, dim=2)
        #     out_seq = []
        #     for asd, ele_index in enumerate(interval_results):          
        #         ele_count = sum(i != 11 for i in ele_index).item()
        #         out_seq.append(ele_count)
        #     print(out_seq)


        output = output.transpose(0, 1) #[length, batch, n_class]
        # print(output.size())   # here can also do: stem_cnn stride=2, input_lengths /= 2

        loss = criterion(output, train_label, input_lengths[index], label_lengths[index]) 
        test_loss += loss.detach().item() 
        loss.backward()

        optimizer.step()
        scheduler.step()

        # ### Use beam decoder
        # beam_decoder = CTCBeamDecoder(["nor_conv", "3_oc_conv", "5_oc_conv", "d_3_oc_conv", "d_5_oc_conv", \
        #     "1_nc_conv", "pool", "skip", "fc", "iner_interval", "outer_interval", "_"], beam_width=5, num_processes=int(mp.cpu_count()), blank_id=11, log_probs_input=True)
        # beam_decode, beam_scores, timesteps, out_lens = beam_decoder.decode(output.transpose(0, 1))
        # decode_results, decode_targets = [], []
        # for i in range(batch_size):        
        #     decode_targets.append(train_label[i][:label_lengths[index[i]]].tolist())
        #     decode_results.append(beam_decode[i][0][:out_lens[i][0]].tolist())
        # batch_oer = OER(decode_targets, decode_results, label_lengths[index])
        # aver_batch_oer = sum(batch_oer)/batch_size
        # oer += aver_batch_oer
       
        if num_batch %20 == 0:
            print("Train Epoch: {}, Batch: {}, Loss: {:.6f}".format(epoch, num_batch, loss.detach().item()))

    test_loss = test_loss / num_batch
    oer = oer / num_batch
    print("Train Epoch: {}, AverageLoss: {:.6f}, Oer:{:6f}".format(epoch, test_loss, oer))



def test(model, test_dataset, criterion, epoch, batch_size):
    print("\nstart evaluating!")
    model.eval()
    data_len = len(test_dataset[0])
    input_sequences, labels, input_lengths, label_lengths = test_dataset
    test_loss, num_batch, oer = 0, 0, 0
    decode_results_list, decode_targets_list = [], []

    input_sequences = input_sequences.unsqueeze(1)
    
    for count in range(0, data_len, batch_size):
        num_batch += 1
        index = range(0, data_len)[count : count + batch_size]
        test_sequence=input_sequences[index].to(device)
        test_label=labels[index].to(device=device, dtype=torch.int64)

        output = model(test_sequence) # [batch, length, n_class]
        # print(output.size())
        # print(output)
        output = F.log_softmax(output, dim=2)
        # print(output)
        output = output.transpose(0, 1) #[length, batch, n_class]

        loss = criterion(output, test_label, input_lengths[index], label_lengths[index]) 
        test_loss += loss.detach().item() 

        ### Use beam decoder
        # print("batch[{}] start beam decode !".format(num_batch))
        beam_decoder = CTCBeamDecoder(["nor_conv", "3_oc_conv", "5_oc_conv", "d_3_oc_conv", "d_5_oc_conv", \
            "1_nc_conv", "pool", "skip", "fc", "iner_interval", "outer_interval", "_"], beam_width=1, num_processes=int(mp.cpu_count()), blank_id=11, log_probs_input=True)
        beam_decode, beam_scores, timesteps, out_lens = beam_decoder.decode(output.transpose(0, 1))
        decode_results, decode_targets = [], []
        # print("beam decode finish !")
        for i in range(batch_size):        
            decode_targets.append(test_label[i][:label_lengths[index[i]]].tolist())
            decode_results.append(beam_decode[i][0][:out_lens[i][0]].tolist())
        # print(decode_results[0], decode_targets[0])
        # print("append finish !")

        ### Use greedy decoder
        # decode_results, decode_targets = GreedyDecoder(output.transpose(0, 1), test_label, label_lengths[index])
        # print(decode_results[0], decode_targets[0])

        if epoch == 99:
            for i in range(batch_size): 
                decode_results_list.append(decode_results[i])
                decode_targets_list.append(decode_targets[i])      

        batch_oer = OER(decode_targets, decode_results, label_lengths[index])
        # print(batch_oer)
        aver_batch_oer = sum(batch_oer)/batch_size
        if num_batch % 20 == 0:
            print("Epoch: {}, batch: {}, loss: {:.6f}, aver_batch_oer:{:6f}".format(epoch, num_batch, loss.item(), aver_batch_oer))
        oer += aver_batch_oer

    test_loss = test_loss / num_batch
    oer = oer / num_batch
    print("Test Epoch: {}, Loss: {:.6f}, oer:{:6f}".format(epoch, test_loss, oer))

    if epoch == 99:
        with open("predicted_labels_ed_0.pkl", "wb+") as plf:
            pickle.dump([decode_results_list,decode_targets_list], plf)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cnn_layers', type=int, default=1, help='number of cnn layers.')
    parser.add_argument('--rnn_layers', type=int, default=1, help='number of rnn layers.')
    parser.add_argument('--rnn_dim', type=int, default=256, help='dimension of rnn imput.') # 128
    parser.add_argument('--n_class', type=int, default=12, help='number of final classes.')
    parser.add_argument('--n_feats', type=int, default=32, help='number of input features.')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', type=int, default=5e-4) # 5e-4
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()

    ## load data and label
    ## train/test dataset format: [[[input_features]], [[label]], [input_length], [label_length]], each list has number_of_sequences elements
    with open("../../train_dataset.pt", "rb") as train_data:
        train_dataset = pickle.load(train_data)
    with open("../../test_dataset.pt", "rb") as test_data:
        test_dataset = pickle.load(test_data)

    model = ToolModule(args.cnn_layers, args.rnn_layers, args.n_feats, args.rnn_dim, args.n_class, args.dropout).to(device)

    # print(model)
    print("Number of model parameters:", sum([param.nelement() for param in model.parameters()]))

    criterion = nn.CTCLoss(blank = 11, zero_infinity=False).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.learning_rate, 
                                            steps_per_epoch=int(len(train_dataset[0])/args.batch_size),
                                            epochs=args.epochs,
                                            anneal_strategy='linear')

    save_path = "crnn_ctc_model_ed_0.pt"
    start_epoch = 0

    if os.path.exists(save_path):
        checkpoint = torch.load(save_path)
        args = checkpoint['args']
        start_epoch = checkpoint['start_epoch']
        model.load_state_dict(checkpoint['base-model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

    for epoch in range(start_epoch,args.epochs):
        # test(model, test_dataset, criterion, epoch, args.batch_size)
        start_time = time.time()
        train(model, train_dataset, criterion, optimizer, scheduler, epoch, args.batch_size)
        end_time = time.time()
        print("training for epoch {} takes {} seconds".format(epoch, end_time-start_time))
        # if epoch % 10 == 0 or epoch == args.epochs - 1 :
        start_time = time.time()
        test(model, test_dataset, criterion, epoch, args.batch_size)
        end_time = time.time()
        print("testing for epoch {} takes {} seconds".format(epoch, end_time-start_time))

        torch.save({
        'args':deepcopy(args),
        'start_epoch':epoch+1,
        'optimizer':optimizer.state_dict(),
        'scheduler':scheduler.state_dict(),
        'base-model':model.state_dict()}, save_path)


if __name__ == "__main__":
    main()













