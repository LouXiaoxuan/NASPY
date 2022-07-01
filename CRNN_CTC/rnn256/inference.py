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

use_cuda = torch.cuda.is_available()
# print(torch.cuda.device_count())
torch.manual_seed(7) # enable repeating the results
device = torch.device("cuda:2" if use_cuda else "cpu")

class ResCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, dropout):
        super(ResCNN, self).__init__()

        self.op = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(in_channels, out_channels, kernel, stride, padding=kernel//2),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel, stride, padding=kernel//2),
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
            *[ResCNN(32, 64, 7, stride=1, dropout=dropout) for _ in range(cnn_layers)]
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

### Evaluation Metric: OER (Operation prediction Error Rate) = Edit_Distance (decodes, targets) / len(targets)
def OER(decode_targets, decode_results, truth_len):
    truth_len = truth_len.tolist()
    batch_size = len(decode_targets)
    truth, hypo = [], []
    edit_distance = []
    for i in range(batch_size):
        truth.append(''.join(str(e) if e <10 else 'a' for e in decode_targets[i]))
        hypo.append(''.join(str(e) if e <10 else 'a' for e in decode_results[i]))
        edit_distance.append(Levenshtein.distance(truth[i], hypo[i]))
    batch_oer = list(map(lambda x: x[0]/x[1], zip(edit_distance, truth_len)))

    return batch_oer


def test(model, test_dataset, batch_size):
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
        output = F.log_softmax(output, dim=2)
        output = output.transpose(0, 1) #[length, batch, n_class]

        ### Use beam decoder
        beam_decoder = CTCBeamDecoder(["nor_conv", "3_oc_conv", "5_oc_conv", "d_3_oc_conv", "d_5_oc_conv", \
            "1_nc_conv", "pool", "skip", "fc", "iner_interval", "outer_interval", "_"], beam_width=1, \
            num_processes=int(mp.cpu_count()), blank_id=11, log_probs_input=True)
        beam_decode, beam_scores, timesteps, out_lens = beam_decoder.decode(output.transpose(0, 1))
        decode_results, decode_targets = [], []
        for i in range(batch_size):        
            decode_targets.append(test_label[i][:label_lengths[index[i]]].tolist())
            decode_results.append(beam_decode[i][0][:out_lens[i][0]].tolist())

        batch_oer = OER(decode_targets, decode_results, label_lengths[index])
        aver_batch_oer = sum(batch_oer)/batch_size
        if num_batch % 20 == 0:
            print("Batch: {}, aver_batch_oer:{:6f}".format(num_batch, aver_batch_oer))
        oer += aver_batch_oer

    oer = oer / num_batch
    print("Test oer:{:6f}".format(oer))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cnn_layers', type=int, default=1, help='number of cnn layers.')
    parser.add_argument('--rnn_layers', type=int, default=1, help='number of rnn layers.')
    parser.add_argument('--rnn_dim', type=int, default=256, help='dimension of rnn imput.') # 128
    parser.add_argument('--n_class', type=int, default=12, help='number of final classes.')
    parser.add_argument('--n_feats', type=int, default=32, help='number of input features.')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', type=int, default=5e-4) # 5e-4
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()

    ## load data and label
    with open("./normal_dataset.pt", "rb") as test_data:
        test_dataset = pickle.load(test_data)

    model = ToolModule(args.cnn_layers, args.rnn_layers, args.n_feats, args.rnn_dim, args.n_class, args.dropout).to(device)

   
    save_path = "./rnn_256.pt"

    if os.path.exists(save_path):
        checkpoint = torch.load(save_path)
        # model = checkpoint['base-model']
        model.load_state_dict(checkpoint['base-model'])

        test(model, test_dataset, args.batch_size)
    else:
        print("model file is not found !")

if __name__ == "__main__":
    main()













