import torch 
import torch.nn as nn 
import torch.optim as optim
import numpy as np
import argparse
import pickle
import Levenshtein
import tqdm
import os, time
from copy import deepcopy

from config import vocab_size, sos_id, eos_id
from data_gen import pad_collate
from transformer.decoder import Decoder
from transformer.encoder import Encoder
from transformer.loss import cal_performance
from transformer.optimizer import TransformerOptimizer
from transformer.transformer import Transformer
from utils import parse_args, save_checkpoint, AverageMeter, get_logger
from xer import cer_function

torch.manual_seed(7)
np.random.seed(7)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def DatasetPreprocess(dataset):
    input_sequences, labels, input_lengths, label_lengths = dataset
    new_sequences = input_sequences.transpose(1, 2)
    new_labels = []
    for i, label in enumerate(labels):
        new_labels.append(label[:label_lengths[i]])
    new_labels = nn.utils.rnn.pad_sequence(new_labels, batch_first=True, padding_value = -1)

    new_dataset = [new_sequences, new_labels, input_lengths, label_lengths]

    return new_dataset


def BatchDataProcess(dataset, index_list, count, batch_size):
    input_sequences, labels, input_lengths, label_lengths = dataset
    index = index_list[count : count + batch_size]
    batch_sequence=input_sequences[index].to(device)
    batch_sequence.requires_grad_()
    batch_label=labels[index].to(device=device, dtype=torch.int64)
    batch_sequence_lengths = input_lengths[index].to(device)
    batch_label_lengths = label_lengths[index].to(device)

    return batch_sequence, batch_label, batch_sequence_lengths, batch_label_lengths

def OER(gt_list, hyp_list, truth_len):
    truth_len = truth_len.tolist()
    batch_size = len(gt_list)
    edit_distance = []
    for i in range(batch_size):
        edit_distance.append(Levenshtein.distance(gt_list[i], hyp_list[i]))
    batch_oer = list(map(lambda x: x[0]/x[1], zip(edit_distance, truth_len)))

    return batch_oer

def test(model, test_dataset):
    model.eval()
    data_len = len(test_dataset[0])
    losses = AverageMeter()
    index_list = range(0, data_len)
    char_list = {0:'0', 1:'1', 2:'2', 3:'3', 4:'4', 5:'5', 6:'6', 7:'7', 8:'8', 9:'9', 10:'a', 11:'<sos>', 12:'<eos>'}
    total_oer = 0

    test_sequence, test_label, test_sequence_lengths, test_label_lengths = test_dataset

    for count in range(0, data_len):
        input_seq = test_sequence[count].to(device)
        input_length = test_sequence_lengths[count].unsqueeze(0).to(device)
        label_length = test_label_lengths[count].to(device)
        output_label = test_label[count][:label_length].tolist()
                
        # print("phase 1 start!")
        # start_time = time.time()       
        with torch.no_grad():
           nbest_hyps = model.recognize(input_seq, input_length, char_list, args)              
        # end_time = time.time()
        # print("phase 1 finish! time is {} seconds".format(end_time - start_time))

        hyp_list, gt_list = [], []

        for hyp in nbest_hyps:
            out = hyp['yseq']
            out = [char_list[idx] for idx in out if idx not in (sos_id, eos_id)]
            out = ''.join(out)
            hyp_list.append(out)

        print(hyp_list)

        gt = [char_list[idx] for idx in output_label if idx not in (sos_id, eos_id)]
        gt = ''.join(gt)
        gt_list.append(gt)
        print(gt_list)

        batch_oer = cer_function(gt_list, hyp_list)
        total_oer += batch_oer

        print("Batch: {}, aver_batch_oer:{:6f}".format(count, batch_oer))
            
        
    avg_oer = total_oer / data_len
    print('Avg_oer {:.6f}\n'.format(avg_oer))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_layers_enc', type=int, default=1, help='number of encoder layers.')
    parser.add_argument('--n_layers_dec', type=int, default=1, help='number of decoder layers.')
    parser.add_argument('--d_model', type=int, default=256, help='dimension of model imput.') 
    parser.add_argument('--n_head', type=int, default=8, help='number of heads in MHA.')
    parser.add_argument('--d_k', type=int, default=32, help='Dimension of key')
    parser.add_argument('--d_v', type=int, default=32, help='Dimension of value')
    parser.add_argument('--d_inner', type=int, default=1024, help='dimension of feedforward layer.')
    parser.add_argument('--d_word_vec', type=int, default=256, help='Dim of decoder embedding.')
    parser.add_argument('--d_input', type=int, default=96, help='number of input features.')
    parser.add_argument('--num-workers', default=4, type=int, help='Number of workers to generate minibatch')
    parser.add_argument('--tgt_emb_prj_weight_sharing', default=1, type=int, help='share decoder embedding with decoder projection')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=int, default=0.001) # 5e-4
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--label_smoothing', default=0.1, type=float,
                        help='label smoothing')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--pe_maxlen', default=7000, type=int, help='Positional Encoding max len')
    parser.add_argument('--beam_size', default=1, type=int, help='Beam size')
    parser.add_argument('--nbest', default=1, type=int, help='Nbest size')
    parser.add_argument('--decode_max_len', default=0, type=int,
                        help='Max output length. If ==0 (default), it uses a '
                             'end-detect function to automatically find maximum '
                             'hypothesis lengths')
    global args
    args = parser.parse_args()

    start_epoch = 0
    best_loss = float('inf')
    epochs_since_improvement = 0

    ## load data and label
    ## train/test dataset format: [[[input_features]], [[label]], [input_length], [label_length]], each list has number_of_sequences elements
    with open("../train_dataset.pt", "rb") as train_data:
        train_dataset = pickle.load(train_data)
    with open("../test_dataset.pt", "rb") as test_data:
        test_dataset = pickle.load(test_data)

    train_dataset = DatasetPreprocess(train_dataset)
    test_dataset = DatasetPreprocess(test_dataset)                   

    checkpoint = "BasicModel_256.pt"
    checkpoint = torch.load(checkpoint, map_location='cpu')
    model = checkpoint['model'].to(device)

    start_time = time.time()
    test(model, test_dataset)
    end_time = time.time()
    print("testing takes {} seconds".format(end_time-start_time))



if __name__ == "__main__":
    main()








