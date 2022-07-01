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
from transformer.decoder import Decoder
from transformer.encoder import Encoder
from transformer.loss import cal_performance
from transformer.optimizer import TransformerOptimizer
from transformer.transformer import Transformer
from utils import parse_args, save_checkpoint, AverageMeter, get_logger

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

def train(model, train_dataset, optimizer, scheduler, epoch):
    model.train()
    data_len = len(train_dataset[0])
    losses = AverageMeter()

    shuffled_index=torch.randperm(data_len)
    num_batch, test_loss, oer = 0, 0, 0

    for count in range(0, data_len, args.batch_size):
        train_sequence, train_label, train_sequence_lengths, train_label_lengths = \
        BatchDataProcess(train_dataset, shuffled_index, count, args.batch_size)

        train_sequence.requires_grad_()

        pred, gold = model(train_sequence, train_sequence_lengths, train_label)
        loss, n_correct = cal_performance(pred, gold, smoothing=args.label_smoothing)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        scheduler.step()

        losses.update(loss.item())
        if count % 1000 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                        'Loss {loss.val:.5f} ({loss.avg:.5f})'.format(epoch, count, data_len, loss=losses))

    return losses.avg


def test(model, test_dataset, epoch):
    model.eval()
    data_len = len(test_dataset[0])
    losses = AverageMeter()
    index_list = range(0, data_len)
    char_list = []
    total_oer = 0

    for count in range(0, data_len, args.batch_size):
        test_sequence, test_label, test_sequence_lengths, test_label_lengths = \
            BatchDataProcess(test_dataset, index_list, count, args.batch_size)

        with torch.no_grad():
            # Forward prop.
            pred, gold = model(test_sequence, test_sequence_lengths, test_label)
            loss, n_correct = cal_performance(pred, gold, smoothing=args.label_smoothing)
        
        # Keep track of metrics
        losses.update(loss.item())

    print('\nTest Loss {:.6f} ({:.6f})\n'.format(losses.val, losses.avg))

    return losses.avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_layers_enc', type=int, default=1, help='number of encoder layers.')
    parser.add_argument('--n_layers_dec', type=int, default=1, help='number of decoder layers.')
    parser.add_argument('--d_model', type=int, default=256, help='dimension of model imput.') 
    parser.add_argument('--n_head', type=int, default=8, help='number of heads in MHA.')
    parser.add_argument('--d_k', type=int, default=32, help='Dimension of key')
    parser.add_argument('--d_v', type=int, default=32, help='Dimension of value')
    parser.add_argument('--d_inner', type=int, default=512, help='dimension of feedforward layer.')
    parser.add_argument('--d_word_vec', type=int, default=256, help='Dim of decoder embedding.')
    parser.add_argument('--d_input', type=int, default=96, help='number of input features.')
    parser.add_argument('--num-workers', default=4, type=int, help='Number of workers to generate minibatch')
    parser.add_argument('--tgt_emb_prj_weight_sharing', default=1, type=int, help='share decoder embedding with decoder projection')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=int, default=5e-4) # 5e-4
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--label_smoothing', default=0.1, type=float,
                        help='label smoothing')
    parser.add_argument('--checkpoint', type=str, default="BasicModel_256.pt")
    parser.add_argument('--pe_maxlen', default=7000, type=int, help='Positional Encoding max len')
    parser.add_argument('--beam_size', default=5, type=int, help='Beam size')
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
    with open("../train_dataset_noinval.pt", "rb") as train_data:
        train_dataset = pickle.load(train_data)
    with open("../test_dataset_noinval.pt", "rb") as test_data:
        test_dataset = pickle.load(test_data)

    train_dataset = DatasetPreprocess(train_dataset)
    test_dataset = DatasetPreprocess(test_dataset)
    # print(train_dataset[0].size(), train_dataset[1][0])
    # os._exit(0)

    if os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']
        scheduler = checkpoint['scheduler']
    else:
        encoder = Encoder(args.d_input, args.n_layers_enc, args.n_head,
                            args.d_k, args.d_v, args.d_model, args.d_inner,
                            dropout=args.dropout, pe_maxlen=args.pe_maxlen)
        decoder = Decoder(sos_id, eos_id, vocab_size,
                            args.d_word_vec, args.n_layers_dec, args.n_head,
                            args.d_k, args.d_v, args.d_model, args.d_inner,
                            dropout=args.dropout,
                            tgt_emb_prj_weight_sharing=args.tgt_emb_prj_weight_sharing,
                            pe_maxlen=args.pe_maxlen)
        model = Transformer(encoder, decoder).to(device)
        # model = nn.DataParallel(model, device_ids=[0,1,2,3])

        # optimizer = TransformerOptimizer(
        #         optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-09), d_model = args.d_model)
        # scheduler = 0

        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-09)
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, 
                                                steps_per_epoch=int(len(train_dataset[0])/args.batch_size),
                                                epochs=args.epochs,
                                                anneal_strategy='linear')
        

    # print(model)
    print("Number of model parameters:", sum([param.nelement() for param in model.parameters()]))                          

    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        train_loss = train(model, train_dataset, optimizer, scheduler, epoch)
        end_time = time.time()
        print("training for epoch {} takes {} seconds".format(epoch, end_time-start_time))
        # if epoch % 10 == 0 or epoch == args.epochs - 1 :
        start_time = time.time()
        test_loss = test(model, test_dataset, epoch)
        end_time = time.time()
        print("testing for epoch {} takes {} seconds".format(epoch, end_time-start_time))

        # Check if there was an improvement
        is_best = test_loss < best_loss
        best_loss = min(test_loss, best_loss)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        best_path = "BestModel_256.pt"
        basic_path = "BasicModel_256.pt"
        save_checkpoint(epoch, epochs_since_improvement, model, optimizer, scheduler, best_loss, is_best, basic_path, best_path)


if __name__ == "__main__":
    main()








