import torch.nn as nn
import time
import os
import sys

import numpy as np
import config
from util import *
import paths


class BaselineLstm(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, output_dim, nlayers, bidirectional, lstm_dropout, dropout, pad_idx):
        super(BaselineLstm, self).__init__()
        # input padding index to embedding to prevent training embedding for paddings
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx = pad_idx)
        self.lstm = nn.LSTM(embed_size, 
                           hidden_size, 
                           num_layers=nlayers, 
                           bidirectional=bidirectional, 
                           dropout=lstm_dropout)
        self.fc = nn.Linear(hidden_size * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, text, text_lengths):
        
        # [sent len, batch size]
        embedded = self.dropout(self.embedding(text)) #[sent len, batch size, emb dim]
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
#         #unpack sequence
#         output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output) # [sent len, batch size, hid dim * num directions]
        # [forward_layer_0, backward_layer_0, forward_layer_1, backward_layer 1, ..., forward_layer_n, backward_layer n]
        # use the top two hidden layers 
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        return self.fc(hidden.squeeze(0))
    
def run(model, optimizer, criterion, train_dataloader, valid_dataloader, best_epoch, best_vali_loss, DEVICE, start_epoch=None):
    best_eval = None
    start_epoch = 0 if start_epoch is None else start_epoch
    max_epoch = config.max_epoch
    batch_size = config.batch_size
    
    model = model.to(DEVICE)
    criterion.to(DEVICE)
    
    for epoch in range(start_epoch, max_epoch+1):
        start_time = time.time()
        model.train()
        # outputs records
        f = open(os.path.join(paths.output_path,'metrics.txt'), 'a')
        print_file_and_screen('### Epoch %5d' % (epoch), f=f)
        
        avg_loss = 0
        avg_acc = 0
        num_batches = len(train_dataloader)
        for idx, batch in enumerate(train_dataloader): # lists, presorted, preloaded on GPU
            optimizer.zero_grad()
            text, text_lengths = batch.text
            predictions = model(text, text_lengths).squeeze(1)
            loss = criterion.forward(predictions, batch.label)
            acc = binary_accuracy(predictions, batch.label)

            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
            avg_acc += acc.item()
            
            if idx%50 == 49:
                print_file_and_screen('Epoch: {}\tBatch: {}\tAvg-Loss: {:.4f}\tAvg-Acc: {:.4f}'.format(epoch, idx+1, avg_loss/50 ,avg_acc/50), f = f)
                avg_loss = 0.0
                avg_acc = 0.0
            # clear memory
            torch.cuda.empty_cache()
#             batch = batch.detach()
            del batch
            del loss
                
        train_loss, train_acc = test_validation(model, criterion, train_dataloader)
        val_loss, val_acc = test_validation(model, criterion, valid_dataloader)
        print_file_and_screen('Train Loss: {:.4f}\tTrain Acc: {:.4f}\tVal Loss: {:.4f}\tVal Acc: {:.4f}'.format(train_loss, train_acc, val_loss, val_acc), f=f)
        
        # check whether the best
        if val_loss < best_vali_loss:
            best_vali_loss = val_loss
            best_epoch = epoch
            is_best = True
        else:
            is_best = False
        
        
        save_checkpoint({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
            'best_vali_loss': best_vali_loss,
            'best_epoch': best_epoch,
            'optimizer_label_state_dict' : optimizer.state_dict()
        }, is_best, paths.output_path, filename=config.model_prefix+str(epoch)+'.pth.tar')
        
        
        end_time = time.time()
        print_file_and_screen('Epoch time used: ', end_time - start_time, 's', f=f)
        
        f.close()
    
    # print summary to the file
    with open(os.path.join(paths.output_path,'metrics.txt'), 'a') as f:
        print_file_and_screen('Summary:', f=f)
        print_file_and_screen('- Best Epoch: %1d | - Best Val Loss: %.4f'%(best_epoch, best_vali_loss), f=f)

def test_validation(model, criterion, valid_dataloader):
    model.eval()
    num_batches = len(valid_dataloader)
    avg_loss = 0.0
    avg_acc = 0.0
    for idx, batch in enumerate(valid_dataloader):
        text, text_lengths = batch.text
        predictions = model(text, text_lengths).squeeze(1)
        loss = criterion.forward(predictions, batch.label)
        acc = binary_accuracy(predictions, batch.label)
        avg_loss += loss.item()
        avg_acc += acc.item()
    return avg_loss/num_batches, avg_acc/num_batches


def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division 
    acc = correct.sum() / len(correct)
    return acc

def predict(model, test_dataloader):
    with torch.no_grad():
        model.eval()
        prediction = []
        for i, batch in enumerate(test_dataloader):
            if i%100 == 0:
                    print(i)
            text, text_lengths = batch.text
            predictions_batch = model(text, text_lengths).squeeze(1)
            prediction.append(predictions_batch)
    return prediction