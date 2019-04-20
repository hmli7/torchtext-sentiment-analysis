import torch.nn as nn
import torch.nn.functional as F
import time
import os
import sys

import numpy as np
import config
from util import *
import paths

class Fasttext(nn.Module):
    '''https://arxiv.org/pdf/1602.02373.pdf lstm+global pooling'''
    def __init__(self, vocab_size, embed_size, output_dim, dropout1, dropout2, pad_idx, train_embedding=True):
        super(Fasttext, self).__init__()
        # input padding index to embedding to prevent training embedding for paddings
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx = pad_idx)
        if not train_embedding:
            self.embedding.weight.requires_grad = False # make embedding non trainable
        self.fc = nn.Linear(embed_size, output_dim)
        self.dropout1 = nn.Dropout(dropout1)
        self.dropout2 = nn.Dropout(dropout2)
    def forward(self, text, text_lengths, testing=False):
        # [sent len, batch size]
        embedded = self.dropout1(self.embedding(text)) #[sent len, batch size, emb dim]
        embedded = embedded.permute(1, 0, 2)
        pooled = self.dropout2(F.avg_pool2d(embedded, (embedded.shape[1], 1)))
        y_hat = self.fc(pooled.squeeze(1))
        return y_hat
    
    
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
            predictions = model(text, text_lengths, testing=False)
            predictions = predictions.squeeze(1) if len(predictions.size())>1 else predictions # prepare for batch size == 1
            loss = criterion.forward(predictions, batch.label)
            acc = binary_accuracy(predictions, batch.label)

            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
            avg_acc += acc.item()
            
            if idx%200 == 199:
                print_file_and_screen('Epoch: {}\tBatch: {}\tAvg-Loss: {:.4f}\tAvg-Acc: {:.4f}'.format(epoch, idx+1, avg_loss/200 ,avg_acc/200), f = f)
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
        predictions = model(text, text_lengths, testing=False)
        predictions = predictions.squeeze(1) if len(predictions.size())>1 else predictions # prepare for batch size == 1
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

def predict(model, test_dataloader, DEVICE):
    model.to(DEVICE)
    with torch.no_grad():
        model.eval()
        prediction = []
        for i, batch in enumerate(test_dataloader):
            if i%400 == 0:
                print(i)
            text, text_lengths = batch.text
            predictions_batch = model(text, text_lengths, testing=True)
            predictions_batch = predictions_batch.squeeze(1) if len(predictions_batch.size())>1 else predictions_batch # prepare for batch size == 1
            rounded_preds = torch.round(torch.sigmoid(predictions_batch))
            prediction.append(rounded_preds)
    return torch.cat(prediction, dim=0).cpu().numpy()