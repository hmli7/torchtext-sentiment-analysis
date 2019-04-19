#!/usr/bin/env python
# coding: utf-8

# In[1]:
# python naivebayes.py dev_text.txt dev_label.txt heldout_text.txt heldout_pred_nb.txt

import numpy as np
import os
from collections import Counter
import pickle
import string
import copy
import sys


# In[3]:


input_path = '.'
dev_text_name = sys.argv[1]
dev_label_name = sys.argv[2]
heldout_text_name = sys.argv[3]
output_name = sys.argv[4]

with open(os.path.join(input_path, dev_text_name), 'r', encoding='utf-8') as f:
    dev_text = f.read().split('\n')

with open(os.path.join(input_path, heldout_text_name), 'r', encoding='utf-8') as f:
    heldout_text = f.read().split('\n')


# In[4]:


# dev_path = os.path.join(path, 'input', 'dev_text_processed.pkl')
dev_label_path = os.path.join(input_path, dev_label_name)
# test_path = os.path.join(path, 'heldout_text_processed.pkl')


# # Data Transformation

# In[5]:


# define stop words


# In[6]:


# stop words refer to spacy.lang.stopwords in sm language model
stopwords = {'a',
 'about',
'across',
 'after',
 'afterwards',
 'again','all',
 'almost',
 'alone',
 'along',
 'already',
 'also',
 'am',
 'among',
 'amongst',
 'amount',
 'an',
 'and',
 'another',
 'any',
 'anyhow',
 'anyone',
 'anything',
 'anyway',
 'anywhere',
 'are',
 'around',
 'as',
 'at',
 'back',
 'be',
 'became',
 'because',
 'become',
 'becomes',
 'becoming',
 'been',
 'before',
 'beforehand',
 'behind',
 'being',
 'below',
 'beside',
 'besides',
 'between',
 'beyond',
 'both',
 'bottom',
 'by',
 'ca',
 'call',
 'could',
 'did',
 'do',
 'does',
 'doing',
 'done',
 'down',
 'due',
 'during',
 'each',
 'eight',
 'either',
 'eleven',
 'else',
 'elsewhere',
 'empty',
 'enough',
 'even',
 'ever',
 'every',
 'everyone',
 'everything',
 'everywhere',
 'except',
 'few',
 'fifteen',
 'fifty',
 'first',
 'five',
 'for',
 'former',
 'formerly',
 'forty',
 'four',
 'from',
 'front',
 'full',
 'further',
 'get',
 'give',
 'go',
 'had',
 'has',
 'have',
 'he',
 'hence',
 'her',
 'here',
 'hereafter',
 'hereby',
 'herein',
 'hereupon',
 'hers',
 'herself',
 'him',
 'himself',
 'his',
 'how',
 'however',
 'hundred',
 'i',
 'if',
 'in',
 'into',
 'is',
 'it',
 'its',
 'itself',
 'just',
 'keep',
 'last',
 'latter',
 'latterly',
 'least',
 'less',
 'made',
 'make',
 'many',
 'may',
 'me',
 'meanwhile',
 'might',
 'mine',
 'more',
 'moreover',
 'most',
 'mostly',
 'move',
 'much',
 'must',
 'my',
 'myself',
 'name',
 'namely',
 'nevertheless',
 'next',
 'nine',
 'noone',
 'nor',
 'now',
 'nowhere',
 'of',
 'often',
 'on',
 'once',
 'one',
 'only',
 'onto',
 'or',
 'other',
 'others',
 'otherwise',
 'our',
 'ours',
 'ourselves',
 'out',
 'over',
 'own',
 'part',
 'per',
 'perhaps',
 'please',
 'put',
 'quite',
 'rather',
 're',
 'really',
 'regarding',
 'same',
 'say',
 'see',
 'seem',
 'seemed',
 'seeming',
 'seems',
 'serious',
 'several',
 'she',
 'should',
 'show',
 'side',
 'since',
 'six',
 'sixty',
 'so',
 'some',
 'somehow',
 'someone',
 'something',
 'sometime',
 'sometimes',
 'somewhere',
 'still',
 'such',
 'take',
 'ten',
 'than',
 'that',
 'the',
 'their',
 'them',
 'themselves',
 'then',
 'thence',
 'there',
 'thereafter',
 'thereby',
 'therefore',
 'therein',
 'thereupon',
 'these',
 'they',
 'third',
 'this',
 'those',
 'though',
 'three',
 'through',
 'throughout',
 'thru',
 'thus',
 'to',
 'together',
 'too',
 'top',
 'toward',
 'towards',
 'twelve',
 'twenty',
 'two',
 'under',
 'unless',
 'until',
 'up',
 'upon',
 'us',
 'used',
 'using',
 'various',
 'very',
 'via',
 'was',
 'we',
 'well',
 'were',
 'what',
 'whatever',
 'when',
 'whence',
 'whenever',
 'where',
 'whereafter',
 'whereas',
 'whereby',
 'wherein',
 'whereupon',
 'wherever',
 'whether',
 'which',
 'while',
 'whither',
 'who',
 'whoever',
 'whole',
 'whom',
 'whose',
 'why',
 'will',
 'with',
 'within',
 'would',
 'yet',
 'you',
 'your',
 'yours',
 'yourself',
 'yourselves'}


# In[179]:


def process(data_list, deep_clean=False, stop_words=None, add_bigram=False):
    '''process the data'''
    processed_list = []
    for data in data_list:
        # tokenize and remove extra spaces
        tokens = data.split()
        # capitalize and remain only words,remove extra spaces
        if stop_words is not None:
            remove_list = copy.deepcopy(stop_words)
            remove_list.update(set(string.punctuation))
            upper_tokens = [clean(token, deep_clean=deep_clean) for token in tokens if token.lower() not in remove_list]
        else:
            upper_tokens = [clean(token, deep_clean=deep_clean) for token in tokens if token not in string.punctuation]
        if add_bigram:
            n_grams = set(zip(*[upper_tokens[i:] for i in range(2)]))
            upper_tokens.extend(n_grams)
        processed_list.append(upper_tokens)
    return processed_list
    
def clean(text, deep_clean=False):
    '''remove punctuations within a string or not'''
    if not deep_clean:
        return text.upper()
    else:
        return text.translate(str.maketrans('','',string.punctuation)).upper()


# __Need to remove infrequent words for unigram and bigram seperately, need to add codes for this__

# In[180]:


dev_X = process(dev_text, deep_clean=True, stop_words=stopwords, add_bigram=False)


# In[182]:


test_X = process(heldout_text, deep_clean=True, stop_words=stopwords, add_bigram=False)


# # Load data

# In[11]:


# with open(train_path, 'rb') as f:
#     dev_X = pickle.load(f)

with open(dev_label_path, 'r', encoding='utf-8') as f:
    dev_y = f.read().split('\n')


# In[12]:


def split_train_test(X, y, percentage=0.75):
    indices = np.random.permutation(len(X))
    threshold = int(len(dev_X)*percentage)
    training_idx, test_idx = indices[:threshold], indices[threshold:]
    return X[training_idx], y[training_idx], X[test_idx], y[test_idx]


# In[183]:


train_X, train_y, valid_X, valid_y = split_train_test(np.array(dev_X), np.array(dev_y), 0.75)


# # Train model

# In[184]:


# collect vocabulary
vocabulary = set(np.r_['0', np.concatenate(train_X),np.concatenate(valid_X),np.concatenate(test_X)])
vocabulary_size = len(vocabulary)


# In[185]:


vocabulary_size


# In[186]:


vocabulary_counter = Counter(np.r_['0', np.concatenate(train_X),np.concatenate(valid_X),np.concatenate(test_X)])


# In[101]:


def get_replace_list(vocabulary_counter, top_threshold_num, low_threshold_frequency):
    '''get a list of removal words that is < top_thres or > low_thres regarding frequency'''
    top_removals = [pair[0] for pair in vocabulary_counter.most_common(top_threshold_num)]
    low_removals = [pair[0] for pair in vocabulary_counter.items() if pair[1]<low_threshold_frequency]
    return top_removals+low_removals




# generate a list of replacing words that is on the top 3 in frequency ranking or having less than 2 frequency
unknownwords = get_replace_list(vocabulary_counter, 3, 10)


# In[131]:


def train_model(train_X, train_y, vocabulary_size, unknownwords=None):
    '''train model based on training set and return counts
    '''
    train_label_count =  Counter(train_y)

    train_label_probs = {label: train_label_count.get(label)/len(train_y) for label in train_label_count}
    
    train_probs = {}
    sudo_probs = {}
    
    for label in train_label_count.keys():
        train_x_label = train_X[train_y == label]
        # train counter
        train_X_concatenate = np.concatenate(train_x_label)
        # replace words that is unknownwords
        if unknownwords is not None:
            train_X_cleaned = [word if word not in unknownwords else '<UNK>' for word in np.concatenate(train_x_label)]
        else:
            train_X_cleaned = train_X_concatenate
        train_counter = Counter(train_X_cleaned)
        train_N = len(np.concatenate(train_X))

        tokens = list(train_counter.keys())
        counts = np.array(list(train_counter.values()))

        # smoothing
        probs = (counts+1)/(train_N+vocabulary_size)

        train_prob = {token:prob for token, prob in zip(tokens, probs) }

        sudo_prob = 1/(train_N+vocabulary_size)
        
        train_probs.update({label: train_prob})
        sudo_probs.update({label: sudo_prob})
    
    return train_probs, train_label_probs, sudo_probs


# In[188]:


train_probs, train_label_probs, train_sudo_probs = train_model(train_X, train_y, vocabulary_size, unknownwords=unknownwords)


# # validate model

# In[133]:


def predict(train_probs, train_label_probs, train_sudo_probs, test_X, unknownwords=None):
    '''
    predict using trained naive bayes model, and return label predictions;
    calculation is based on log base'''
    prediction_results = []
    for X in test_X:
        if unknownwords is None:
            words = X
        else:
            words = [word if word not in unknownwords else '<UNK>' for word in X]
        log_probs = {}
        for label in train_label_probs.keys():
            # get corresponding model
            label_prob = train_label_probs.get(label)
            word_prob = train_probs.get(label)
            sudo_prob = train_sudo_probs.get(label)
            # predict
            log_prob = [np.log(word_prob.get(word)) if word_prob.get(word) is not None else np.log(sudo_prob) for word in words]
            log_prob = sum(log_prob)+np.log(label_prob)
            log_probs.update({label: log_prob})
        prediction = sorted(log_probs.items(), key=lambda x:x[1],reverse=True)[0][0]
        prediction_results.append(prediction)
    return prediction_results


# # In[63]:


# def evaluate(test_prediction, test_y):
#     '''evaluate the prediction and return accuracy and prediction and recall for all classes'''
#     correct_prediction = 0
#     classes = set(test_y)
#     # for each class
#     correct_predicteds = {label:0 for label in classes}
#     num_of_labels = {label:0 for label in classes}
#     num_of_predictions = {label:0 for label in classes}
    
#     for prediction, label in zip(test_prediction, test_y):
#         num_of_predictions.update({prediction:num_of_predictions.get(prediction)+1})
#         num_of_labels.update({label:num_of_labels.get(label)+1})
#         if prediction == label:
#             correct_prediction+=1
#             correct_predicteds.update({label: correct_predicteds.get(label)+1})
#     accuracy = correct_prediction/len(test_prediction)
#     precisions = {label: float(correct_predicteds.get(label))/num_of_predictions.get(label) if num_of_predictions.get(label)!=0 else 0 for label in classes}
#     recalls = {label: float(correct_predicteds.get(label))/num_of_labels.get(label) if num_of_predictions.get(label)!=0 else 0 for label in classes}
#     return accuracy, precisions, recalls


# # In[59]:


# def accuracy_score(y, y_hat):
#     num_accurate = 0
#     for true_value, predicted_value in zip(y, y_hat):
#         if true_value == predicted_value:
#             num_accurate+=1
#     return num_accurate/len(y)


# # In[175]:


# def print_scores(results, output_path):
#     '''add results to the output file'''
#     with open(output_path, 'a') as f:
#         # acc
#         print('overall accuracy', file = f)
#         print(results[0], file = f)
#         print('precision for red', file = f)
#         print(results[1].get('RED'), file = f)
#         print('recall for red', file = f)
#         print(results[2].get('RED'), file = f)
#         print('precision for blue', file = f)
#         print(results[1].get('BLUE'), file = f)
#         print('recall for blue', file = f)
#         print(results[2].get('BLUE'), file = f)
#         print(file = f)


# # In[189]:


# valid_prediction = predict(train_probs, train_label_probs, train_sudo_probs, valid_X, unknownwords=unknownwords)


# # In[190]:


# accuracy_score(valid_prediction, valid_y)


# # In[191]:


# print(evaluate(valid_prediction, valid_y))


# # Test model

# In[41]:


test_prediction = predict(train_probs, train_label_probs, train_sudo_probs, test_X, unknownwords=unknownwords)


# In[ ]:


with open(os.path.join(input_path, output_name), 'w', encoding='utf-8') as f:
    [f.write(prediction_string+'\n') for prediction_string in test_prediction]

