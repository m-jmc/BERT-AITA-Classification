#! python3

#%% Importing

import re
import pyperclip
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import STOPWORDS
from collections import defaultdict

from sklearn.model_selection import train_test_split, StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report

import tokenization # This might not be needed using bt
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.layers import Dense, Input, Dropout, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
import tensorflow_hub as hub

# use pip install bert-for-tf2
from bert.tokenization import bert_tokenization as bt


#%% Data loading

"""
________________________________
Abbr - Def (classification)
________________________________
NTA - Not the Asshole (NTA)
ESH - Everyone Sucks Here (YTA)
YTA - You're the Asshole (YTA)
NAH - No Assholes Here (NTA)
________________________________
"""

comments = pd.read_csv("comment_df.csv")
sub = pd.read_csv("sub_df.csv")


# %% Comment Parsing, Finding the top voted result for the thread

comments = comments[comments.rply_score != 1] # remove mod post reply wtih vote of "1"
comments['outcome'] = comments['reply'].str.extract('(NTA|ESH|YTA|NAH|Nta)', expand = True) # Extract the sentiment

idx = comments.groupby(['id'])['rply_score'].transform(max) == comments['rply_score'] # find index of highest scoring comments
comments = comments[idx] # Create df of only highest scoring sentiment outcome for each sub
comments['outcome'] = comments['outcome'].str.upper() # Make sure the outcomes are all upper case
comments = comments.dropna(how='any',axis=0) # drop every row that has any na/null value

#%% Boolean values for a-hole check

comments['Ass'] = np.where(comments['outcome'].str.contains('ESH|YTA'), 1, 0)

# %% Check distribution of a-holes

ax = sns.countplot(comments['Ass'],label="Count",palette="colorblind")

# %% Title and submission text parsing

# removing META, UPDATE titled submissions and those which do not contain AITA or WIBTA
sub = sub[~sub['title'].str.contains('META|UPDATE', case=False)] 
sub = sub[sub['title'].str.contains('AITA|WIBTA', case=False)]

# %% Joining the submission df wtih the highest voted outcome response 

final = pd.merge(sub,comments[['id','Ass','outcome']], on='id', how = 'inner')

# %%
final.rename(columns = {'Ass':'target','title':'text','Unnamed: 0':'error'}, inplace = True)

# %% Splitting into training / testing sets using randomized 70/30 split

#X_train, X_test, y_train, y_test = train_test_split(final['text'], final['target'], test_size = 0.3, random_state = 0)
train, test = train_test_split(final, test_size = 0.4, random_state = 0)
submission = pd.read_csv("sample_submission.csv")



#%% Generate Meta Counts

# word_count
train['word_count'] = train['text'].apply(lambda x: len(str(x).split()))
test['word_count'] = test['text'].apply(lambda x: len(str(x).split()))

# unique_word_count
train['unique_word_count'] = train['text'].apply(lambda x: len(set(str(x).split())))
test['unique_word_count'] = test['text'].apply(lambda x: len(set(str(x).split())))


# mean_word_length
train['mean_word_length'] = train['text'].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
test['mean_word_length'] = test['text'].apply(lambda x: np.mean([len(w) for w in str(x).split()]))


#%% Meta Plots #################################

meta = ['word_count', 'unique_word_count', 'mean_word_length']
YTA = train['target'] == 1

fig, axes = plt.subplots(ncols=2, nrows=len(meta), figsize=(20, 50), dpi=100)

for i, feature in enumerate(meta):
    sns.distplot(train.loc[~YTA][feature], label='NTA', ax=axes[i][0], color='gray')
    sns.distplot(train.loc[YTA][feature], label='YTA', ax=axes[i][0], color='orange')

    sns.distplot(train[feature], label='Training', ax=axes[i][1], color='gray')
    sns.distplot(test[feature], label='Test', ax=axes[i][1], color='orange')
    
    for j in range(2):
        axes[i][j].set_xlabel('')
        axes[i][j].tick_params(axis='x', labelsize=12)
        axes[i][j].tick_params(axis='y', labelsize=12)
        axes[i][j].legend()
    
    axes[i][0].set_title(f'{feature} Target Distribution in Training Set', fontsize=13)
    axes[i][1].set_title(f'{feature} Training & Test Set Distribution', fontsize=13)

plt.savefig('DataDistribution.png')
plt.show()

#%% Grams #################################

def generate_ngrams(text, n_gram=1):
    token = [token for token in text.lower().split(' ') if token != '' if token not in STOPWORDS]
    ngrams = zip(*[token[i:] for i in range(n_gram)])
    return [' '.join(ngram) for ngram in ngrams]

N = 25


#%% Build the count #################################
# Unigrams
yta_unigrams = defaultdict(int)
nta_unigrams = defaultdict(int)

for term in train[YTA]['text']:
    for word in generate_ngrams(term):
        yta_unigrams[word] += 1
        
for term in train[~YTA]['text']:
    for word in generate_ngrams(term):
        nta_unigrams[word] += 1
        
df_yta_unigrams = pd.DataFrame(sorted(yta_unigrams.items(), key=lambda x: x[1])[::-1])
df_nta_unigrams = pd.DataFrame(sorted(nta_unigrams.items(), key=lambda x: x[1])[::-1])


# Bigrams
yta_bigrams = defaultdict(int)
nta_bigrams = defaultdict(int)

for term in train[YTA]['text']:
    for word in generate_ngrams(term, n_gram=2):
        yta_bigrams[word] += 1
        
for term in train[~YTA]['text']:
    for word in generate_ngrams(term, n_gram=2):
        nta_bigrams[word] += 1
        
df_yta_bigrams = pd.DataFrame(sorted(yta_bigrams.items(), key=lambda x: x[1])[::-1])
df_nta_bigrams = pd.DataFrame(sorted(nta_bigrams.items(), key=lambda x: x[1])[::-1])


#%% Build plot MONO #################################

fig, axes = plt.subplots(ncols=2, figsize=(18, 50), dpi=100)
plt.tight_layout()

sns.barplot(y=df_yta_unigrams[0].values[:N], x=df_yta_unigrams[1].values[:N], ax=axes[0], color='orange')
sns.barplot(y=df_nta_unigrams[0].values[:N], x=df_nta_unigrams[1].values[:N], ax=axes[1], color='gray')

for i in range(2):
    axes[i].spines['right'].set_visible(False)
    axes[i].set_xlabel('')
    axes[i].set_ylabel('')
    axes[i].tick_params(axis='x', labelsize=13)
    axes[i].tick_params(axis='y', labelsize=13)

axes[0].set_title(f'Top {N} most common unigrams in YTA', fontsize=15)
axes[1].set_title(f'Top {N} most common unigrams in NTA', fontsize=15)

plt.savefig('Monogram.png')
plt.show()


#%% Build plot BI #################################

fig, axes = plt.subplots(ncols=2, figsize=(18, 50), dpi=100)
plt.tight_layout()

sns.barplot(y=df_yta_bigrams[0].values[:N], x=df_yta_bigrams[1].values[:N], ax=axes[0], color='orange')
sns.barplot(y=df_nta_bigrams[0].values[:N], x=df_nta_bigrams[1].values[:N], ax=axes[1], color='gray')

for i in range(2):
    axes[i].spines['right'].set_visible(False)
    axes[i].set_xlabel('')
    axes[i].set_ylabel('')
    axes[i].tick_params(axis='x', labelsize=13)
    axes[i].tick_params(axis='y', labelsize=13)

axes[0].set_title(f'Top {N} most common bigrams for YTA', fontsize=15)
axes[1].set_title(f'Top {N} most common bigrams for NTA', fontsize=15)

plt.savefig('Bigrams.png')
plt.show()

#%% Classification Report #############################
#######################################################
#######################################################
#######################################################
#######################################################

class ClassificationReport(Callback):
    
    def __init__(self, train_data=(), validation_data=()):
        super(Callback, self).__init__()
        
        self.X_train, self.y_train = train_data
        self.train_precision_scores = []
        self.train_recall_scores = []
        self.train_f1_scores = []
        
        self.X_val, self.y_val = validation_data
        self.val_precision_scores = []
        self.val_recall_scores = []
        self.val_f1_scores = [] 
               
    def on_epoch_end(self, epoch, logs={}):
        train_predictions = np.round(self.model.predict(self.X_train, verbose=0))        
        train_precision = precision_score(self.y_train, train_predictions, average='macro')
        train_recall = recall_score(self.y_train, train_predictions, average='macro')
        train_f1 = f1_score(self.y_train, train_predictions, average='macro')
        self.train_precision_scores.append(train_precision)        
        self.train_recall_scores.append(train_recall)
        self.train_f1_scores.append(train_f1)
        
        val_predictions = np.round(self.model.predict(self.X_val, verbose=0))
        val_precision = precision_score(self.y_val, val_predictions, average='macro')
        val_recall = recall_score(self.y_val, val_predictions, average='macro')
        val_f1 = f1_score(self.y_val, val_predictions, average='macro')
        self.val_precision_scores.append(val_precision)        
        self.val_recall_scores.append(val_recall)        
        self.val_f1_scores.append(val_f1)
        
        print('\nEpoch: {} - Training Precision: {:.6} - Training Recall: {:.6} - Training F1: {:.6}'.format(epoch + 1, train_precision, train_recall, train_f1))
        print('Epoch: {} - Validation Precision: {:.6} - Validation Recall: {:.6} - Validation F1: {:.6}'.format(epoch + 1, val_precision, val_recall, val_f1))  


#%% Preprocess data and BERT Layer Download
# Load BERT downloaded from the TF hub
#             module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/2"
# Load CSV files containing training data
# Load tokenizer from the bert layer
# Encode the text into tokens, masks, and segment flags

module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/2"
bert_layer = hub.KerasLayer(module_url, trainable=True)

#%% Cross Validation
SEED = 333
K = 2
skf = StratifiedKFold(n_splits=K, random_state=SEED, shuffle=True)

#%% Oversample YTA responses in our training set, 
# Determine size of the NTA class

train_save = train
max_size = train['target'].value_counts().max()

oversample = [train]
for class_index, group in train.groupby('target'):
    oversample.append(group.sample(max_size-len(group), replace=True))
train = pd.concat(oversample, ignore_index=True) #concat of the oversample causes duplicate indexes, ignore_index will reindex the df key for k-fold split


#%%

YTA = train['target'] == 1
print('Training Set Shape = {}'.format(train.shape))
print('Training Set Unique target Count = {}'.format(train['target'].nunique()))
print('Training Set Target Rate (YTA) {}/{} (NTA)'.format(train[YTA]['target'].count(), train[~YTA]['target'].count()))

for fold, (trn_idx, val_idx) in enumerate(skf.split(train['text'], train['target']), 1):
    print('\nFold {} Training Set Shape = {} - Validation Set Shape = {}'.format(fold, train.loc[trn_idx, 'text'].shape, train.loc[val_idx, 'text'].shape))
    print('Fold {} Training Set Unique keyword Count = {} - Validation Set Unique keyword Count = {}'.format(fold, train.loc[trn_idx, 'outcome'].nunique(), train.loc[val_idx, 'outcome'].nunique()))  



#%% BERT ##############################################
#######################################################
#######################################################
#######################################################
#######################################################


class AssholeDetector:
    
    def __init__(self, bert_layer, max_seq_length=128, lr=0.0001, epochs=3, batch_size=32):
        
        # BERT and Tokenization params
        self.bert_layer = bert_layer
        
        self.max_seq_length = max_seq_length        
        vocab_file = self.bert_layer.resolved_object.vocab_file.asset_path.numpy()
        do_lower_case = self.bert_layer.resolved_object.do_lower_case.numpy()
        self.tokenizer = bt.FullTokenizer(vocab_file, do_lower_case)
        
        # Learning control params
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        
        self.models = []
        self.scores = {}
        
        
    def encode(self, texts):
                
        all_tokens = []
        all_masks = []
        all_segments = []

        # Standard Text tokenization 
        for text in texts:
            text = self.tokenizer.tokenize(text)
            text = text[:self.max_seq_length - 2]
            input_sequence = ['[CLS]'] + text + ['[SEP]']
            pad_len = self.max_seq_length - len(input_sequence)

            tokens = self.tokenizer.convert_tokens_to_ids(input_sequence)
            tokens += [0] * pad_len
            pad_masks = [1] * len(input_sequence) + [0] * pad_len
            segment_ids = [0] * self.max_seq_length

            all_tokens.append(tokens)
            all_masks.append(pad_masks)
            all_segments.append(segment_ids)

        return np.array(all_tokens), np.array(all_masks), np.array(all_segments)
    
    
    def build_model(self):
        
        input_word_ids = Input(shape=(self.max_seq_length,), dtype=tf.int32, name='input_word_ids')
        input_mask = Input(shape=(self.max_seq_length,), dtype=tf.int32, name='input_mask')
        segment_ids = Input(shape=(self.max_seq_length,), dtype=tf.int32, name='segment_ids')    
        
        pooled_output, sequence_output = self.bert_layer([input_word_ids, input_mask, segment_ids])   
        clf_output = sequence_output[:, 0, :]
        out = Dense(1, activation='sigmoid')(clf_output)
        
        model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
        optimizer = SGD(learning_rate=self.lr, momentum=0.8)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        
        return model
    
    
    def train(self, X):
        
        for fold, (trn_idx, val_idx) in enumerate(skf.split(X['text'], X['target'])):
            
            print('\nFold {}\n'.format(fold))

            X_trn_encoded = self.encode(X.loc[trn_idx, 'text'].str.lower())
            y_trn = X.loc[trn_idx, 'target']
            X_val_encoded = self.encode(X.loc[val_idx, 'text'].str.lower())
            y_val = X.loc[val_idx, 'target']
            
            # Callbacks
            metrics = ClassificationReport(train_data=(X_trn_encoded, y_trn), validation_data=(X_val_encoded, y_val))
            
            # Model
            model = self.build_model()        
            model.fit(X_trn_encoded, y_trn, validation_data=(X_val_encoded, y_val), callbacks=[metrics], epochs=self.epochs, batch_size=self.batch_size)
            
            self.models.append(model)
            self.scores[fold] = {
                'train': {
                    'precision': metrics.train_precision_scores,
                    'recall': metrics.train_recall_scores,
                    'f1': metrics.train_f1_scores                    
                },
                'validation': {
                    'precision': metrics.val_precision_scores,
                    'recall': metrics.val_recall_scores,
                    'f1': metrics.val_f1_scores                    
                }
            }
                    
                
    def plot_learning_curve(self):
        
        fig, axes = plt.subplots(nrows=K, ncols=2, figsize=(20, K * 6), dpi=100)
    
        for i in range(K):
            
            # Classification Report curve
            sns.lineplot(x=np.arange(1, self.epochs + 1), y=clf.models[i].history.history['val_accuracy'], ax=axes[i][0], label='val_accuracy')
            sns.lineplot(x=np.arange(1, self.epochs + 1), y=clf.scores[i]['validation']['precision'], ax=axes[i][0], label='val_precision')
            sns.lineplot(x=np.arange(1, self.epochs + 1), y=clf.scores[i]['validation']['recall'], ax=axes[i][0], label='val_recall')
            sns.lineplot(x=np.arange(1, self.epochs + 1), y=clf.scores[i]['validation']['f1'], ax=axes[i][0], label='val_f1')        

            axes[i][0].legend() 
            axes[i][0].set_title('Fold {} Validation Classification Report'.format(i), fontsize=14)

            # Loss curve
            sns.lineplot(x=np.arange(1, self.epochs + 1), y=clf.models[0].history.history['loss'], ax=axes[i][1], label='train_loss')
            sns.lineplot(x=np.arange(1, self.epochs + 1), y=clf.models[0].history.history['val_loss'], ax=axes[i][1], label='val_loss')

            axes[i][1].legend() 
            axes[i][1].set_title('Fold {} Train / Validation Loss'.format(i), fontsize=14)

            for j in range(2):
                axes[i][j].set_xlabel('Epoch', size=12)
                axes[i][j].tick_params(axis='x', labelsize=12)
                axes[i][j].tick_params(axis='y', labelsize=12)

        plt.savefig('Learning_Curve.png')
        plt.show()
        
        
    def predict(self, X):
        
        X_test_encoded = self.encode(X['text'].str.lower())
        y_pred = np.zeros((X_test_encoded[0].shape[0], 1))

        for model in self.models:
            y_pred += model.predict(X_test_encoded) / len(self.models)

        return y_pred



#%% Asshole Detector

clf = AssholeDetector(bert_layer, max_seq_length=16, lr=0.0001, epochs=12, batch_size=32)

clf.train(train)

#%% Plot Learning Curves

clf.plot_learning_curve()


#%% Predict on test set, add to Test DF
y_pred = clf.predict(test)
test['predict'] = y_pred.round().astype(int)


#%% Confusion Matrix

con_matrix = confusion_matrix(y_true=test['target'], y_pred=test['predict'])


# %% Pretty Confusion Matrix

fig, ax = plt.subplots(figsize=(3.5, 3.5))
ax.matshow(con_matrix, cmap=plt.cm.gist_heat, alpha=0.3)
for i in range(con_matrix.shape[0]):
    for j in range(con_matrix.shape[1]):
        ax.text(x=j, y=i,s=con_matrix[i, j], va='center', ha='center', size='large')
 
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.savefig('confusion.png')
plt.show()

# %% Classification Report

print(classification_report(test['target'],test['predict']))

# %% Print classification report to image

plt.rc('figure', figsize=(2, 2))
plt.text(0.01, 0.05, str(classification_report(test['target'],test['predict'])), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
plt.axis('off')
plt.tight_layout()
plt.savefig('classification.png')

