#!/usr/bin/env python
# coding: utf-8

# In[6]:


from __future__ import division
import argparse
import pandas as pd
from collections import defaultdict
# useful stuff
import numpy as np
from scipy.special import expit
from sklearn.preprocessing import normalize
from nltk.tokenize import word_tokenize
import copy
import pickle
from tqdm import tqdm
# In[19]:
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

import nltk

nltk.download('punkt')

# In[75]:


import nltk

nltk.download('wordnet')

# # Step1. Preprocessing


# defining the function to remove punctuation
# library that contains punctuation
import string


def remove_punctuation(corpus):
    punctuationfree = []
    punctuationfree.extend(i for i in corpus if i not in string.punctuation)
    return punctuationfree


# In[57]:


# remove stopwords

# Stop words present in the library
stopwords = nltk.corpus.stopwords.words('english')


def remove_stopwords(text):
    output = [i for i in text if i not in stopwords]

    return output


# In[72]:


# Lemmatization
# It is also known as the text standardization step where the words are stemmed or diminished to their root/base form.
# For example, words like ‘programmer’, ‘programming, ‘program’ will be stemmed to ‘program’.

# It stems the word but makes sure that it does not lose its meaning.
# Lemmatization has a pre-defined dictionary that stores the context of words and checks the word in the dictionary while diminishing.

from nltk.stem import WordNetLemmatizer

# defining the object for Lemmatization
wordnet_lemmatizer = WordNetLemmatizer()


# defining the function for lemmatization
def lemmatizer(text):
    lemm_text = [wordnet_lemmatizer.lemmatize(word) for word in text]
    return lemm_text


# In[85]:


def text2sentences(path):
    # feel free to make a better tokenization/pre-processing
    sentences = []
    with open(path) as f:
        for l in f:
            sentences.append(l.lower().split())

    # remove punctuations
    sentences = [remove_punctuation(sent) for sent in sentences]

    # Remove numbers
    sentences = [list(filter(lambda x: x.isalpha(), sent)) for sent in sentences]

    # Lemmatization
    sentences = [lemmatizer(sent) for sent in sentences]

    return sentences


# In[131]:


# # Define all helper functions

# In[3]:


def loadPairs(path):
    data = pd.read_csv(path, delimiter='\t')
    pairs = zip(data['word1'], data['word2'], data['similarity'])
    return pairs


# In[223]:


import math


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


# In[357]:


def cosine_distance(vec1, vec2):
    assert vec1.shape == vec2.shape
    return np.dot(vec1, vec2) / (np.sqrt(np.dot(vec1, vec1)) * np.sqrt(np.dot(vec2, vec2)))


# # Step 2. Build the skip-gram model architecture

# In[182]:


# reference:https://aegis4048.github.io/optimize_computational_efficiency_of_skip-gram_with_negative_sampling

class mySkipGram:
    def __init__(self, sentences, nEmbed=100, negativeRate=5, winSize=5, minCount=5, learning=0.01):

        # Storing hyper parameters as class variables
        self.nEmbed = nEmbed
        self.negativeRate = negativeRate
        self.winSize = winSize
        self.minCount = minCount
        self.lr = learning

        self.vocab_dict = self.vocab_dict_generator(sentences)
        # word to ID mapping
        self.w2id = {key: values['word_index'] for key, values in self.vocab_dict.items()}
        self.trainset = sentences
        self.vocab = list(self.vocab_dict.keys())  # list of valid words
        self.word_freq = {values['word_index']: values['word_freq'] for key, values in self.vocab_dict.items()}

        # initialize input_embedding matrix and output_weight matrix

        # with a different initialization
        self.input_embedding = np.random.uniform(low=-0.5 / (self.nEmbed ** (3 / 4)),
                                                 high=0.5 / (self.nEmbed ** (3 / 4)),
                                                 size=(len(self.vocab_dict), self.nEmbed))

        # plain choice, all zeros
        # self.input_embedding = np.zeros([len(self.vocab_dict), self.nEmbed])
        self.output_weights = np.zeros([len(self.vocab_dict), self.nEmbed])
        self.G0 = np.zeros_like(self.input_embedding)
        self.G1 = np.zeros_like(self.output_weights)

        # create variables to keep track of model performance
        self.trainWords = 0
        self.accLoss = 0
        self.loss = []

        # create an 'unknown_vector' to represent word in test data that doesn't exist in our train set
        self.unknown_vector = np.random.normal(0, 1, (self.nEmbed,))

    def vocab_dict_generator(self, sentences):
        """generate a dictionary to store information for each single word

        Parameters
        ----------
        sentences: a list of sentences

        Returns
        -------
        vocab: a dictionary
        where key is unique word from corpus and value is a dictionary {'word_count': how many times a word has appeared in the entire train set,
        'word_freq': the frequency of a word, 'word_index': assigning an id to each word}
        """

        vocab = defaultdict(dict)
        vocab_words = ['int']
        vocab['int']['word_count'] = 0
        vocab_size = 0
        for sent_tokens in sentences:
            vocab_size += len(sent_tokens)
            for word in sent_tokens:
                if word not in vocab:
                    vocab[word]['word_count'] = 1
                    vocab_words.append(word)
                else:
                    vocab[word]['word_count'] += 1

        # remove words appearing fewer than min_count times
        low_freq_words = []
        for word in vocab:
            if vocab[word]['word_count'] < self.minCount:
                low_freq_words.append(word)
        for word in low_freq_words:
            vocab_size -= vocab[word]['word_count']
            del vocab[word]
            vocab_words.remove(word)
        sorted_vocab = []
        for word in vocab:
            sorted_vocab.append((word, vocab[word]['word_count']))
        sorted_vocab.sort(key=lambda tup: tup[1], reverse=True)
        for idx, word in enumerate(sorted_vocab):
            vocab[word[0]]['word_freq'] = vocab[word[0]]['word_count'] / vocab_size
            vocab[word[0]]['word_index'] = idx
        return vocab

    def negative_sampling(self, word_tuple, alpha=0.75):
        """samples negative words, ommitting word_tuple, Words that actually appear within the context window of the center word
            and generate ids of words that are randomly drawn from a noise distribution

        Parameters
        ----------
        word_tuple : tuple of {wIdx, ctxtId}
        wIdx is the index of center word, ctxtId is the index of context word

        alpha: a hyper-parameter that can be empircially tuned
        in the noise distribution — normalized frequency distribution of words raised to the power of α.

        Returns
        -------
        negativeIds: a dictionary with key as word, probability of being chosen as value
        representing words that doesn't appear within the context window of the centre word but exist in the corpus

        """
        word_freq_copy = copy.deepcopy(self.word_freq)
        # remove positive sample
        for id in word_tuple:
            word_freq_copy.pop(id)

        # generate noise distribution
        noise_dist = {key: val ** alpha for key, val in word_freq_copy.items()}
        Z = sum(noise_dist.values())
        noise_dist_normalized = {key: val / Z for key, val in noise_dist.items()}

        negativeIds = np.random.choice(list(noise_dist_normalized.keys()), size=self.negativeRate,
                                       p=list(noise_dist_normalized.values()))

        return negativeIds

    def loss_function(self, wordId, contextId, negativeIds):
        """ Returns the loss for the given word, its context and the negative samples"""

        l_sum = np.log(sigmoid(np.dot(self.input_embedding[wordId, :], self.output_weights[contextId, :])))

        for negativeId in negativeIds:
            l_sum *= np.log(sigmoid(-np.dot(self.input_embedding[negativeId, :], self.output_weights[contextId, :])))

        return l_sum

    def train(self):
        for counter, sentence in enumerate(tqdm(self.trainset)):
            sentence = list(filter(lambda word: word in self.vocab, sentence))

            for wpos, word in enumerate(sentence):
                wIdx = self.w2id[word]
                # dynamic window size, the winSize denotes the maximal window size. For each word in the corpus, a window size k' is randomly sampled uniformly from 1,,,,winSize
                winsize = np.random.randint(self.winSize) + 1
                start = max(0, wpos - winsize)
                end = min(wpos + winsize + 1, len(sentence))

                for context_word in sentence[start:end]:
                    ctxtId = self.w2id[context_word]
                    if ctxtId == wIdx: continue
                    negativeIds = self.negative_sampling({wIdx, ctxtId})
                    self.trainWord(wIdx, ctxtId, negativeIds)

                    # keep record of loss during training
                    self.accLoss += self.loss_function(wIdx, ctxtId, negativeIds)
                    self.trainWords += 1

            if (counter+1) % 100 == 0:
                print(' > training %d of %d' % (counter, len(self.trainset)))
                self.loss.append(self.accLoss / self.trainWords)
                self.trainWords = 0
                self.accLoss = 0

            if (counter + 1) % 1000 == 0:
                print(self.loss[-1])

    # Back propagation

    def trainWord(self, wordId, contextId, negativeIds):

        # positive_loss = np.log(1 / (1 + np.exp(-np.dot(self.W[wIdx], self.C[ctxId]))))
        # negative_loss = 0
        # for i in negativeIds:
        #     negative_loss += expit(-np.dot(self.W[wIdx], self.C[i]))  # remove - sign
        # self.accLoss -= (negative_loss + positive_loss)

        neg_sample = [(wordId, 1)]
        wv_h = self.input_embedding[contextId]
        # For each positive word-context pair (w,cpos),
        # K new negative samples are randomly drawn from a noise distribution.

        for neg_word in negativeIds:
            neg_sample.append((neg_word, 0))

        # Adagrad
        dh = np.zeros(self.nEmbed)

        for neg_w in neg_sample:
            target, label = neg_w[0], neg_w[1]

            wv_j = self.output_weights[target]
            dwjh = sigmoid(np.dot(wv_h, wv_j)) - label
            dwj = dwjh * wv_h
            self.G1[target] += np.power(dwj, 2)
            dwj /= (np.sqrt(self.G1[target]) + 1e-6)  # to avoid 0 in denominator
            assert dwj.shape == wv_j.shape
            dh += dwjh * wv_j
            # Update the output weight matrix
            self.output_weights[target] -= self.lr * dwj

        # Update the input embedding matrix
        self.G0[contextId] += np.power(dh, 2)
        dh /= np.sqrt(self.G0[contextId]) + 1e-6
        assert dh.shape == wv_h.shape
        self.input_embedding[contextId] -= self.lr * dh

    def save(self, path):
        pickle.dump(self, open(path, 'wb'))
        # with open(path, 'wb') as f:
        #
        #     np.save(f, self.input_embedding, allow_pickle=False, fix_imports=True)

    def similarity(self, word1, word2):
        """
        computes similiarity between the two words measured by consine distance. unknown words are mapped to one common vector
        :param word1:
        :param word2:
        :return: a float \in [0,1] indicating the similarity (the higher the more similar)
        """

        if word1 == word2:
            print('same word')
            return 1
        else:
            vec1 = self.input_embedding[self.w2id[word1]] if word1 in self.vocab else self.unknown_vector
            vec2 = self.input_embedding[self.w2id[word2]] if word2 in self.vocab else self.unknown_vector
            return np.dot(vec1, vec2) / (np.sqrt(np.dot(vec1, vec1)) * np.sqrt(np.dot(vec2, vec2)))

    @staticmethod
    def load(path):
        return pickle.load(open(path, 'rb'))


# In[4]:


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--text', help='path containing training data', required=True)
    parser.add_argument('--model', help='path to store/read model (when training/testing)', required=True)
    parser.add_argument('--test', help='enters test mode', action='store_true')

    opts = parser.parse_args()

    if not opts.test:
        sentences = text2sentences(opts.text)
        sg = mySkipGram(sentences, minCount=2)
        sg.train()
        sg.save(opts.model)

    else:
        print('enter test mode')
        pairs = loadPairs(opts.text)

        sg = mySkipGram.load(opts.model)
        predict_similar = []
        true_similar = []
        for a, b, true_score in pairs:
            # make sure this does not raise any exception, even if a or b are not in sg.vocab

            pred = sg.similarity(a, b)

            predict_similar.append(pred)
            true_similar.append(true_score)
        print('correlation result is: ', round(pd.Series(predict_similar).corr(pd.Series(true_similar)), 4))