import numpy as np
from bs4 import BeautifulSoup
import nltk
import random
import pickle

def lang_model(corpus):
    
    ngram = {}
    for review in corpus:
        s = review.text.lower()
        tokens = nltk.tokenize.word_tokenize(s)
        
        for i in range(len(tokens) - 2):
            k = (tokens[i], nltk.pos_tag([tokens[i+1]])[0][1], tokens[i+2])
            if k not in ngram:
                ngram[k] = []
            ngram[k].append(tokens[i+1])
    
    # turn each array of middle-words into a probability vector
    ngram_prob = {}
    for k, words in ngram.items():

        if len(set(words)) > 1:
            d = {}
            n = 0
            for w in words:
                if w not in d:
                    d[w] = 0
                d[w] += 1
                n += 1
            for w, c in d.items():
                d[w] = float(c) / n
            ngram_prob[k] = d
            
    return ngram_prob


def random_sample(d_probs):
    r = random.random()
    cumulative = 0
    for w, p in d_probs.items():
        cumulative += p
        if r < cumulative:
            return w


def test_spinner(reviews, ngram):
    
    chosen_prob = 1
    s = reviews.lower()
    
    print("Original Text:", s)
    tokens = nltk.tokenize.word_tokenize(s)
    for i in range(len(tokens) - 2):
        if random.random() < chosen_prob:
            k = (tokens[i], nltk.pos_tag([tokens[i+1]])[0][1], tokens[i+2])
            if k in ngram:
                w = random_sample(ngram[k])
                tokens[i+1] = w
    print("Updated Text:")
    print(" ".join(tokens).replace(" .", ".").replace(" '", "'").replace(" ,", ",").replace("$ ", "$").replace(" !", "!"))
    return " ".join(tokens).replace(" .", ".").replace(" '", "'").replace(" ,", ",").replace("$ ", "$").replace(" !", "!")
    
    
    
def train_model():
    positive_reviews = BeautifulSoup(open('electronics/positive.review').read(), "lxml")
    reviews = positive_reviews.findAll('review_text')
    ngram = lang_model(reviews)
    return ngram
    


#with open('article_spinner_model.pickle', 'wb') as handle:
#    pickle.dump(train_model(), handle)
    