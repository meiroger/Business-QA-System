# import libraries
import os
import nltk
from nltk.tokenize import word_tokenize,sent_tokenize,RegexpTokenizer
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import spacy
import gensim
from gensim import corpora
from gensim.summarization.bm25 import *
from nltk.corpus import wordnet as wn
from rake_nltk import Rake
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print("Setting up corpus and data...")

# initializing stop words 
stop_words = set(stopwords.words('english'))

# file paths
path_13 = './2013'
path_14 = './2014'
path_train = './training/'

# load training set
text_2013 = []
text_2014 = []

# reading 2013 texts
for file in os.listdir(path_13):
    text_2013.append(open(os.path.join(path_13,file),'rb').read().decode("utf-8",errors="replace"))
    
# reading 2014 texts
for file in os.listdir(path_14):
    text_2014.append(open(os.path.join(path_14,file),'rb').read().decode("utf-8",errors="replace"))
    
# Combining both 2013 and 2014 texts into one list
all_text = text_2013 + text_2014

# word tokenizing all documents in corpus:
def tokenize_texts(all_text):
    '''
    input:
        all_text - list of str: list of all documents
    return:
        tokenized_texts - list of list of str: list of word tokenized documents
    '''
    tokenized_texts = []
    for text in all_text:
        tokenized_texts.append(word_tokenize(text))
    return tokenized_texts
        
tokenized_texts = tokenize_texts(all_text)

# creating document retrieval hashmap
def create_hashmap(all_text):    
    hashmap = {}
    '''
    input:
        all_text - list of str: list of all documents
    return:
        hashmap - dict: dictionary of word, document index pairs that map a word to documents containing that word
    '''
    for i in range(len(all_text)):
        words_in_doc = set(word_tokenize(all_text[i].lower()))
        words_in_doc = [word for word in words_in_doc if word not in stop_words]
        for word in words_in_doc:
            if not hashmap.get(word):                
                hashmap[word] = [i]
            else:
                hashmap[word].append(i)
    return hashmap

hashmap = create_hashmap(all_text)

# question type classification
def question_class(q):
    '''
    input:
        q - str: question to be asked
    return:
        classification - list of str: classifications of types of answers expected as a spacy named entity
    '''
    q = word_tokenize(q.lower())
    classification = []
    if ("who" == q[0]) or ("whom" == q[0]):
        classification = ['PERSON']
    elif ("where") == q[0]:
        classification = ['LOC','GPE']
    elif ("when" == q[0]) or ("what" == q[0] and q[1] in ['date','time']):
        classification = ["TIME","DATE"]
    elif ("how" == q[0]) and (q[1] in ['few','great','little','many','much']):
        classification = ['QUANTITY','MONEY','PERCENT']    
    elif ("which" == q[0]) and (q[1] in ['company','companies']):
        classification = ['ORG']
    elif ("what" == q[0]) and (q[1] in ['%','percent','percents','percentage','percentages']):
        classification = ['PERCENT']
    # answers are nouns
    elif "what" == q[0]:
        classification = ['NN']
    return classification

def keyword_extract(q):
    '''
    input:
        q - str: question to be asked
    return:
        filtered_result - list of str: keywords found in question
    '''
    result = []
    
    # extract keywords
    r = Rake()
    r.extract_keywords_from_text(q)
    keywords = set(r.get_ranked_phrases())
    # if more than one word in a string, split the string
    for kw in keywords:
        if " " in kw:
            split_kw = kw.split()
            for word in split_kw:
                result.append(word)
        else:
            result.append(kw)
    
    # remove verbs
    filtered_result = []
    tags = nltk.pos_tag(word_tokenize(q))
    for tag in tags:
        if 'VB' not in tag[1] or tag[1] == 'VBG':
            if tag[0].lower() in result:
                filtered_result.append(tag[0].lower())
     
    return filtered_result

def doc_select(keywords,hashmap):
    '''
    input:
        keywords - list of str: keywords found in question
        hashmap - dict: dictionary of word, document index pairs that map a word to documents containing that word
    return:
        docs - list of int: indices of documents to be used for answer
    '''
    docs = []
    for k in keywords:
        try:
            docs += hashmap[k]
        except:
            continue
    return list(set(docs))

def okapi_scoring(q,tokenized_texts,hashmap):
    '''
    input:
        q - str: question to score each document against
        tokenized_texts - list of list of str: list of word tokenized documents to be scored
        hashmap - dict: dictionary of word, document index pairs that map a word to documents containing that word
    return:
        best_docs - list of int: list of indices of the 5 most relevant document to the question based on okapi scores
    '''
    # setup
    keywords = keyword_extract(q)
    docs = doc_select(keywords,hashmap)
    filtered_texts = [tokenized_texts[i] for i in docs]
    # scoring
    bm25 = BM25(filtered_texts)
    query = word_tokenize(q)
    scores = bm25.get_scores(query)
    best_docs = []
    
    for idx in list(np.argsort(scores)[-5:]):
        best_docs.append(docs[idx])

    return best_docs

def sentence_select(q,docs):
    '''
    input:
        q - str: question to score each sentence against
        docs - list of str: list of documents to select sentence from
    return:
        answers - list of str: most relevant answers to query
 
    '''
    best_sent = ""
    best_cs = 0
    
    q_type = question_class(q)
    for doc in docs:
        # sentence tokenize document
        sentences = sent_tokenize(doc)
        filtered_sentences = []

        # keep only sentences with wanted entities
        nlp = spacy.load('en')
        if len(q_type) and q_type[0] != 'NN':
            for sent in sentences:
                doc_obj = nlp(sent)
                for ent in doc_obj.ents:
                    if ent.label_ in q_type:
                        filtered_sentences.append(sent)
        else:
            filtered_sentences = sentences

        # compute cosine similarity of question with each sentence
        tfidf_vectorizer = TfidfVectorizer(analyzer="char")
        for sent in filtered_sentences:
            compared_docs = (q,sent) 
            tfidf_matrix = tfidf_vectorizer.fit_transform(compared_docs)
            cs = cosine_similarity(tfidf_matrix[0:1],tfidf_matrix)
            if cs[0][1] > best_cs:
                best_cs = cs[0][1]
                best_sent = sent
                
    # extract answers from best sentence
    answers = []
    if len(q_type) and q_type[0] != 'NN':
        best_obj = nlp(best_sent)
        for ent in best_obj.ents:
            if ent.label_ in q_type:
                answers.append(ent.text)
    elif q_type[0] == 'NN':
        tags = nltk.pos_tag(word_tokenize(best_sent))
        for tag in tags:
            if 'NN' in tag[1]:
                answers.append(tag[0])
    else:
        answers.append(best_sent)
        
    # sentence is also returned for debugging purposes
    return answers, best_sent


print("Done with setup!")

while True:
    q = input('What is your question? (or type \'exit\' to exit): ')
    if q == 'exit':
        print("Bye!")
        break
    print("Thinking...")
    docs = []
    for idx in okapi_scoring(q,tokenized_texts,hashmap):
        docs.append(all_text[idx])
        
    answer,sent = sentence_select(q,docs)
    print("answer: ", answer)
    print("extracted from sentence: ", sent)
    
