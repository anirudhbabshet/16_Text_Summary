from math import ceil
from re import M
import string

import numpy as np
import networkx as nx

from nltk import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance


nltk_stopwords = stopwords.words('english')
adhoc_stopwords = ['in', 'also']
final_stopwords = []
final_stopwords = nltk_stopwords + adhoc_stopwords


def sentences_tokenizer(text):
    sentences = []
    orig_sentences = []
    text_sentences = sent_tokenize(text)
    for sentence in text_sentences:
        orig_sentences.append(sentence)
        sentences.append(clean_text(word_tokenize(sentence)))
    return sentences, orig_sentences

def clean_text(text):
    text = [i.lower() for i in text if i not in string.punctuation]
    text = [i for i in text if i not in final_stopwords]
    return text
        
def sentence_similarity(sent1, sent2):
    all_words = list(set(sent1 + sent2))
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
    # build the vector for the first sentence
    for w in sent1:
        vector1[all_words.index(w)] += 1
    # build the vector for the second sentence
    for w in sent2:
        vector2[all_words.index(w)] += 1
    return 1 - cosine_distance(vector1, vector2)
 
def build_similarity_matrix(sentences):
    # Create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2: #ignore if both are same sentences
                continue 
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2])
    return similarity_matrix

def generate_summary(text, max_sentence = 0):
    summarize_text_short = ""
    summarize_text_full = ""
    summarize_text =[]
    # Step 1 - Text Preprocess
    sentences , orig_sentences = sentences_tokenizer(text)
    if max_sentence == 0:
        max_sentence = ceil(len(sentences)*0.4)
    print("Total Sentences Length: "+str(len(sentences)))
    print("Summary Sentences Length: "+str(max_sentence))

    # Step 2 - Generate Similary Martix across sentences
    sentence_similarity_martix = build_similarity_matrix(sentences)

    # Step 3 - Rank sentences in similarity martix
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
    scores = nx.pagerank(sentence_similarity_graph, max_iter= 1000, tol=0.0001)

    # Step 4 - Sort the rank and pick top sentences
    ranked_sentence_full = sorted(((scores[i],s) for i,s in enumerate(orig_sentences)), reverse=True)    
    ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)   
    
    # Step 5 - Offcourse, output the summarize texr
    for line in orig_sentences:
        for tpl in ranked_sentence_full[0:max_sentence]:
            if  line == tpl[1]:
                summarize_text_full = summarize_text_full + str(line)+" "
                
    for line in sentences:
        for tpl in ranked_sentence[0:max_sentence]:
            if  line == tpl[1]:
                summarize_text_short = summarize_text_short + " ".join(line) + "."
    reduction = max_sentence*100/len(sentences)
    
    return summarize_text_short, summarize_text_full, reduction

if __name__ == "__main__":
    print("Running Main")
    print(generate_summary("Natural language processing (NLP) is an interdisciplinary subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language, in particular how to program computers to process and analyze large amounts of natural language data. The goal is a computer capable of understanding the contents of documents, including the contextual nuances of the language within them. The technology can then accurately extract information and insights contained in the documents as well as categorize and organize the documents themselves.Challenges in natural language processing frequently involve speech recognition, natural-language understanding, and natural-language generation. "))
