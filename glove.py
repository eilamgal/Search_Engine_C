import numpy as np
from nltk.corpus import stopwords
import utils
from numpy import dot
from numpy.linalg import norm


def create_search_dict(vocabulary_dict, embedding_dict=None):
    if not embedding_dict:
        embedding_dict = utils.load_obj("embedding_dict")
    new_embedding_dict = {}
    for word in embedding_dict.keys():
        if word in vocabulary_dict.keys():
            new_embedding_dict[word] = embedding_dict[word]
    utils.save_obj(new_embedding_dict, "new_embedding_dict")


def find_closest_embeddings(embedding, number_of_words, search_in_dict):
    return sorted(search_in_dict.keys(), key=lambda word: norm(embedding, search_in_dict[word]))[1:number_of_words]


def cosine(embedding, number_of_words, search_in_dict):
    return sorted(search_in_dict.keys(), key=lambda word: dot(embedding, search_in_dict[word])/(norm(embedding)*norm(search_in_dict[word])))[1:number_of_words]


class Glove:
    def __init__(self, embeddings_path, vocabulary_dict=None):
        stopwords_set = stopwords.words('english')
        self.embeddings_dict = {}
        with open(embeddings_path, 'r', encoding="utf-8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                if not word.isalpha() or len(word) < 3 or word in stopwords_set:
                    continue
                vector = np.asarray(values[1:], "float32")
                self.embeddings_dict[word] = vector
        #if vocabulary_dict:
         #   self.dict_adapter(embeddings_dict, vocabulary_dict)
        #utils.save_obj(embeddings_dict, "embedding_dict")

    def expand_query(self, query_as_list):
        new_query_list = []
        embedding_dict = utils.load_obj("embedding_dict")
        new_embedding_dict = utils.load_obj("new_embedding_dict")
        for term in query_as_list:
            if term in embedding_dict.keys():
                new_query_list.extend(find_closest_embeddings(embedding_dict[term], 4, new_embedding_dict))
        return new_query_list



