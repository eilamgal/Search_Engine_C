# from ranker import Ranker
# import utils
#
#
# # DO NOT MODIFY CLASS NAME
# class Searcher:
#     # DO NOT MODIFY THIS SIGNATURE
#     # You can change the internal implementation as you see fit. The model
#     # parameter allows you to pass in a precomputed model that is already in
#     # memory for the searcher to use such as LSI, LDA, Word2vec models.
#     # MAKE SURE YOU DON'T LOAD A MODEL INTO MEMORY HERE AS THIS IS RUN AT QUERY TIME.
#     def __init__(self, parser, indexer, model=None):
#         self._parser = parser
#         self._indexer = indexer
#         self._ranker = Ranker()
#         self._model = model
#
#     # DO NOT MODIFY THIS SIGNATURE
#     # You can change the internal implementation as you see fit.
#     def search(self, query, k=None):
#         """
#         Executes a query over an existing index and returns the number of
#         relevant docs and an ordered list of search results (tweet ids).
#         Input:
#             query - string.
#             k - number of top results to return, default to everything.
#         Output:
#             A tuple containing the number of relevant search results, and
#             a list of tweet_ids where the first element is the most relavant
#             and the last is the least relevant result.
#         """
#         query_as_list = self._parser.parse_sentence(query)
#
#         relevant_docs = self._relevant_docs_from_posting(query_as_list)
#         n_relevant = len(relevant_docs)
#         ranked_doc_ids = Ranker.rank_relevant_docs(relevant_docs)
#         return n_relevant, ranked_doc_ids
#
#     # feel free to change the signature and/or implementation of this function
#     # or drop altogether.
#     def _relevant_docs_from_posting(self, query_as_list):
#         """
#         This function loads the posting list and count the amount of relevant documents per term.
#         :param query_as_list: parsed query tokens
#         :return: dictionary of relevant documents mapping doc_id to document frequency.
#         """
#         relevant_docs = {}
#         for term in query_as_list:
#             posting_list = self._indexer.get_term_posting_list(term)
#             for doc_id, tf in posting_list:
#                 df = relevant_docs.get(doc_id, 0)
#                 relevant_docs[doc_id] = df + 1
#         return relevant_docs


from parser_module import Parse
from ranker import Ranker
import utils
import math
from numpy import dot
from numpy.linalg import norm
import numpy as np
EXPANSION_THRESHOLD = 10000000
EXPANSION_SIZE = 3


def bm25(corpus_term_frequency, tweet_term_frequency, avg_tweet_length, tweet_length, corpus_size=10000000, k=1.4,
         b=0.75, base=10):
    avg_tweet_length = 26.0 if avg_tweet_length == 0 else avg_tweet_length

    return ((tweet_term_frequency * (1 + k)) / (
                tweet_term_frequency + k * (1 - b + b * (tweet_length / avg_tweet_length)))) * math.log(
        (corpus_size + 1) / corpus_term_frequency, base)


def cosine(vector1, vector2):
    if vector1 is None or vector2 is None:
        return 0
    if norm(vector1) == 0 or norm(vector2) == 0:
        return 0
    return dot(vector1, vector2)/((norm(vector1))*(norm(vector2)))


def euclidean_distance(vector1, vector2):
    return np.norn(vector1, vector2)


class Searcher:

    def __init__(self, inverted_index, tweet_dict, config=None):
        """
        :param inverted_index: dictionary of inverted index
        """
        self.parser = Parse()
        self.ranker = Ranker()
        self.inverted_index = inverted_index
        self.tweet_dict = tweet_dict
        self.avg_tweet_length = tweet_dict["metadata"]["avgLength"]
        self.max_referrals = tweet_dict["metadata"]["maxReferrals"]
        self.min_timestamp = tweet_dict["metadata"]["minTimestamp"]
        self.max_timestamp = tweet_dict["metadata"]["maxTimestamp"]
        self.config = config
        self.use_glove = self.config.use_glove
        self.use_thesaurus = self.config.use_thesaurus
        if self.use_glove:
            self.glove_dict = self.config.glove_dict
        if self.use_thesaurus:
            self.thesaurus = self.config.thesaurus

    def relevant_docs_from_posting(self, query):
        """
        This function loads the posting list and count the amount of relevant documents per term.
        :param query: query
        :return: dictionary of relevant documents.
        """
        if self.use_thesaurus:
            query.extend(self.query_expansion(query))
        query_vector = None
        if self.use_glove:
            query_vector = self.get_query_vector(query)

        relevant_docs = {}  # {tweet_id ID : [0-glove score(some agabric distance), 1-BM25, 2-retweet score, 3-time score(more relevant -> better score)]}
        # put each term in his bucket for less disk reads
        buckets = {}
        self.get_buckets_dictionary(buckets, query)
        # for each bucket read and go over all the posting list of terms in this bucket
        for bucket in buckets.keys():
            posting = utils.load_obj("bucket" + str(bucket))
            for term in buckets[bucket]:
                if term not in self.inverted_index.keys():
                    if term.islower() and term.upper() in self.inverted_index.keys():
                        term = term.upper()
                    elif term.isupper() and term.lower() in self.inverted_index.keys():
                        term = term.lower()
                    else:
                        continue
                posting_doc = posting[self.inverted_index[term][1][1]]
                for doc_tuple in posting_doc:
                    tweet_id = doc_tuple[0]
                    # all the meta data on tweet we need for the ranking functions
                    term_freq = self.inverted_index[term][0]
                    tweet_timestamp = self.tweet_dict[tweet_id][0]
                    tweet_referrals = self.tweet_dict[tweet_id][1]
                    tweet_length = self.tweet_dict[tweet_id][4]
                    tweet_vector = self.tweet_dict[tweet_id][5]  # None if not using Glove
                    if tweet_id not in relevant_docs.keys():
                        # if the tweet_id not in relevant tweet_id so we need to insert all the ranking data else just adding
                        # the BM25 of the term
                        relevant_docs[tweet_id] = [cosine(query_vector, tweet_vector),  # cosine similarity
                                              bm25(corpus_term_frequency=term_freq, tweet_term_frequency=doc_tuple[1],
                                                   avg_tweet_length=self.avg_tweet_length,
                                                   tweet_length=tweet_length),  # BM25
                                              (tweet_referrals/self.max_referrals),  # Referrals count
                                              ((tweet_timestamp - self.min_timestamp)/(self.max_timestamp - self.min_timestamp))]  # Timestamp
                    else:
                        relevant_docs[tweet_id][1] += bm25(term_freq, doc_tuple[1], self.avg_tweet_length,
                                                      tweet_length)
        return relevant_docs

    def get_buckets_dictionary(self, buckets, query):
        for term in query:
            if term not in self.inverted_index.keys():
                if term.islower() and term.upper() in self.inverted_index.keys():
                    term = term.upper()
                elif term.isupper() and term.lower() in self.inverted_index.keys():
                    term = term.lower()
                else:
                    continue
            if self.inverted_index[term][1][0] not in buckets.keys():
                buckets[self.inverted_index[term][1][0]] = [term]
            else:
                buckets[self.inverted_index[term][1][0]].append(term)

    def get_query_vector(self, query):
        glove_dict = self.glove_dict
        query_vector = np.full(25, 0)
        for term in query:
            if term.lower() in glove_dict.keys():
                query_vector = query_vector + (1 / len(query)) * glove_dict[term.lower()]
            elif term.upper() in glove_dict.keys():
                query_vector = query_vector + (1 / len(query)) * glove_dict[term.upper()]
            elif term in glove_dict.keys():
                query_vector = query_vector + (1 / len(query)) * glove_dict[term.lower]
        return query_vector

    def query_expansion(self, query):
        expansion_terms = []
        # TODO: we need to decide if we want to clean the expansion terms
        for term in query:
            in_index, term = self.__query_term_in_index_check(term)
            if not in_index or self.inverted_index[term][0] < EXPANSION_THRESHOLD:
                expansion_terms.extend(list(self.thesaurus.synonyms(term, fileid="simN.lsp"))[0:EXPANSION_SIZE])
            """    
            if term not in self.inverted_index.keys():
                if term.islower() and term.upper() in self.inverted_index.keys():
                    term = term.upper()
                elif term.isupper() and term.lower() in self.inverted_index.keys():
                    term = term.lower()
                else:
                    expansion_terms.extend(list(self.thesaurus.synonyms(term, fileid="simN.lsp"))[0:EXPANSION_SIZE])
                    continue
            if self.inverted_index[term][0] < EXPANSION_THRESHOLD:
                expansion_terms.extend(list(self.thesaurus.synonyms(term, fileid="simN.lsp"))[0:EXPANSION_SIZE])
            """
        return expansion_terms

    def __query_term_in_index_check(self, term):
        in_index = True
        term_in_index = term
        if term not in self.inverted_index.keys():
            if term.islower() and term.upper() in self.inverted_index.keys():
                term = term.upper()
            elif term.isupper() and term.lower() in self.inverted_index.keys():
                term = term.lower()
            else:
                in_index = False
        return in_index, term_in_index






