from postings_handler import PostingsHandler
import utils
import numpy
from tweet_vectors_handler import TweetVectorsHandler


class Indexer:

    def __init__(self, config):
        self.total_tweet_lengths = 0
        self.number_of_tweets = 0
        self.config = config
        self.document_dict = {}  # {id : [0-timestamp, 1-referrals, 2-uniques, 3-max_tf, 4-length, 5- vector_pos]}
        self.inverted_idx = {}  # {term : [idf(term), address = (posting #, line #)]}
        self.entities_inverted_idx = {}  # {term : [idf(term), address = (posting #, line #)]}
        self.posting_handler = PostingsHandler(config, contains="terms")
        self.entities_posting_handler = PostingsHandler(config, contains="entities")
        self.tweet_vectors_handler = TweetVectorsHandler(config)
        self.referrals_counter = {}
        self.avg_tweet_length = 0
        self.min_timestamp = 2000000000
        self.max_timestamp = 0
        self.max_referrals = 0
        self.use_glove = config.use_glove

    def add_new_doc(self, document):
        """
        This function perform indexing process for a document object.
        Saves the information in an inverted index for all terms and entities, and builds a document dictionary
        for other important metadata. Also saves the average Glove vectors for each document in a different
        set of files.
        Saved information is captures via two dictionaries ('inverted index' and 'posting')
        :param document: a document need to be indexed.
        """
        self.number_of_tweets += 1

        if document.tweet_timestamp:
            self.max_timestamp = max(self.max_timestamp, document.tweet_timestamp)
            self.min_timestamp = min(self.min_timestamp, document.tweet_timestamp)

        entities_doc_dictionary = document.entities_doc_dictionary
        if entities_doc_dictionary:
            for entity in entities_doc_dictionary.keys():
                if entity not in self.entities_inverted_idx.keys():
                    self.entities_inverted_idx[entity] = [1, (-1, -1)]
                else:
                    self.entities_inverted_idx[entity][0] += 1
                self.entities_posting_handler.append_term(entity, document.tweet_id, entities_doc_dictionary[entity],
                                                          self.entities_inverted_idx)
        document_dictionary = document.term_doc_dictionary

        self.total_tweet_lengths += document.tweet_length
        self.document_dict[document.tweet_id] = [document.tweet_timestamp,
                                                 0,  # referrals
                                                 len(document_dictionary.keys()) + len(entities_doc_dictionary.keys()),
                                                 -1,
                                                 document.tweet_length,
                                                 None]

        if self.use_glove:
            tweet_vector = numpy.full(25, 0)

        max_tf = 0
        for term in document_dictionary.keys():
            # Update inverted index and posting
            frequency = document_dictionary[term]
            if frequency > max_tf:
                max_tf = frequency  # update max idf
            if self.use_glove and term in self.config.glove_dict.keys():
                tweet_vector = tweet_vector + (frequency/document.tweet_length) * self.config.glove_dict[term]
            if term.lower() not in self.inverted_idx.keys() and term.upper() not in self.inverted_idx.keys():
                self.inverted_idx[term] = [1, (-1, -1)]
            elif term.isupper() and term.lower() in self.inverted_idx.keys():
                self.inverted_idx[term.lower()][0] += 1
                term = term.lower()
            elif term.islower() and term.upper() in self.inverted_idx.keys():
                self.inverted_idx[term] = [self.inverted_idx[term.upper()][0] + 1,
                                           self.inverted_idx[term.upper()][1]]
                del self.inverted_idx[term.upper()]
                self.posting_handler.change_term_case(term.upper(), term)
            else:
                self.inverted_idx[term][0] += 1
            self.posting_handler.append_term(term, document.tweet_id, frequency, self.inverted_idx)

        if self.use_glove:
            self.tweet_vectors_handler.append_vector(document.tweet_id, tweet_vector, self.document_dict)

        self.document_dict[document.tweet_id][3] = max_tf
        # insert to referrals dictionary
        if document.referrals_ids:
            for referral in document.referrals_ids:
                if referral not in self.referrals_counter.keys():
                    self.referrals_counter[referral] = 1
                else:
                    self.referrals_counter[referral] += 1
                    self.max_referrals = max(self.referrals_counter[referral], self.max_referrals)

    def finish_indexing(self):
        """
        Dump the remaining buckets from the memory and save the final inverted index, tweets dictionary and all of the
        extra metadata to the disk.
        """
        self.avg_tweet_length = self.total_tweet_lengths / self.number_of_tweets
        self.posting_handler.finish_indexing(self.inverted_idx)
        self.__check_entities()
        self.inverted_idx.update(self.entities_inverted_idx)
        self.__update_referrals()

        if self.use_glove:
            self.tweet_vectors_handler.finish_indexing(self.document_dict)

        # KEEP THOSE LINES LAST
        self.__save_metadata()
        utils.save_obj(self.document_dict, "docDictionary")
        utils.save_obj(self.inverted_idx, "inverted_index")

    def __check_entities(self):
        """
        Clean out potential entities that were found, but did not have a second appearance in the corpus
        """
        for entity in self.entities_inverted_idx.keys():
            if self.entities_inverted_idx[entity][0] > 1:
                self.inverted_idx[entity] = self.entities_inverted_idx[entity]

        self.entities_posting_handler.finish_indexing(self.entities_inverted_idx)
        self.entities_inverted_idx.clear()

    def __update_referrals(self):
        """
        Update every tweet that was indexed to include the number of referrals that were encountered during the indexing
        of the corpus.
        """
        for doc_id in self.document_dict.keys():
            if doc_id in self.referrals_counter.keys():
                self.document_dict[doc_id][1] = self.referrals_counter[doc_id]

    def __save_metadata(self):
        """
        Add an entry including all important tweets metadata so it could be loaded for later calculations (ranking etc)
        """
        self.document_dict["metadata"] = {"minTimestamp": self.min_timestamp,
                                          "maxTimestamp": self.max_timestamp,
                                          "avgLength": self.avg_tweet_length,
                                          "maxReferrals": self.max_referrals,
                                          "tweet_vector_buckets": self.tweet_vectors_handler.last_bucket_index}


