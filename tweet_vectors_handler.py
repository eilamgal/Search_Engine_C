from bucket import Bucket
import utils
import time
MAX_SIZE = 1000000


class TweetVectorsHandler:
    """
    Similar to the postings_handler, but simpler - builds a dictionary for tweets and their average Glove vectors
    and dumps them to the disk once MAX_SIZE is reached in memory.
    """
    def __init__(self, config):
        self.bucket = Bucket()
        self.last_bucket_index = 0
        self.size = 0
        self.config = config

    def __flush_bucket(self, doc_dictionary):  # just write to the desk and clean the bucket
        new_posting = []
        for doc_id in self.bucket.get_dict_terms():
            new_posting.append(self.bucket.get_term_posting(doc_id)[0])
            doc_dictionary[doc_id][5] = (self.last_bucket_index, len(new_posting) - 1)

        self.size = 0
        self.bucket.clean_bucket()
        start_time = time.time()
        utils.save_obj(new_posting, "avgVector" + str(self.last_bucket_index))
        # print("glove vector write time: ", time.time()-start_time)
        self.last_bucket_index += 1

    def append_vector(self, doc_id, vector, inverted_idx):  # gets tweet vector for insert to the bucket
        self.bucket.append_vector(doc_id, vector)
        self.size += 1
        if self.size > MAX_SIZE:
            self.__flush_bucket(inverted_idx)

    def finish_indexing(self, doc_dictionary):
        self.__flush_bucket(doc_dictionary)

