from bucket import Bucket
import utils

MAX_SIZE = 10000000
THRESHOLD = 200000
abc_frequency_dict = {'a': 1.7, 'b': 4.4, 'c': 5.2, 'd': 3.2, 'e': 2.8, 'f': 4, 'g': 1.6, 'h': 4.2, 'i': 7.3, 'j': 7.3,
                      'k': 0.86, 'l': 2.4, 'm': 3.8, 'n': 2.3, 'o': 7.6, 'p': 4.3, 'q': 0.22, 'r': 2.8, 's': 6.7,
                      't': 16, 'u': 1.2, 'v': 0.82, 'w': 5.5, 'x': 0.045, 'y': 0.76, 'z': 0.045}


class PostingsHandler:
    """
    A class for managing posting files for terms and entities in memory, and writing them to the disk when they
    get too big (controlled via the MAX_SIZE and THRESHOLD constants).
    """

    def __init__(self, config, number_of_buckets=6, first_bucket_index=0, contains="terms"):
        if contains == "terms":
            number_of_buckets = config.get_number_of_term_buckets()
        elif contains == "entities":
            number_of_buckets = config.get_number_of_entities_buckets()
            first_bucket_index = config.get_entities_index()
        self.buckets = []
        self.letters_dict = {}  # mapping between a letter and a bucket number
        self.size = 0
        # self.config = config
        self.__equal_width_buckets(number_of_buckets)
        self.buckets_mapping = {}  # mapping between bucket in memory to a bucket index on the disk
        self.initialize_buckets(number_of_buckets, first_bucket_index)

    def initialize_buckets(self, num_of_buckets, first_bucket_index=0):
        """
        Creates the bucket files on the disk and saves a mapping for their indices
        """
        for i in range(num_of_buckets):
            utils.save_obj([], "bucket" + str(first_bucket_index + i))
            self.buckets_mapping[i] = first_bucket_index + i

    def __flush_bucket(self, inverted_idx, bucket_index):
        """
        Writes a bucket to the disk. If it already has data saved on the disk it first loads and extends the data
        with the new data in the memory and only then saves it again.
        """
        new_posting = utils.load_obj("bucket" + str(self.buckets_mapping[bucket_index]))
        for term in self.buckets[bucket_index].get_dict_terms():
            if inverted_idx[term][1][1] < 0:  # update the inverted index pointer if needed
                new_posting.append([])
                inverted_idx[term][1] = (self.buckets_mapping[bucket_index], len(new_posting) - 1)
            new_posting[inverted_idx[term][1][1]].extend(self.buckets[bucket_index].get_term_posting(term))
        self.size -= self.buckets[bucket_index].get_size()
        self.buckets[bucket_index].clean_bucket()
        utils.save_obj(new_posting, "bucket" + str(self.buckets_mapping[bucket_index]))

    def __equal_width_buckets(self, number_of_buckets):
        for key in abc_frequency_dict.keys():
            self.letters_dict[key] = (ord(key) - ord('a')) % number_of_buckets
        for i in range(number_of_buckets):
            new_bucket = Bucket()
            self.buckets.append(new_bucket)

    def __find_the_biggest_bucket(self):
        max_size_bucket = 0
        for i in range(1, len(self.buckets)):
            if max_size_bucket < self.buckets[i].get_size():
                max_size_bucket = i
        return max_size_bucket

    def append_term(self, term, tweet_id, frequency, inverted_idx):
        """"
        This function gets posting tuple data for term posting file and insert the tuple in term bucket
        If after the insertion the total size of all the buckets is bigger then MAX_SIZE so each bucket
        with size bigger then THRESHOLD will be flush.
        :param term is a term in the corpus
        :param tweet_id
        :param frequency of term in tweet_id
        :param inverted_idx for update pointer if needed or find bucket in the disk
        """
        if len(term) == 0 or not term[0].isalpha():
            if len(term) > 1 and term[1].isalpha():
                self.buckets[self.letters_dict[term[1].lower()]].append_term(term, tweet_id, frequency)
            else:
                self.buckets[-1].append_term(term, tweet_id, frequency)
        else:
            self.buckets[self.letters_dict[term[0].lower()]].append_term(term, tweet_id, frequency)
        self.size += 1
        if self.size > MAX_SIZE:
            for i in range(len(self.buckets)):
                if self.buckets[i].get_size() > THRESHOLD:
                    self.__flush_bucket(inverted_idx, i)

    def change_term_case(self, old_term, new_term):
        self.buckets[self.letters_dict[old_term[0].lower()]].rename_key(old_term, new_term)

    def finish_indexing(self, inverted_idx):  # flush all the buckets
        for i in range(len(self.buckets)):
            if self.buckets[i].get_size != 0:
                self.__flush_bucket(inverted_idx, i)


