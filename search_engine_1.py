import pandas as pd
from reader import ReadFile
from configuration import ConfigClass
from parser_module import Parse
from indexer import Indexer
from searcher import Searcher
import utils
import glob


# DO NOT CHANGE THE CLASS NAME
class SearchEngine:

    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation, but you must have a parser and an indexer.
    def __init__(self, config=None):
        self._config = config
        self._config.use_thesaurus = False
        self._parser = Parse()
        self._indexer = Indexer(config)
        self._inverted_index = None
        self._tweet_dict = None
        self._model = None

    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation as you see fit.
    def build_index_from_parquet(self, fn):
        """
        Reads parquet file and passes it to the parser, then indexer.
        Input:
            fn - path to parquet file
        Output:
            No output, just modifies the internal _indexer object.
        """
        r = ReadFile(corpus_path="Data")
        p = Parse(self._config.stemming)
        indexer = Indexer(self._config)
        all_files_paths = glob.glob("Data" + "\\*\\*.snappy.parquet")
        all_files_paths.extend(glob.glob("Data" + "\\*.snappy.parquet"))
        all_files_names = [file_name[file_name.find("\\") + 1:] for file_name in all_files_paths]
        for file_name in all_files_names:
            documents_list = [document for document in r.read_file(file_name=file_name)]
            # Iterate over every document in the file
            for idx, document in enumerate(documents_list):
                parsed_document = p.parse_doc(document)
                indexer.add_new_doc(parsed_document)
        indexer.finish_indexing()
        print('Finished parsing and indexing.')

    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation as you see fit.
    def load_index(self, fn):
        """
        Loads a pre-computed index (or indices) so we can answer queries.
        Input:
            fn - file name of pickled index.
        """
        self._inverted_index = utils.load_obj(fn)
        self._tweet_dict = self.__load_tweet_dict()

    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation as you see fit.
    def load_precomputed_model(self, model_dir=None):
        """
        Loads a pre-computed model (or models) so we can answer queries.
        This is where you would load models like word2vec, LSI, LDA, etc. and
        assign to self._model, which is passed on to the searcher at query time.
        """
        pass

    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation as you see fit.
    def search(self, query, k=2000):
        """
        Executes a query over an existing index and returns the number of
        relevant docs and an ordered list of search results.
        Input:
            query - string.
        Output:
            A tuple containing the number of relevant search results, and
            a list of tweet_ids where the first element is the most relavant
            and the last is the least relevant result.
        """
        query_as_list, entities = self._parser.parse_text(query)
        full_query = query_as_list + entities
        searcher = Searcher(self._parser, self._indexer, self._config)
        relevant_docs = searcher.relevant_docs_from_posting(full_query)
        ranked_docs = searcher.ranker.rank_relevant_doc(relevant_docs)
        retrieve_list = searcher.ranker.retrieve_top_k(ranked_docs, k)
        return len(retrieve_list), retrieve_list

    def __load_tweet_dict(self):
        """
        read the tweet vector files and insert the vectors to the tweet dictionary
        :return tweet_Dictionary including the Glove vector data
        """
        tweet_dict = utils.load_obj("docDictionary")
        if self._config.use_glove:
            tweet_vectors = []
            for i in range(tweet_dict["metadata"]["tweet_vector_buckets"]):
                tweet_vectors.append(utils.load_obj("avgVector" + str(i)))
            for tweet_id in tweet_dict.keys():
                if tweet_id == "metadata":
                    continue
                address = tweet_dict[tweet_id][5]
                tweet_dict[tweet_id][5] = tweet_vectors[address[0]][address[1]]
        return tweet_dict