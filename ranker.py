GLOVE_WEIGHT = 0.3
BM25_WEIGHT = 0.2
REFERRAL_WEIGHT = 0.3
RELEVANCE_WEIGHT = 0.2


class Ranker:
    def __init__(self):
        pass

    @staticmethod
    def rank_relevant_doc(relevant_doc):
        """
        This function provides rank for each relevant document and sorts them by their scores.
        The current score considers solely the number of terms shared by the tweet (full_text) and query.
        :param relevant_doc: dictionary of documents that contains at least one term from the query.
        :return: sorted list of documents by score
        """
        return sorted(relevant_doc.items(), key=lambda item:   GLOVE_WEIGHT*item[1][0]
                                                             + BM25_WEIGHT*item[1][1]
                                                             + REFERRAL_WEIGHT*item[1][2]
                                                             + RELEVANCE_WEIGHT*item[1][3], reverse=True)

    @staticmethod
    def retrieve_top_k(sorted_relevant_doc, k=1):
        """
        return a list of top K tweets based on their ranking from highest to lowest
        :param sorted_relevant_doc: list of all candidates docs.
        :param k: Number of top document to return
        :return: list of relevant document
        """
        k = min(k, len(sorted_relevant_doc))
        return sorted_relevant_doc[:k]

    def __inner_product(self):
        return

    def __cosine_similarity(self):
        return



    def __euclidean_distance(Self):
        return