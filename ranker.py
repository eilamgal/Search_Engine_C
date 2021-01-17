GLOVE_WEIGHT = 1
BM25_WEIGHT = 0.1
REFERRAL_WEIGHT = 0
RELEVANCE_WEIGHT = 0


class Ranker:
    def __init__(self, glove_weight=0.7, bm25_weight=0.28, referral_weight=0.01, relevance_weight=0.01):
        self.glove_weight = glove_weight
        self.bm25_weight = bm25_weight
        self.referral_weight = referral_weight
        self.relevance_weight = relevance_weight

    def rank_relevant_doc(self, relevant_doc):
        """
        This function provides rank for each relevant document and sorts them by their scores.
        The current score considers solely the number of terms shared by the tweet (full_text) and query.
        :param relevant_doc: dictionary of documents that contains at least one term from the query.
        :return: sorted list of documents by score
        """
        x = sorted(relevant_doc.keys(), key=lambda key:   self.glove_weight*relevant_doc[key][0]
                                                             + self.bm25_weight*relevant_doc[key][1]
                                                             + self.referral_weight*relevant_doc[key][2]
                                                             + self.relevance_weight*relevant_doc[key][3], reverse=True)
        y = sorted(relevant_doc.items(), key=lambda item:   self.glove_weight*item[1][0]
                                                             + self.bm25_weight*item[1][1]
                                                             + self.referral_weight*item[1][2]
                                                             + self.relevance_weight*item[1][3], reverse=True)
        return sorted(relevant_doc.keys(), key=lambda key:   self.glove_weight*relevant_doc[key][0]
                                                             + self.bm25_weight*relevant_doc[key][1]
                                                             + self.referral_weight*relevant_doc[key][2]
                                                             + self.relevance_weight*relevant_doc[key][3], reverse=True)

    def retrieve_top_k(self, sorted_relevant_doc, k=1):
        """
        return a list of top K tweets based on their ranking from highest to lowest
        :param sorted_relevant_doc: list of all candidates docs.
        :param k: Number of top document to return
        :return: list of relevant document
        """
        k = min(k, len(sorted_relevant_doc))
        return sorted_relevant_doc[:k]

