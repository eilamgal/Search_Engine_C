from glove import Glove


class ConfigClass:
    def __init__(self, corpus_path='', number_of_term_buckets=1, number_of_entities_buckets=1, output_path='', use_glove=True, use_thesaurus=True, stemming=True):
        self.number_of_term_buckets = number_of_term_buckets
        self.number_of_entities_buckets = number_of_entities_buckets
        self.first_entities_bucket_index = number_of_term_buckets
        self.corpusPath = corpus_path
        self.output_path = output_path
        self.glove_path = "glove.twitter.27B.25d.txt"
        self.use_glove = use_glove
        self.use_thesaurus = use_thesaurus
        self.stemming = stemming
        if self.use_glove:
            self.glove_dict = Glove(self.glove_path).embeddings_dict
        if self.use_thesaurus:
            from nltk.corpus import lin_thesaurus
            self.thesaurus = lin_thesaurus
            self.thesaurus.synonyms("read")
        # self.saveFilesWithStem = self.savedFileMainFolder + "/WithStem"
        # self.saveFilesWithoutStem = self.savedFileMainFolder + "/WithoutStem"
        # print('Project was created successfully..')

    def get__corpusPath(self):
        return self.corpusPath

    def get_model_url(self):
        return None

    def get_download_model(self):
        return self._download_model

    @property
    def model_dir(self): 
        return self._model_dir

    @model_dir.setter 
    def model_dir(self, model_dir):
        self._model_dir = model_dir

    def get_corpusPath(self):
        return self.corpusPath

    def get_output_path(self):
        return self.output_path

    def get_number_of_term_buckets(self):
        return self.number_of_term_buckets

    def get_number_of_entities_buckets(self):
        return self.number_of_entities_buckets

    def get_entities_index(self):
        return self.first_entities_bucket_index