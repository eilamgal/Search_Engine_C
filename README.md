# Search_Engine
To use the the program simply run main.py. 
Make sure to set index_corpus and query_engine variables to True/False to determine the functionality you desire:
index_corpus - Build the engine from scratch running on the entire corpus
query_engine - Allows the user to manually enter queries to run on the built engine

WHEN RUNNING LOCALLY MAKE SURE TO COMMENT OUT THE CORRECT LINE IN search_engine.main()
(THE FILE SHOULD BE FOUND IN THE MAIN PROJECT'S FOLDER) :

  glove_dict = GloveStrategy("glove.twitter.27B.25d.txt").embeddings_dict  - **LOCAL RUN** 
  OR 
  glove_dict = GloveStrategy("../../../../glove.twitter.27B.25d.txt").embeddings_dict  - **SUBMISSION SYS RUN**

## Configuration info:

-ConfigClass(corpus_path, number_of_term_buckets=term_buckets, number_of_entities_buckets=entities_buckets, use_glove, use_thesaurus) will 
set the number of number of term and entities buckets

-for stemming just change main(stemming=True) the default is stemming=False
-main(corpus_path=corpus_path) with the corpus path on your PC
-main(output_path=output_path) to set the output path on your PC

postings_handler.py:
-adjust the max size of the all buckets before flush just change MAX_SIZE in posting_handler.py
-adjust the threshold  of the buckets who will flush just change THRESHOLD in posting_handler.py

ranker.py:
-adjust the weight of each ranking measure of Glove just change GLOVE_WEIGHT in ranker.py
-adjust the weight of each ranking measure of BM25 just change BM25_WEIGHT in ranker.py
-adjust the weight of each ranking measure of referral-rank just change REFERRAL_WEIGHT in ranker.py
-adjust the weight of each ranking measure of time-rank just change RELEVANCE_WEIGHT in ranker.py


## Important info:
-run_engine method we initialize ConfigClass we to have 26 terms buckets and 2 entities buckets
-only when finish_indexing is called all the referrals are insert into the inverted index
