# IR-project
Implementation of search engine for wiki dumps using gcp and colab

Our project repo contains 4 files:
1) project_gcp.ipynb
2) run_frontend_in_colab.ipynb
3) search_frontend.py
4) Project Report and list of all index files

The first file is the notebook in which you can see how we worked on the data in order to make the following data structres: Inverted Indexes- Title Index, Body Index, Anchor Index and Titles Names Index, Dictionaries- PageRank dictionary, PageView dictionary and Association model dictionary. In addition, you can see how we managed to upload them to our bucket and Importing them as an instance locally. In the file you can see the size of each Index as well.
The second file is the notebook where you can see how we created the Indexes and dictionaries in colab notebook and used ngrok for testing our engine.
The third file contains the implementation of the functions that retreives the relevant documents from corpus, for example - Search function: 
We searched the given query after performing query expansion, where we generated 2 synonyms and 4 associated words. We took the set of all these words and the query tokens in order to eliminate duplicates from our search. We then sent the query after expansion to our three search methods : Search_Body(),Search_Title() and Search_Anchor(). From each method we returned the top 50 results and limited their posting lists to 10000 documents. We then standardized their scores (Body score based on cosine similarity, Title and Anchor based on ranking position) using min max scaler. After standardizing the scores returned from each method, we combined the results to a single ranking, by giving each ranking method a weight: Title-50%, Body-30% and Anchor-20%. For short queries (less than 3 tokens), we gave a different weight distribution- 70% Title, 20% Body and 10% Anchor. After creating the combined rankings based on weighted scores we returned the top 50 results.
