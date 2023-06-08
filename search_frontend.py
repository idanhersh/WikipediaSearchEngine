#pip install -q google-cloud-storage==1.43.0
from google.cloud import storage
from flask import Flask, request, jsonify
import nltk
from nltk.corpus import wordnet
from nltk import stem
from nltk.corpus import stopwords
from nltk.stem.porter import *
from collections import Counter
from contextlib import closing
import gensim
from gensim.models import KeyedVectors
from gensim.models.keyedvectors import KeyedVectors
import json
import numpy as np
import math
import pandas as pd
import re
import gzip
import io
from inverted_index_gcp import *
from main import *

class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)

app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False


#regex, stopwords and tokenizing helpers
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
stopwords_frozen = frozenset(stopwords.words('english'))

def tokenize(text):
    """
    This function aims in tokenize a text into a list of tokens. Moreover, it filter stopwords.

    Parameters:
    -----------
    text: string , represting the text to tokenize.

    Returns:
    -----------
    list of tokens (e.g., list of tokens).
    """
    list_of_tokens = [token.group() for token in RE_WORD.finditer(text.lower()) if
                      token.group() not in stopwords_frozen]
    return list_of_tokens

def get_synonyms(word, n=2):
    synonyms = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.append(lemma.name())
    synonyms.remove(word)
    return list(set(synonyms))[:n]

def get_associated_words(word, model, n=4):
    similar_words = model.most_similar(positive=[word], topn=n)
    if len(similar_words) == 0:
        return []
    return [word for word, similarity in similar_words if similarity > 0.75]


# function that reads posting list from the Inverted Index
TUPLE_SIZE = 6
TF_MASK = 2 ** 16 - 1 # Masking the 16 low bits of an integer
def read_posting_list(inverted, w):
  with closing(MultiFileReader()) as reader:
    locs = inverted.posting_locs[w]
    if len(locs) == 0:
        return []
    b = reader.read(locs, inverted.df[w] * TUPLE_SIZE, inverted.indexname)
    posting_list = []
    for i in range(inverted.df[w]):
      doc_id = int.from_bytes(b[i*TUPLE_SIZE:i*TUPLE_SIZE+4], 'big')
      tf = int.from_bytes(b[i*TUPLE_SIZE+4:(i+1)*TUPLE_SIZE], 'big')
      posting_list.append((doc_id, tf))
    return posting_list


@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. This is 
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the 
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use 
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    # search function that return relevant documents according to the following procces:
    # Firstly we will do query expension
    # Secondly we get the relevant documents from Title & Anchor. Title = [(id, score), (id,score)..]
    # Then we do standardization to the inner ranking score so all values between 0-1.
    # Lastly we will give final score according the following weights:
    # Title = 50%
    # Anchor = 20 %
    # Body = 30 %
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
       return jsonify(res)
    alpha = 0.5   # Title weight
    beta = 0.2   # Anchor weight
    gamma = 0.3    # Body weight
    if len(query) < 3:  # if the title is small:
        alpha = 0.7
        beta = 0.1
        gamma = 0.2
    epsilon = 0.00001
    query1 = tokenize(query)
    query_expansion = []
    for i in range(len(query1)):
        try:
            query_expansion.extend(get_synonyms(query1[i]))
            query_expansion.extend(get_associated_words(query1[i], newmodel))
        except:
            pass
    query1.extend(query_expansion)
    query = list(set(query1))
    # starting with title:
    ranked_titles_dict = {}
    for i in range(len(query)):
        posting_list = read_posting_list(Title_Index, query[i])[:10000]  # posting list for token q[i]
        for j in posting_list:
            if j[0] not in ranked_titles_dict.keys():
                ranked_titles_dict[j[0]] = 1
            else:
                ranked_titles_dict[j[0]] += 1
    title_docs = sorted(list(ranked_titles_dict.items()), key=lambda x: x[1], reverse=True)  # [(id, score)..]
    if len(title_docs) > 1:
        values = list(ranked_titles_dict.values())
        maxval = max(values)
        minval = min(values)
        title_standardized_list = [(x[0], (x[1] - minval) / (maxval - minval + epsilon)) for x in title_docs]
    elif len(title_docs) == 1:
        title_standardized_list = [(title_docs[0][0], 1)]
    else:
        title_standardized_list = title_docs

    # anchor text part:
    ranked_anchor_dict = {}
    for i in range(len(query)):
        posting_list = read_posting_list(Anchor_Index, query[i])[:10000]   # posting list for token q[i]
        # making list of tuples: [(anchor_text, (dest doc_id, tf = 1))..]
        for j in posting_list:
            if j[0] not in ranked_anchor_dict.keys():
                ranked_anchor_dict[j[0]] = 1
            else:
                ranked_anchor_dict[j[0]] += 1
    anchor_docs = sorted(list(ranked_anchor_dict.items()), key=lambda x: x[1], reverse=True)  # [(id, score)..]
    if len(anchor_docs) > 1:
        values = list(ranked_anchor_dict.values())
        maxval = max(values)
        minval = min(values)
        anchor_standardized_list = [(x[0], (x[1] - minval) / (maxval - minval + epsilon)) for x in anchor_docs]
    elif len(anchor_docs) == 1:
        anchor_standardized_list = [(anchor_docs[0][0], 1)]
    else:
        anchor_standardized_list = anchor_docs

    # body text part:
    querylist = list(Counter(query).items())
    querynorm = np.sqrt(np.sum([i[1]**2 for i in querylist]))
    sims = {}   #dict that stores similarity (doc_id, q). key = doc_id, val = similarity
    for i in range(len(querylist)):  # iterating over query tokens
        if querylist[i][0] not in Body_Index.df.keys():
            continue
        posting_list = read_posting_list(Body_Index, querylist[i][0])[:10000]  #posting list for token q[i]
        df = Body_Index.df[querylist[i][0]]
        idf = math.log((Body_Index.number_of_docs) / (df + epsilon), 10)  # smoothing
        for j in range(len(posting_list)):          # calculating similarity between doc_i and q
            if posting_list[j][0] not in sims.keys():   # by calcilationg tf and then idf
                sims[posting_list[j][0]] = posting_list[j][1] * querylist[i][1] * idf
            else:
                sims[posting_list[j][0]] += posting_list[j][1] * querylist[i][1] * idf
    # normalizing by factors:
    dict_items = [(key, value*(Body_Index.docs_normal[key])*(1/querynorm)) for key, value in sims.items()]
    body_docs = sorted(dict_items, key = lambda x: x[1], reverse=True)
    if len(body_docs) > 1:
        values = list(sims.values())
        maxval = max(values)
        minval = min(values)
        body_standardized_list = [(x[0], (x[1] - minval) / (maxval - minval + epsilon)) for x in body_docs]
    elif len(body_docs) == 1:
        body_standardized_list = [(body_docs[0][0], 1)]
    else:
        body_standardized_list = body_docs
    new_dict = {}
    for i in range(len(title_standardized_list)):
        if title_standardized_list[i][0] not in new_dict.keys():
            new_dict[title_standardized_list[i][0]] = title_standardized_list[i][1] * alpha
        else:
            new_dict[title_standardized_list[i][0]] += title_standardized_list[i][1] * alpha
    for i in range(len(anchor_standardized_list)):
        if anchor_standardized_list[i][0] not in new_dict.keys():
            new_dict[anchor_standardized_list[i][0]] = anchor_standardized_list[i][1] * beta
        else:
            new_dict[anchor_standardized_list[i][0]] += anchor_standardized_list[i][1] * beta
    for i in range(len(body_standardized_list)):
        if body_standardized_list[i][0] not in new_dict.keys():
            new_dict[body_standardized_list[i][0]] = body_standardized_list[i][1] * gamma
        else:
            new_dict[body_standardized_list[i][0]] += body_standardized_list[i][1] * gamma
    # making a final list
    sorted_dict = sorted(new_dict.items(), key=lambda x: x[1], reverse=True)[:50]
    #med = [str(key) for key, value in sorted_dict]  # list of relevant doc_ids in decending order
    med = [key for key, value in sorted_dict]  # list of relevant doc_ids in decending order
    title = [TitleNames_Index.doc_name[i[0]] if i[0] in TitleNames_Index.doc_name.keys() else '' for i in sorted_dict]
    res = list(zip(med, title))
    return jsonify(res)


@app.route("/search_body")
def search_body():
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the 
        staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords. 

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    N = 50
    epsilon = 0.00001
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    query = tokenize(query)
    querylist = list(Counter(query).items())
    querynorm = np.sqrt(np.sum([i[1]**2 for i in querylist]))
    sims = {}   #dict that stores similarity (doc_id, q). key = doc_id, val = similarity
    for i in range(len(querylist)):  # iterating over query tokens
        posting_list = read_posting_list(Body_Index, querylist[i][0])[:10000]  #posting list for token q[i]
        df = Body_Index.df[querylist[i][0]]
        idf = math.log((Body_Index.number_of_docs) / (df + epsilon), 10)  # smoothing
        for j in range(len(posting_list)):          # calculating similarity between doc_i and q
            if posting_list[j][0] not in sims.keys():   # by calcilationg tf and then idf
                sims[posting_list[j][0]] = posting_list[j][1] * querylist[i][1] * idf
            else:
                sims[posting_list[j][0]] += posting_list[j][1] * querylist[i][1] * idf
    # normalizing by factors:
    dict_items = [(key, value*(Body_Index.docs_normal[key])*(1/querynorm)) for key, value in sims.items()]
    lsorted = sorted(dict_items, key = lambda x: x[1], reverse=True)[:N]
    med = [key for (key, value) in lsorted]    # list of top N relevant doc_ids
    #med = [str(key) for (key, value) in lsorted]    # list of top N relevant doc_ids
    title = [TitleNames_Index.doc_name[i[0]] for i in lsorted]
    res = list(zip(med, title))
    return jsonify(res)

@app.route("/search_title")
def search_title():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF 
        DISTINCT QUERY WORDS that appear in the title. DO NOT use stemming. DO 
        USE the staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords. For example, a document 
        with a title that matches two distinct query words will be ranked before a 
        document with a title that matches only one distinct query word, 
        regardless of the number of times the term appeared in the title (or 
        query). 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    query1 = tokenize(query)
    query = list(set(query1))
    ranked_titles_dict = {}
    for i in range(len(query)):
        posting_list = read_posting_list(Title_Index, query[i])  # posting list for token q[i]
        for j in posting_list:
            if j[0] not in ranked_titles_dict.keys():
                ranked_titles_dict[j[0]] = 1
            else:
                ranked_titles_dict[j[0]] += 1
    sorted_dict = sorted(ranked_titles_dict.items(), key=lambda x: x[1], reverse=True)
    #med = [str(key) for key, value in sorted_dict]  # list of relevant doc_ids in decending order
    med = [key for key, value in sorted_dict]  # list of relevant doc_ids in decending order
    title = [TitleNames_Index.doc_name[i[0]] if i[0] in TitleNames_Index.doc_name.keys() else '' for i in sorted_dict]
    res = list(zip(med, title))
    return jsonify(res)

@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE ANCHOR TEXT of articles, ordered in descending order of the 
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page. 
        DO NOT use stemming. DO USE the staff-provided tokenizer from Assignment 
        3 (GCP part) to do the tokenization and remove stopwords. For example, 
        a document with a anchor text that matches two distinct query words will 
        be ranked before a document with anchor text that matches only one 
        distinct query word, regardless of the number of times the term appeared 
        in the anchor text (or query). 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    query1 = tokenize(query)
    query = list(set(query1))
    ranked_anchor_dict = {}
    for i in range(len(query)):
        posting_list = read_posting_list(Anchor_Index, query[i])  # posting list for token q[i]
        for j in posting_list:
            if j[0] not in ranked_anchor_dict.keys():
                ranked_anchor_dict[j[0]] = 1
            else:
                ranked_anchor_dict[j[0]] += 1
    sorted_dict = sorted(ranked_anchor_dict.items(), key=lambda x: x[1], reverse=True)
    #med = [str(key) for key, value in sorted_dict]  # list of relevant doc_ids in decending order
    med = [key for key, value in sorted_dict]  # list of relevant doc_ids in decending order
    title = [TitleNames_Index.doc_name[i[0]] if i[0] in TitleNames_Index.doc_name.keys() else '' for i in sorted_dict]
    res = list(zip(med, title))
    return jsonify(res)


@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    ''' Returns PageRank values for a list of provided wiki article IDs. 

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    '''

    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    res = [PageRank_dict[i] if i in PageRank_dict.keys() else 0 for i in wiki_ids]
    return jsonify(res)

@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the 
          provided list article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    res = [wid2pv[i] if i in wid2pv.keys() else 0 for i in wiki_ids]
    return jsonify(res)


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)
