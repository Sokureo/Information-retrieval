from collections import defaultdict
from preprocessing import *
import json
from math import log
from flask import Flask
from flask import render_template, request
import re
from gensim.models import Word2Vec
import pickle
from gensim import matutils
import numpy as np

app = Flask(__name__)

with open('inverted_index.json') as f:
    inverted_index = json.load(f)

with open('term_doc_matrix.json') as f:
    term_doc_matrix = json.load(f)

with open('files_length.json') as f:
    files_length = json.load(f)

with open('files_list.json') as f:
    files_list = json.load(f)

model = Word2Vec.load('./model/araneum_none_fasttextskipgram_300_5_2018.model')
with open('w2v_indexed_base.pkl', 'rb') as f:
    w2v_base = pickle.load(f)

def score_BM25(qf, dl, avgdl, N, n) -> float:
    """
    Compute similarity score between search query and documents from collection
    :return: score
    """

    k1 = 2.0
    b = 0.75
    score = log((N - n + 0.5) / (n + 0.5)) * (k1 + 1) * qf / (qf + k1 * (1 - b + b * (dl / avgdl)))

    return score


def compute_sim(lemma, inverted_index, term_doc_matrix, files_length) -> float:
    """
    Compute similarity score between search query and documents from collection
    :return: score
    """

    relevance_score = {}
    avgdl = sum(files_length) / len(files_length)
    N = len(files_length)

    for doc in range(N):
        if lemma in term_doc_matrix:
            qf = term_doc_matrix[lemma][doc]
            n = len(inverted_index[lemma])
        else:
            qf = 0
            n = 0

        relevance_score[doc] = score_BM25(qf, files_length[doc], avgdl, N, n)

    return relevance_score


def search_inv_index(query, inverted_index, term_doc_matrix, files_length, n_results) -> list:
    """
    Compute sim score between search query and all documents in collection
    :param query: input text
    :return: list of doc_ids
    """

    relevance_dict = defaultdict(float)
    lemmas = preprocessing(query)

    for lemma in lemmas:
        sims = compute_sim(lemma, inverted_index, term_doc_matrix, files_length)
        for doc in sims:
            relevance_dict[doc] += sims[doc]

    result = sorted(relevance_dict, key=relevance_dict.get, reverse=True)[:n_results]

    return [re.split('/Friends - season [0-9]/Friends - ',
                     files_list[doc].strip('.ru.txt'))[1] for doc in result]


def similarity(v1, v2):
    v1_norm = matutils.unitvec(np.array(v1))
    v2_norm = matutils.unitvec(np.array(v2))
    return np.dot(v1_norm, v2_norm)


def get_w2v_vectors(model, lemmas):
    """Получает вектор документа"""

    vec_list = []

    for lemma in lemmas:
        try:
            vec_list.append(model.wv[lemma])
        except:
            # continue
            vec_list.append(np.array([0] * 300))
    #
    # if not vec_list:
    #     return np.array([0] * 300)
    #
    # else:
    #     doc_vec = sum(vec_list) / len(vec_list)
    #     return doc_vec
    doc_vec = sum(vec_list) / len(vec_list)
    return doc_vec


def search_w2v(query, model, w2v_base, n_results):
    query_vec = get_w2v_vectors(model, preprocessing(query))

    similarities = {}

    for doc in w2v_base:
        sim = similarity(query_vec, doc['vec'])
        # print(query_vec)
        similarities[sim] = doc['index']

    results = [re.split('/Friends - season [0-9]/Friends - ',
                        similarities[sim].strip('.ru.txt'))[1]
               for sim in sorted(similarities, reverse=True)[:n_results]]

    return results


def search(query, search_method, n_results=10):

    if search_method == 'inverted_index':
        search_result = search_inv_index(query, inverted_index,
                                         term_doc_matrix, files_length, n_results)

    elif search_method == 'word2vec':
        search_result = search_w2v(query, model, w2v_base, n_results)

    else:
        raise TypeError('unsupported search method')

    return search_result


@app.route('/')
def start():
    return render_template('start.html')


@app.route('/inverted_index')
def inverted_index():
    request.args.get('method', 'inverted_index')
    return render_template('inverted_index.html')


@app.route('/word2vec')
def word2vec():
    request.args.get('method', 'word2vec')
    return render_template('word2vec.html')


@app.route('/results')
def results():

    query = request.args['query']
    method = request.args['method']

    results = search(query, method)

    return render_template('results.html', res=results)


if __name__ == '__main__':
    app.run(debug=True)