from collections import defaultdict
from preprocessing import *
import json
from math import log
from flask import Flask
from flask import render_template, request
import re

app = Flask(__name__)


with open('inverted_index.json') as f:
    inverted_index = json.load(f)

with open('term_doc_matrix.json') as f:
    term_doc_matrix = json.load(f)

with open('files_length.json') as f:
    files_length = json.load(f)

with open('files_list.json') as f:
    files_list = json.load(f)


def score_BM25(qf, dl, avgdl, k1, b, N, n) -> float:
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

        relevance_score[doc] = score_BM25(qf, files_length[doc], avgdl,
                                          2.0, 0.75, N, n)

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

# res = search_inv_index('рождественские каникулы', inverted_index, term_doc_matrix, files_length, 10)
# for elem in res:
#     print('{}: {}'.format(elem[0], elem[1]))


@app.route('/')
def start():
    return render_template('start.html')


@app.route('/results')
def search():
    query = request.args['query']

    results = search_inv_index(query,
                               inverted_index, term_doc_matrix, files_length, 10)

    return render_template('results.html', res=results)


if __name__ == '__main__':
    app.run(debug=True)