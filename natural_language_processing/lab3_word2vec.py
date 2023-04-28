import pickle
from hashlib import sha256
from os import mkdir, path

import numpy
import spacy
import wikipedia
from sklearn.cluster import KMeans


def main():
    nlp = spacy.load("en_core_web_sm")
    page = download_wiki_page('Philosophy')

    words = []
    not_in_vocab = []

    entities = set()
    for ent in nlp(page, disable=['parser', 'lemmatizer', 'attribute_ruler']).ents:
        entities.add(ent.label_)

    for i, doc in enumerate(nlp(page, disable=['parser', 'lemmatizer', 'attribute_ruler'])):
        w = str(doc)
        if not w.isalnum():
            continue
        if i == 600:
            break

        words.append((w, doc.tensor))

        if w not in nlp.vocab:
            not_in_vocab.append(w.lower())

    print(f"Words in vocabulary: {len(nlp.vocab)}")
    print(f"Words not in vocabulary: {not_in_vocab}")

    min_dist = 999999999999999999999
    min_sol = None
    max_dist = 0
    max_sol = None
    for a, at in words:
        for b, bt in words:
            if a == b:
                continue
            dist = numpy.linalg.norm(at - bt)
            if dist < min_dist:
                min_sol = (a, b, dist)
                min_dist = dist
            if dist > max_dist:
                max_sol = (a, b, dist)
                max_dist = dist

    print(
        f"Minimum distance between {min_sol[0]} - {min_sol[1]} = {min_sol[2]}")
    print(
        f"Maximum distance between {max_sol[0]} - {max_sol[1]} = {max_sol[2]}")

    print(f"NER in text: {entities}")

    X = numpy.array(list(map(lambda x: x[1], words)), dtype=object)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init="auto").fit(X)

    f = zip(kmeans.labels_, list(map(lambda x: x[0], words)))
    x = [[], [], []]
    for a, b in list(f)[:50]:
        x[a].append(b)

    print("Example of clustered words:")
    print(x[0])
    print(x[1])
    print(x[2])


def cache(limit=None):
    if not path.isdir('.cache'):
        mkdir('.cache')

    def _cache(f):
        def cached_f(*args, **kwargs):
            args_part = args if limit is None else args[:limit]
            filename = '.cache/' + sha256(str(args_part).encode(
                'utf8')).hexdigest() + '.pickle'

            if path.isfile(filename):
                return pickle.load(open(filename, 'rb'))

            value = f(*args, **kwargs)
            pickle.dump(value, open(filename, 'wb'))
            return value

        return cached_f
    return _cache


@ cache()
def download_wiki_page(title):
    return wikipedia.page(title).content


if __name__ == "__main__":
    main()
