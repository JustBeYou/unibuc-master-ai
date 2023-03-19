import pickle
from hashlib import sha256
from os import path, mkdir

import nltk
import wikipedia
from nltk.tag.stanford import StanfordPOSTagger
import pytictoc
from tabulate import tabulate


def main():
    """
    1. Choose a wikipedia article. You will download and acces the article using this
    python module: wikipedia. Use the content property to extract the text.
    Print the title of the article and the first N=200 words from the article to
    verify that all works well. Print the POS-tagging for the first N=20 sentences.
    """

    print("=== Section 1 ===")
    tagger = StanfordPOSTagger(
        'english-bidirectional-distsim.tagger', java_options="-mx4096m")
    page = download_wiki_page('Philosophy')
    words = nltk.word_tokenize(page)
    words = clear_words(words)
    print(tagger.tag(words[:20]))
    print("=== === ===\n")

    """
    2. Create a function that receives a part of speech tag and returns a list with all the
    words from the text (can be given as a parameter too) that represent that part of speech.
    Create a function that receives a list of POS tags and returns a list with words having any of
    the given POS tags (use the first function in implementing the second one).
    """
    print('=== Section 2 ===')
    tagged_words = batch_tagging(words[:100], 25, tagger, verbose=True)
    words_by_pos = group_by_pos(tagged_words)
    example_1 = words_of_type('VBZ', words_by_pos)
    print('some verbs', example_1)
    example_2 = words_of_types(['VBZ', 'NN'], words_by_pos)
    print('some verbs and nouns', example_2)
    print('=== === ===\n')

    """
    3. Use the function above to print all the nouns (there are multiple tags for nouns),
    and, respectively all the verbs (corresponding to all verb tags). Also,
    print the percentage of content words (noun+verbs) from the entire text.
    """
    print('=== Section 3 ===')
    t = pytictoc.TicToc()
    t.tic()
    tagged_words = batch_tagging(words, 100, tagger, verbose=True)
    t.toc('POS tagging took')

    words_by_pos = group_by_pos(tagged_words)
    nouns_and_verbs_tags = filter(lambda pos: pos.startswith(
        'V') or pos.startswith('N'), words_by_pos.keys())
    nouns_and_verbs_tags = list(nouns_and_verbs_tags)
    nouns_and_verbs = set().union(*[words_by_pos[tag]
                                    for tag in nouns_and_verbs_tags])
    nouns_and_verbs = list(nouns_and_verbs)
    print("\n", nouns_and_verbs[:100], "\n")

    counts_by_pos = count_by_pos(tagged_words)
    nouns_and_verbs_count = sum([counts_by_pos[tag]
                                for tag in nouns_and_verbs_tags])

    print(
        f"Percentage of nouns and verbs (unique words): {len(nouns_and_verbs)/len(set(words))*100:.2f}% ({len(nouns_and_verbs)}/{len(set(words))})")
    print(
        f"Percentage of nouns and verbs (all words): {nouns_and_verbs_count/len(words)*100:.2f}% ({nouns_and_verbs_count}/{len(words)})")
    print('=== === ===\n')

    """
    4. ...  Original word | POS | Simple lemmatization | Lemmatization with POS 
    """
    print("=== Section 4 ===")
    N = 5

    sentences = nltk.sent_tokenize(page)[:N]
    words_by_sentence = [clear_words(
        nltk.word_tokenize(sentence)) for sentence in sentences]

    sent_words = []
    for sentence in words_by_sentence:
        sent_words.extend(sentence)

    words_tagged = batch_tagging(sent_words, 100, tagger, verbose=True)

    lemmatizer = nltk.stem.WordNetLemmatizer()
    results = set()
    for word, tag in words_tagged:
        if tag[0].lower() not in "nvas":
            continue

        lemma = lemmatizer.lemmatize(word)
        lemma_pos = lemmatizer.lemmatize(word, tag[0].lower())

        if lemma != lemma_pos:
            results.add((word, tag, lemma, lemma_pos))

    print(tabulate(list(results), headers=[
          'Original word', 'POS', 'Simple lemmatization', 'Lemmatization with POS']))
    print("=== === ===\n")

    """
    5. Print a graphic showing the number of words for each part of speech. If there are 
    too many different parts of speech, you can print only those with a higher number 
    of corresponding words.
    """
    print('=== Section 5 ===')
    N = 5
    counts = [(count, pos) for pos, count in counts_by_pos.items()]
    counts = sorted(counts, reverse=True)[:N]
    percent = int(0.01 * len(words))
    for count, pos in counts:
        print(pos.rjust(6, ' '), (int(count/percent) *
              '#').ljust(20, ' '), str(str(count) + '/' + str(len(words))).rjust(15))
    print('=== === ===')


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


def words_of_type(tag, words_by_pos):
    return words_by_pos[tag]


def words_of_types(tags, words_by_pos):
    result = set()
    for tag in tags:
        result |= words_of_type(tag, words_by_pos)
    return result


@cache(limit=2)
def batch_tagging(words, batch_size, tagger, verbose=False):
    result = []
    for i in range(0, len(words), batch_size):
        if verbose:
            print(
                f"Tagging {min(i+batch_size, len(words))}/{len(words)} words.")

        batch = words[i:i+batch_size]
        tagged_batch = tagger.tag(batch)
        result.extend(tagged_batch)
    return result


def group_by_pos(tagged_words):
    groups = {}
    for word, tag in tagged_words:
        if tag not in groups:
            groups[tag] = set([word])
        else:
            groups[tag].add(word)
    return groups


def count_by_pos(tagged_words):
    groups = {}
    for _, tag in tagged_words:
        if tag not in groups:
            groups[tag] = 1
        else:
            groups[tag] += 1
    return groups


def clear_words(words):
    return list(filter(lambda w: w.isalnum(), words))


@cache()
def download_wiki_page(title):
    return wikipedia.page(title).content


if __name__ == "__main__":
    main()
