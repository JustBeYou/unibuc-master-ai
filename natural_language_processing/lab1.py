# Lab 1: text_preprocessing

import collections
import re
import statistics
from hashlib import sha256
from os import path

import nltk
from num2words import num2words
from requests import get
from tabulate import tabulate

# For debugging purposes and fast processing
LIMIT_TEXT_SIZE = int(0)


def main():
    """
    1. Download it through python (inside the code, so you don't have to upload the 
    file too when you send the solution for this exercise) with urlopen() from module
    urllib and read the entire text in one single string. If the download takes too
    much time at each running, download the file, but leave the former instructions 
    in a comment (to show that you know how to access an online file) 
    """
    print("=== === === Section 1 === === ===")
    text = download_text("https://www.gutenberg.org/files/2554/2554-0.txt")
    print(f"Text size: {len(text) / 1024 / 1024:0.2f} MiB")

    if LIMIT_TEXT_SIZE:
        text = text[:LIMIT_TEXT_SIZE]

    print("=== === === === === === === ===\n")

    """
    2. Remove the header (keep only the text starting from the title)
    """
    print("=== === === Section 2 === === ===")
    title_uppercase = "CRIME AND PUNISHMENT"
    title_before_preface = text.find(title_uppercase)
    title_in_gutenberg_note = text.find(
        title_uppercase, title_before_preface + 1)
    title_before_text = text.find(title_uppercase, title_in_gutenberg_note + 1)
    text = text[title_before_text:]
    print(f"Book: {title_uppercase}")

    # Extra: Remove the footer
    end_of_book_mark = "*** END OF THE PROJECT GUTENBERG EBOOK CRIME AND PUNISHMENT ***"
    end_of_book = text.find(end_of_book_mark)
    if end_of_book > 0:
        text = text[:end_of_book]

    # Extra: Remove chapter titles and other stuff that messes the text
    text = re.sub(r'[A-Z]{2,}', '', text)
    text = re.sub(f'[IVXLCDM]+\n', '', text)

    # Extra: Contract multiple white spaces into one (including newlines, tabs etc)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    print("=== === === === === === === ===\n")

    """
    3. Print the number of sentences in the text. 
    Print the average length (number of words) of a sentence.
    """
    print("=== === === Section 3 === === ===")
    sentences = nltk.sent_tokenize(text)
    print(f"Number of sentences: {len(sentences)}")

    words_by_sentence = [nltk.word_tokenize(
        sentence) for sentence in sentences]
    average_words_per_sentence = statistics.mean(map(len, words_by_sentence))
    print(f"Average words per sentence: {average_words_per_sentence:0.2f}")
    print("=== === === === === === === ===\n")

    """
    4. Find the collocations in the text (bigram and trigram). Use the nltk.collocations module 
    You will print them only once not each time they appear.

    Reference: https://www.nltk.org/howto/collocations.html
    """
    print("=== === === Section 4 === === ===")
    N = 5
    words = [
        word.lower() for words in words_by_sentence for word in words if word.isalnum()]

    bigrams_finder = nltk.collocations.BigramCollocationFinder.from_words(
        words)
    bigrams_measure = nltk.collocations.BigramAssocMeasures()
    best_bigrams = bigrams_finder.nbest(bigrams_measure.pmi, N)
    print(
        f"Top-{N} bigrams: {', '.join([' + '.join(bigram) for bigram in best_bigrams])}")

    trigrams_finder = nltk.collocations.TrigramCollocationFinder.from_words(
        words)
    trigrams_measure = nltk.collocations.TrigramAssocMeasures()
    best_trigrams = trigrams_finder.nbest(trigrams_measure.pmi, N)
    print(
        f"Top-{N} trigrams: {', '.join([' + '.join(trigram) for trigram in best_trigrams])}")
    print("=== === === === === === === ===\n")

    """
    5. Create a list of all the words (in lower case) from the text, without the punctuation.
    """
    # Already done for the previous exercise, because computing bigrams
    # and trigrams was affected by punctuation and case. See "words" variable.

    """
    6. Print the first N most frequent words (alphanumeric strings) together with their number of appearances.
    """
    print("=== === === Section 6 === === ===")
    word_counts = collections.Counter(words)
    word_counts_str = ', '.join(
        [f"{word} ({count})" for word, count in word_counts.most_common(N)])
    print(f"Most common {N} words: {word_counts_str}")
    print("=== === === === === === === ===\n")

    """
    7. Remove stopwords and assign the result to variable lws
    """
    stopwords = nltk.corpus.stopwords.words('english')
    lws = [word for word in words if word not in stopwords]

    """
    8.  Apply stemming (Porter) on the list of words (lws). 
    Print the first 200 words. 
    Do you see any words that don't appear in the dictionary?

    Answer: yeah, a few, like 'earli', 'forc' ...
    """
    print("=== === === Section 8 === === ===")
    port = nltk.PorterStemmer()
    stem_lws = [port.stem(word) for word in lws]
    print("First 200 stemmed words: ", ', '.join(stem_lws[:200]))
    print("=== === === === === === === ===\n")

    """
    9. Print a table of three columns (of size N, where N is the maximum length for the words in the text).
    """
    print("=== === === Section 9 === === ===")
    NW = 10
    first_unique_words = set()
    for word in lws:
        if word in first_unique_words:
            continue

        first_unique_words.add(word)

        if len(first_unique_words) == NW:
            break

    port = nltk.PorterStemmer()
    lancaster = nltk.LancasterStemmer()
    snowball = nltk.SnowballStemmer('english')
    stems = [(word, port.stem(word), lancaster.stem(word),
              snowball.stem(word)) for word in first_unique_words]
    stems = filter(lambda pair: pair[1] != pair[2]
                   and pair[2] != pair[3] and pair[1] != pair[3], stems)
    stems = list(stems)
    print(
        f"Words with different stems accross all algorithms (first {NW} unique words in text)")
    print(tabulate(stems, headers=['Word', 'Porter', 'Lancaster', 'Snowball']))
    print("=== === === === === === === ===\n")

    """
    10. Print a table of two columns, simillar to the one above, that will compare the results of stemming and lemmatization
    """
    print("=== === === Section 10 === === ===")
    lemmatizer = nltk.stem.WordNetLemmatizer()
    pairs = [(snowball.stem(word), lemmatizer.lemmatize(word))
             for word in first_unique_words]
    pairs = filter(lambda pair: pair[0] != pair[1], pairs)
    pairs = list(pairs)
    print(
        f"Words with different stem and lemma (first {NW} unique words in text)")
    print(tabulate(pairs, headers=['Snowball', 'WordNetLemmatizer']))
    print("=== === === === === === === ===\n")

    """
    11. Print the first N most frequent lemmas (after the removal of stopwords) together with their number of appearances.
    """
    print("=== === === Section 11 === === ===")
    N = 5
    unique_words = set(lws)
    lemmas = [lemmatizer.lemmatize(word) for word in unique_words]
    lemma_counts = collections.Counter(lemmas)
    lemma_counts_str = ', '.join(
        [f"{lemma} ({count})" for lemma, count in lemma_counts.most_common(N)])
    print(f"Most common {N} lemmas: {lemma_counts_str}")
    print("=== === === === === === === ===\n")

    """
    12. Change all the numbers from lws into words. Print the number of 
    changes, and also the portion of list that contains first N changes (for example N=10).
    """
    print("=== === === Section 12 === === ===")
    N = 5
    found = 0
    portion = 0
    changes = []
    for i, w in enumerate(words):
        if w.isnumeric():
            changes.append((w, num2words(w)))
            found += 1

        if found == N:
            portion = i + 1
            break

    # this is really huge, I advise not to print it :P
    # portion_containing_nums = words[:portion]
    # print(portion_containing_nums)
    print(
        f"{len(changes)} numbers converted to numbers: {'; '.join(map(lambda x: x[1]+' ('+x[0]+')', changes))}")
    print("=== === === === === === === ===\n")

    """
    13. Create a function that receives an integer N and a word W as parameter 
    (it can also receive the list of words from the text). We want to print the concordance 
    data for that word. This means printing the window of text (words on consecutive positions) 
    of length N, that has the givend word W in the middle. For example, for the text 
    ""I have two dogs and a cat. Do you have pets too? My cat likes to chase mice.
     My dogs like to chase my cat." and a window of length 3, the concordance data for the word "cat" 
     would be ["dogs", "cat", "pets"] and ["pets","cat", "likes"] (we consider the text without 
     stopwords and punctuation). However, as you can see, the window of text may contain words 
     from different sentences. Create a second function that prints windows of texts that 
     contain words only from the phrase containing word W. We want to print concordance data 
     for all the inflexions of word W.
    """
    print("=== === === Section 13 === === ===")
    K = 5
    words = text_to_words(text)
    test_concordance = concordance(3, 'nothing', words)[:5]
    print(f"Example of concordance windows: {test_concordance}")

    sentences = text_to_words(text, flatten=False)
    test_concordance = concordance_by_sentence(3, 'nothing', sentences)[:5]
    print(
        f"Example of concordance windows with sentence boundary: {test_concordance}")
    print("=== === === === === === === ===\n")


def concordance(N, W, words):
    assert N % 2 == 1
    N = (N - 1) // 2

    lemmatizer = nltk.WordNetLemmatizer()
    # As we want to look at all inflexions, we lemmatize first
    W_lemma = lemmatizer.lemmatize(W)
    concordances = []
    for i, cw in enumerate(words):
        cw_lemma = lemmatizer.lemmatize(cw)
        if cw_lemma != W_lemma:
            continue

        concordances.append(words[max(0, i-N):min(len(words), i+N+1)])

    return concordances


def concordance_by_sentence(N, W, sentences):
    concordances = []
    for sentence in sentences:
        sentence_concordances = concordance(N, W, sentence)
        concordances.extend(sentence_concordances)
    return concordances


def text_to_words(text, flatten=True):
    sentences = nltk.sent_tokenize(text)
    words_by_sentence = [nltk.word_tokenize(
        sentence) for sentence in sentences]
    words_by_sentence = [clear_words(sentence)
                         for sentence in words_by_sentence]

    if flatten:
        return [word for sentence in words_by_sentence for word in sentence]
    return words_by_sentence


def clear_words(words):
    words = [word.lower() for word in words if word.isalnum()]
    stopwords = nltk.corpus.stopwords.words('english')
    return [word for word in words if word not in stopwords]


def download_text(url, cache=True):
    filename = sha256(url.encode('ascii')).hexdigest() + '.txt'
    if cache and path.isfile(filename):
        return open(filename).read()

    text = get(url).text
    if cache:
        with open(filename, 'w') as cache_file:
            cache_file.write(text)

    return text


if __name__ == "__main__":
    main()
