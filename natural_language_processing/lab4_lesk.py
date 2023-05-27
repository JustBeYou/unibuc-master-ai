from nltk import word_tokenize, pos_tag
from nltk.wsd import lesk
from nltk.corpus import wordnet
from nltk.corpus import stopwords

english_stopwords = set(stopwords.words('english'))

# https://dl.acm.org/doi/pdf/10.1145/318723.318728
def original_lesk(context_tokens, ambiguous_word, pos=None):    
    word_synsets = wordnet.synsets(ambiguous_word)
    
    # Automatically detect the POS of the first occurence of the ambiguous word
    if pos == 'auto':
        context_tags = pos_tag(context_tokens)
        for word, tag in context_tags:
            if word != ambiguous_word:
                continue
                
            tag = penn2morphy(tag)
            if tag is None:
                continue
                
            pos = tag
            break
    
    if pos is not None:
        word_synsets = list(filter(lambda synset: synset.pos() == pos, word_synsets))
        
    
    # Gather all words from all definitions of the surrounding words
    context = set()
    for context_token in set(context_tokens):
        if context_token == ambiguous_word:
            continue
        
        synsets = wordnet.synsets(context_token)
        
        for synset in synsets:
            for token in synset.definition().split():
                context.add(token)
    
    # Remove the stopwords
    context = context.difference(english_stopwords)
    
    _, best_sense = max([(lesk_measure(context, sense), sense) for sense in word_synsets])
        
    return best_sense
    
def lesk_measure(context: set, sense):
    definition_tokens = set(sense.definition().split())
    return len(context.intersection(definition_tokens))

# https://stackoverflow.com/questions/35458896/python-map-nltk-stanford-pos-tags-to-wordnet-pos-tags
morphy_tag = {
    'NN':wordnet.NOUN, 
    'JJ':wordnet.ADJ,
    'VB':wordnet.VERB, 
    'RB':wordnet.ADV
}

def penn2morphy(penntag):
    penntag = penntag[:2]
    return morphy_tag[penntag] if penntag in morphy_tag else None


text = word_tokenize('Students enjoy going to school, studying and reading books.')
word = 'school'


syn = original_lesk(text, word, 'auto')
print("Original lesk:", syn, syn.definition())

syn = lesk(text, word, 'n')
print("NLTK lesk:", syn, syn.definition())


# https://www.nltk.org/_modules/nltk/corpus/reader/wordnet.html
default_extensions = [
    'hypernyms',
    'instance_hypernyms',
    'hyponyms',
    'instance_hyponyms',
    'member_holonyms',
    'substance_holonyms',
    'part_holonyms',
    'member_meronyms',
    'substance_meronyms',
    'part_meronyms',
    'attributes',
    'similar_tos',
    'also_sees'
]

# https://www.researchgate.net/profile/Ted-Pedersen/publication/2563220_Extended_Gloss_Overlaps_as_a_Measure_of_Semantic_Relatedness/links/00b49520cec00a4b51000000/Extended-Gloss-Overlaps-as-a-Measure-of-Semantic-Relatedness.pdf?origin=publication_detail
def extended_lesk(context_tokens, ambiguous_word, extensions = default_extensions):
    ambiguous_word = ambiguous_word.lower()
    word_synsets = wordnet.synsets(ambiguous_word)
    
    # Filter stopwords
    context_tokens = filter(lambda token: token not in english_stopwords, context_tokens)
    context_tokens = list(context_tokens)
    
    # Collect words and their tags
    context_tags = pos_tag(context_tokens)
    context_tags = map(lambda wt: (wt[0].lower(), penn2morphy(wt[1])), context_tags)
    context_tags = list(filter(lambda wt: wt[1] is not None, context_tags))
    
    context_synsets = []
    target_word_synsets = []
    
    # Collect the extended gloss for each word
    for word, tag in context_tags:
        syns = wordnet.synsets(word)
        syns = list(filter(lambda s: s.pos() == tag, syns))
        
        extended_syns = [getattr(syn, ext)() for ext in extensions for syn in syns]
        extended_syns = [syn for syn_list in extended_syns for syn in syn_list]
        
        if word == ambiguous_word:
            target_word_synsets.extend(syns)
        else:
            context_synsets.extend(syns)
            context_synsets.extend(extended_syns)
    
    # Tokenize and filter stopwords
    _, best_sense = max([(extended_lesk_measure_context(context_synsets, sense), sense) for sense in target_word_synsets])
    
    return best_sense

def sense_to_text(sense):
    text = sense.definition().split()
    return list(filter(lambda word: word not in english_stopwords, text))

def extended_lesk_measure_context(context, target_sense):
    target_sense = sense_to_text(target_sense)
    overlap = 0
    for context_sense in context:
        overlap += extended_lesk_measure(sense_to_text(context_sense), target_sense)
    return overlap

def extended_lesk_measure(a, b):
    a, b = a.copy(), b.copy()
    overlap = 0
    
    # Find all overlapping phrases first
    while True:
        max_len, (a_pos, b_pos) = longest_common_subarray(a, b)

        if max_len < 2:
            break
            
        overlap += max_len ** 2
        
        a[a_pos:a_pos+max_len] = '#'
        b[b_pos:b_pos+max_len] = '$'
    
    # When no overlapping phrases are left, do an element-wise intersection
    return overlap + len(set(a).intersection(set(b)))

def longest_common_subarray(a, b):
    n, m = len(a), len(b)
    dp = [[0 for _ in range(m + 1)] for _ in range(n + 1)]    
    max_len, max_pos = 0, (-1, -1)
    
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            if a[i] == b[j]:
                dp[i][j] = dp[i + 1][j + 1] + 1
                if dp[i][j] > max_len:
                    max_len = dp[i][j]
                    max_pos = (i, j)
                    
    return max_len, max_pos
    

print("LCS example", longest_common_subarray(
    [1, 2, 3, 4, 5, 6, 7, 9],
    [1, 2, 4, 5, 6, 9, 10, 11, 12]
))

print("Extended lesk measure example: ",extended_lesk_measure(
    [1, 2, 3, 4, 5, 6, 7, 9],
    [1, 2, 4, 5, 6, 9, 10, 11, 12]
))

syn = extended_lesk(text, word)
print("Extended Lesk 1:", syn, syn.definition())

syn = extended_lesk(text, word, extensions = [
    
])
print("Extended Lesk 2:", syn, syn.definition())


syn = extended_lesk(text, word, extensions = [
    'member_meronyms',
    'substance_meronyms',
    'part_meronyms',
    'attributes',
    'similar_tos',
    'also_sees'
])
print("Extended Lesk 3:", syn, syn.definition())


syn = extended_lesk(text, word, extensions = [
    'hypernyms',
    'instance_hypernyms',
    'hyponyms',
    'instance_hyponyms',
    'member_holonyms',
])
print("Extended Lesk 4:", syn, syn.definition())