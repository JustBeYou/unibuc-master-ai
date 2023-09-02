from nltk import word_tokenize, sent_tokenize, pos_tag
from nltk.wsd import lesk
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from pprint import pprint
import numpy as np

english_stopwords = set(stopwords.words('english'))

def fitness(C, dictionary, cache, extensions):
    s = 0
    for a in C:
        for b in C:
            s += extended_lesk_measure(a, b, dictionary, cache, extensions)
    return s

def extended_lesk_measure(a, b, dictionary, cache, extensions):
    # Support for both word vectors and synset names
    word_vectors = type(a) is list and type(b) is list
    if not word_vectors:
        key_1 = (a.name(), b.name())
        key_2 = (b.name(), a.name())
    
        if key_1 in cache: return cache[key_1]
        if key_2 in cache: return cache[key_2]
    
        a = sense_to_tokens_set(a, dictionary, cache, extensions)['extended']
        b = sense_to_tokens_set(b, dictionary, cache, extensions)['extended']
    
    # Intersection in O(max(N, M)) for two sorted arrays
    overlap, i, j = 0, 0, 0
    while i < len(a) and j < len(b) and a[i] is not None and b[j] is not None:
        if a[i] == b[j]:
            overlap += 1
            i += 1
            j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1
    
    if not word_vectors:
        cache[key_1] = overlap
    
    return overlap

def sense_to_tokens_set(sense, dictionary, cache, extensions):
    if sense.name() in cache:
        return cache[sense.name()]
    
    # Get all Wordnet relations
    extended_senses = [getattr(sense, ext)() for ext in extensions]
    extended_senses = [current_sense for sense_list in extended_senses for current_sense in sense_list] + [sense]
    
    # Create a bag-of-words for all definitions
    tokens = set()
    self_tokens = None
    
    for current_sense in extended_senses:
        current_tokens = set(
            filter(
                lambda word: word.isalpha() and word not in english_stopwords, 
                word_tokenize(current_sense.definition())
            )
        )
        current_tokens = [word.lower() for word in current_tokens]
        tokens.update(current_tokens)
        
        if sense.name() == current_sense.name():
            self_tokens = set(current_tokens)
    
    # Obtain unique ids for each word and create a sorted array of word ids
    token_ids = [-1 for _ in range(len(tokens))]
    for i, token in enumerate(tokens):
        if token not in dictionary:
            dictionary[token] = len(dictionary)
            
        token_ids[i] = dictionary[token]
        
    self_token_ids = [-1 for _ in range(len(self_tokens))]
    for i, token in enumerate(self_tokens):
        self_token_ids[i] = dictionary[token]
        
    token_ids.sort()
    self_token_ids.sort()
    
    cache[sense.name()] = {'extended': token_ids, 'simple': self_token_ids}
    return cache[sense.name()]

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


default_parameters = {
    'E_a': 5,        # energy taken by an ant
    'E_max': 30,      # max energy of ant
    'delta': 0.3,     # evaporation rate of cycles
    'E_0': 10,        # initial enery in nodes
    'omega': 5,      # ant life-span
    'L_v': 100,        # odour vector length
    'delta_v': 0.5,   # percent of odour deposited at a node
    'c_ac': 17,        # number of cycles
    'theta': 0.05,       # amount of ant pheromone
}

def ant_colony(text, extensions, parameters):
    dictionary = {}
    cache = {}
    
    # Pre-process and tokenize the input text
    text = text.replace('\n', ' ')
    sentences = sent_tokenize(text)
    sentences = [word_tokenize(sentence) for sentence in sentences]
    sentences = [[word.lower() for word in sentence if word.isalpha()] for sentence in sentences]
    sentences = [[(word, wordnet.synsets(word)) for word in sentence] for sentence in sentences]
    sentences = [[(word, senses) for word, senses in sentence if len(senses) > 0] for sentence in sentences]
    
    graph, graph_edges = build_graph(sentences, dictionary, cache, extensions, parameters['E_0'], parameters['L_v'])
    
    nest_ids = [idx for idx, node in enumerate(graph) if node['type'] == 'sense']
    nest_ids.sort()
    
    word_ids = [idx for idx, node in enumerate(graph) if node['type'] == 'word']
    word_ids.sort()
    best_C = None
    C = [None for _ in range(len(word_ids))]
    best_score = 0
    
    
    next_ant_id = 0
    ants = {}
    
    for cycle in range(parameters['c_ac']):
        print("=== === ===")
        print(f"Cycle {cycle+1}/{parameters['c_ac']}")
        
        # Age all ants
        for ant_id in ants:
            ants[ant_id]['age'] += 1
        
        # (1) Eliminate dead ants
        to_eliminate = [(ant_id, ant) for ant_id, ant in ants.items() if ant['age'] >= parameters['omega']]
        for ant_id, ant in to_eliminate:
            graph[ant['position']]['energy'] += ant['energy']
            
#             print(f"Ant {ant_id} of {ant['motherland']} died at {ant['position']} - {str_node(graph[ant['position']])}")
        
            del ants[ant_id]
        print(f"Dead ants {len(to_eliminate)}")
            
        # (2) Generate new ants
        new_ants = 0
        for nest_id in nest_ids:
            nest = graph[nest_id]
            
            if nest['energy'] == 0:
                continue
            
            energy = nest['energy']
            prob = np.arctan(energy) / np.pi + 0.5
            should_generate = np.random.binomial(1, prob)
            
            if not should_generate:
                continue
               
            new_ants += 1
            ants[next_ant_id] = {
                'motherland': nest['sense'].name(),
                'energy': 1,
                'age': 0,
                'position': nest_id,
                'odour': nest['odour'], # copy by reference because this will never be modified
            }

            graph[nest_id]['energy'] -= 1
            
#             print(f"Ant {next_ant_id} was born: {ants[next_ant_id]}")
#             print(f"New energy of the nest: {graph[nest_id]['energy']}")
            
            next_ant_id += 1
        print(f"New ants {new_ants}")
    
        # Pre-compute lesk measure for all nodes and ants
        lesk_n_f = []
        
        for node in graph:
            row = {}
            
            for ant_id, ant in ants.items():
                lesk_score = extended_lesk_measure(node['odour'], ant['odour'], dictionary, cache, extensions)
                row[ant_id] = lesk_score
            
            lesk_n_f.append(row)
    
        # Pre-compute stuff for seeking ant movement
        eval_f_n_a = {}
        
        for node in graph:
            total_eval_f_n_a = 0
            
            total_nodes_energy = 0
            for edge in node['edges']:
                next_node = graph[edge['node']]
                total_nodes_energy += next_node['energy']
            
            for edge in node['edges']:
                next_node = graph[edge['node']]
                
                eval_f_n = next_node['energy'] / total_nodes_energy
                eval_f_a = 1 - graph_edges[edge['idx']]['pheromone']
                
                value = eval_f_n + eval_f_a
                total_eval_f_n_a += value
                
                if next_node['idx'] not in eval_f_n_a:
                    eval_f_n_a[next_node['idx']] = {}
                    
                eval_f_n_a[next_node['idx']][edge['idx']] = value
                
            if total_eval_f_n_a == 0:
                for edge in node['edges']:
                    next_node = graph[edge['node']]

                    if next_node['idx'] not in eval_f_n_a:
                        eval_f_n_a[next_node['idx']] = {}

                    eval_f_n_a[next_node['idx']][edge['idx']] = 1/len(node['edges'])

                continue
                
            for edge in node['edges']:
                next_node = graph[edge['node']]
                
                eval_f_n_a[next_node['idx']][edge['idx']] /= total_eval_f_n_a
            
        # Pre-compute stuff for returning ant movement
        return_eval_f_n_a = {}
        
        for node in graph:
            return_total_eval_f_n_a = 0
            
            total_lesk = 0
            for edge in node['edges']:
                next_node = graph[edge['node']]
                
                total_lesk += lesk_n_f[next_node['idx']][ant_id]
            
            if total_lesk == 0:
                for edge in node['edges']:
                    next_node = graph[edge['node']]
                
                    if next_node['idx'] not in return_eval_f_n_a:
                        return_eval_f_n_a[next_node['idx']] = {}
                
                    return_eval_f_n_a[next_node['idx']][edge['idx']] = 1/len(node['edges'])
                
                continue
            
            for edge in node['edges']:
                next_node = graph[edge['node']]
                
                return_eval_f_n = lesk_n_f[next_node['idx']][ant_id] / total_lesk
                return_eval_f_a = graph_edges[edge['idx']]['pheromone']
                
                value = return_eval_f_n + return_eval_f_a
                return_total_eval_f_n_a += value
                
                if next_node['idx'] not in return_eval_f_n_a:
                    return_eval_f_n_a[next_node['idx']] = {}
                    
                return_eval_f_n_a[next_node['idx']][edge['idx']] = value
               
            for edge in node['edges']:
                next_node = graph[edge['node']]
                
                return_eval_f_n_a[next_node['idx']][edge['idx']] /= return_total_eval_f_n_a
        
    
        # (3) Determine ant mode; move; create bridges
        moves = []
        
        for ant_id, ant in ants.items():
            return_mode_prob = ant['energy'] / parameters['E_max']
            return_mode = np.random.binomial(1, return_mode_prob)
            
            ant_node = graph[ant['position']]
            
            if return_mode:
                probs = [return_eval_f_n_a[edge['node']][edge['idx']] for edge in ant_node['edges']]
            else:
                probs = [eval_f_n_a[edge['node']][edge['idx']] for edge in ant_node['edges']]

             
            move_to = np.random.choice(ant_node['edges'], p=probs)
            moves.append((ant_id, move_to))
            
    
        # (4) Update the environment
        for ant_id, move_to in moves:
            next_node = graph[move_to['node']]
            
            graph_edges[move_to['idx']]['pheromone'] += parameters['theta']
            graph_edges[move_to['idx']]['pheromone'] = min(graph_edges[move_to['idx']]['pheromone'], 1)
            ants[ant_id]['position'] = next_node['idx']
            
            # Update the energy if we reached home or odours if we reached a regular node
            if next_node['type'] == 'sense' and next_node['sense'].name() == ants[ant_id]['motherland']:
                graph[move_to['node']]['energy'] += ants[ant_id]['energy'] - 1
                ants[ant_id]['energy'] = 1
            elif next_node['type'] != 'sense':
                # Try to add the ant's odour to the node
                for i, word in enumerate(ants[ant_id]['odour']):
                    if len(next_node['odour']) < parameters['L_v']:
                        graph[move_to['node']]['odour'].append(word)
                    else:
                        break
                
                # Replace random positons if the node is full
                for j in range(i, len(ants[ant_id]['odour'])):
                    rnd_idx = np.random.randint(0, len(graph[move_to['node']]['odour']))
                    graph[move_to['node']]['odour'][rnd_idx] = ants[ant_id]['odour'][j]
    
        for i in range(len(graph_edges)):
            graph_edges[i]['pheromone'] *= (1-parameters['delta'])
    
        # Compute the score of the generation
#         word_ids = [idx for idx, node in enumerate(graph) if node['type'] == 'word']
#         word_ids.sort()
#         C = [graph[graph[word_id]['edges'][1]['node']]['sense'] for word_id in word_ids]
    
        for i, word_id in enumerate(word_ids):
            best_energy = 0
            best_sense = graph[graph[word_id]['edges'][1]['node']]['sense']
            
            for edge in graph[word_id]['edges'][1:]:
                assert graph[edge['node']]['type'] == 'sense'
                
                if graph[edge['node']]['energy'] > best_energy:
                    best_energy = graph[edge['node']]['energy']
                    best_sense = graph[edge['node']]['sense']
                
            C[i] = best_sense
    
        score = fitness(C, dictionary, cache, extensions)
        print(f"Fitness score: {score}")
        
        if score > best_score:
            best_C = C.copy()
            best_score = score
    
        print("=== === ===")
        print()
        
    print(f"Solution (score = {best_score}):")
    for word_id, sense in zip(word_ids, best_C):
        print("#", graph[word_id]['word'], f"f{sense.name()} =", sense.definition())
        print()
    
def str_node(node):
    t = node['type']
    s = t
    
    if t == 'word':
        s += f'- {node["word"]}'
    elif t == 'sense':
        s += f'- {node["sense"].name()}'
        
    return s
    
def build_graph(sentences, dictionary, cache, extensions, initial_energy, odour_length):
    graph = [{
        'idx': 0, 
        'type': 'text', 
        'parent': None, 
        'edges': [],
        'energy': initial_energy,
        'odour': [],
    }]
        
    idx = 1
    edge_idx = 0
    
    for sentence in sentences:
        graph.append({
            'idx': idx,
            'type': 'sentence',
            'parent': 0,
            'edges': [],
            'energy': initial_energy,
            'odour': [],
        })
        
        graph[idx]['edges'].append({'node': 0, 'idx': edge_idx})        
        graph[0]['edges'].append({'node': idx, 'idx': edge_idx})
        edge_idx += 1

    
        sentence_idx = idx
        idx += 1
        
        for word, senses in sentence:
            graph.append({
                'idx': idx,
                'type': 'word',
                'parent': sentence_idx,
                'edges': [],
                'energy': initial_energy,
                'odour': [],
                
                'word': word,
            })
            
            graph[idx]['edges'].append({'node': sentence_idx, 'idx': edge_idx})            
            graph[sentence_idx]['edges'].append({'node': idx, 'idx': edge_idx})
            edge_idx += 1

            word_idx = idx
            idx += 1
            
            for sense in senses:
                graph.append({
                    'idx': idx,
                    'type': 'sense',
                    'parent': word_idx,
                    'edges': [],
                    'energy': initial_energy,
                    'odour': [],
                    
                    'sense': sense
                })
                
                graph[idx]['edges'].append({'node': word_idx, 'pheromone': 0, 'idx': edge_idx})                
                graph[word_idx]['edges'].append({'node': idx, 'pheromone': 0, 'idx': edge_idx})
                edge_idx += 1
                
                tokens = sense_to_tokens_set(sense, dictionary, cache, extensions)['simple']
                graph[idx]['odour'] = tokens[:odour_length]
                
                idx += 1
        
    edges = [{'pheromone': 0} for _ in range(edge_idx)]
    
    return graph, edges
    
text = """A retired clerk of the commissariat department came, too; he was
drunk, had a loud and most unseemly laugh and only fancy--was without
a waistcoat! One of the visitors sat straight down to the table without
even greeting Katerina Ivanovna. Finally one person having no suit
appeared in his dressing-gown, but this was too much, and the efforts of
Amalia Ivanovna and the Pole succeeded in removing him. The Pole brought
with him, however, two other Poles who did not live at Amalia Ivanovna’s
and whom no one had seen here before. All this irritated Katerina
Ivanovna intensely."""

ant_colony(text, default_extensions, default_parameters)