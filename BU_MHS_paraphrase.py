# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import sys
import attr
from config import config
import spacy
# nltk.download('wordnet')  # cancel this comment for the first run
from nltk.corpus import wordnet as wn
from get_NE_list import NE_list
import numpy as np
from unbuffered import Unbuffered
from itertools import combinations
from functools import partial
from nltk import ngrams
from collections import Counter

sys.stdout = Unbuffered(sys.stdout)
nlp = spacy.load('en_core_web_sm')

# load bigram_candidates
bigram_candidate = np.load('bigram/bigram_syn_agnews.npy')  # change the path if use other dataset
bigram_candidate_list = []
for i in range(len(bigram_candidate)):
    bigram_candidate_list.append(list(bigram_candidate[i]))
bigrams_have_syns = [item[0] for item in bigram_candidate_list]

supported_pos_tags = [
    'CC',  # coordinating conjunction, like "and but neither versus whether yet so"
    # 'CD',   # Cardinal number, like "mid-1890 34 forty-two million dozen"
    # 'DT',   # Determiner, like all "an both those"
    # 'EX',   # Existential there, like "there"
    # 'FW',   # Foreign word
    # 'IN',   # Preposition or subordinating conjunction, like "among below into"
    'JJ',  # Adjective, like "second ill-mannered"
    'JJR',  # Adjective, comparative, like "colder"
    'JJS',  # Adjective, superlative, like "cheapest"
    # 'LS',   # List item marker, like "A B C D"
    # 'MD',   # Modal, like "can must shouldn't"
    'NN',  # Noun, singular or mass
    'NNS',  # Noun, plural
    'NNP',  # Proper noun, singular
    'NNPS',  # Proper noun, plural
    # 'PDT',  # Predeterminer, like "all both many"
    # 'POS',  # Possessive ending, like "'s"
    # 'PRP',  # Personal pronoun, like "hers herself ours they theirs"
    # 'PRP$',  # Possessive pronoun, like "hers his mine ours"
    'RB',  # Adverb
    'RBR',  # Adverb, comparative, like "lower heavier"
    'RBS',  # Adverb, superlative, like "best biggest"
    # 'RP',   # Particle, like "board about across around"
    # 'SYM',  # Symbol
    # 'TO',   # to
    # 'UH',   # Interjection, like "wow goody"
    'VB',  # Verb, base form
    'VBD',  # Verb, past tense
    'VBG',  # Verb, gerund or present participle
    'VBN',  # Verb, past participle
    'VBP',  # Verb, non-3rd person singular present
    'VBZ',  # Verb, 3rd person singular present
    # 'WDT',  # Wh-determiner, like "that what whatever which whichever"
    # 'WP',   # Wh-pronoun, like "that who"
    # 'WP$',  # Possessive wh-pronoun, like "whose"
    # 'WRB',  # Wh-adverb, like "however wherever whenever"
]


@attr.s
class SubstitutionCandidate:
    token_position = attr.ib()
    similarity_rank = attr.ib()
    original_token = attr.ib()
    candidate_word = attr.ib()


def vsm_similarity(doc, original, synonym):
    window_size = 3
    start = max(0, original.i - window_size)
    return doc[start: original.i + window_size].similarity(synonym)


def _get_wordnet_pos(spacy_token):
    '''Wordnet POS tag'''
    pos = spacy_token.tag_[0].lower()
    if pos in ['r', 'n', 'v']:  # adv, noun, verb
        return pos
    elif pos == 'j':
        return 'a'  # adj


def _synonym_prefilter_fn(token, synonym):
    '''
    Similarity heuristics go here
    '''
    if (len(synonym.text.split()) > 2 or (  # the synonym produced is a phrase
            synonym.lemma == token.lemma) or (  # token and synonym are the same
            synonym.tag != token.tag) or (  # the pos of the token synonyms are different
            token.text.lower() == 'be')):  # token is be
        return False
    else:
        return True


def _generate_synonym_candidates(token, token_position, dataset_dict, word_candidate, rank_fn=None):
    '''
    Generate synonym candidates.
    For each token in the doc, the list of WordNet synonyms is expanded.
    :return candidates, a list, whose type of element is <class '__main__.SubstitutionCandidate'>
            like SubstitutionCandidate(token_position=0, similarity_rank=10, original_token=Soft, candidate_word='subdued')
    '''
    if rank_fn is None:
        rank_fn = vsm_similarity
    candidates = []
    if token.tag_ in supported_pos_tags:
        wordnet_pos = _get_wordnet_pos(token)  # 'r', 'a', 'n', 'v' or None
        wordnet_synonyms = []

        synsets = wn.synsets(token.text, pos=wordnet_pos)
        for synset in synsets:
            wordnet_synonyms.extend(synset.lemmas())

        synonyms = []
        for wordnet_synonym in wordnet_synonyms:
            spacy_synonym = nlp(wordnet_synonym.name().replace('_', ' '))[0]
            synonyms.append(spacy_synonym)

        # get hownet synonyms
        if dataset_dict.dict.__contains__(token.lower_):
            id = dataset_dict.dict[token.lower_]
            id_syn_how = word_candidate[id]
            id_syn_how_all = []
            for value in id_syn_how.values():
                id_syn_how_all.extend(value)
            for synonym_id in id_syn_how_all:
                synonym_str = list(dataset_dict.dict.keys())[list(dataset_dict.dict.values()).index(synonym_id)]
                synonyms.append(nlp(synonym_str.replace('_', ' '))[0])
            synonyms = filter(partial(_synonym_prefilter_fn, token), synonyms)
        else:
            synonyms = filter(partial(_synonym_prefilter_fn, token), synonyms)

        candidate_set = set()
        for _, synonym in enumerate(synonyms):
            candidate_word = synonym.text
            if candidate_word in candidate_set:  # avoid repetition
                continue
            candidate_set.add(candidate_word)
            candidate = SubstitutionCandidate(
                token_position=token_position,
                similarity_rank=None,
                original_token=token,
                candidate_word=candidate_word)
            candidates.append(candidate)
    return candidates


def _compile_perturbed_tokens(doc, accepted_candidates):
    '''
    Traverse the list of accepted candidates and do the token substitutions.
    '''

    candidate_by_position = {}
    for candidate in accepted_candidates:
        candidate_by_position[candidate.token_position] = candidate

    final_tokens = []
    for position, token in enumerate(doc):
        word = token.text
        if position in candidate_by_position:
            candidate = candidate_by_position[position]
            word = candidate.candidate_word.replace('_', ' ')
        final_tokens.append(word)

    original_token = accepted_candidates[0].original_token
    if len(original_token) == 2 and isinstance(original_token, tuple):
        connect = '_'
        bigram_connect = connect.join([original_token[0].text, original_token[1].text])
        if bigram_connect in bigrams_have_syns:
            tail_position = accepted_candidates[0].token_position + 1
            # print(accepted_candidates)
            # print(doc.text)
            bigram_tail = doc[tail_position].text
            final_tokens.remove(bigram_tail)

    return final_tokens

def BU_MHS(
        doc,
        true_y,
        dataset,
        dataset_dict,
        word_candidate,
        word_saliency_list=None,
        rank_fn=None,
        heuristic_fn=None,
        halt_condition_fn=None,
        origin_perturbed_vector_fn=None,
        delta_P_fn=None,
        verbose=True):

    # defined in Eq.(8)
    def softmax(x):
        exp_x = np.exp(x)
        softmax_x = exp_x / np.sum(exp_x)
        return softmax_x

    def current_generation_comb(last_generation_comb, sorted_substitute_tuple_list, Max_attack_word):
        ''' Return candidate list of the current generation'''
        if last_generation_comb:
            current_generation_candidate = []
            sorted_last_generation_comb = sorted(last_generation_comb, key=lambda t: t[-1],
                                                 reverse=True)  # t[-1]: the last element
            for i in range(1):
                last_generation_i_comb = list(sorted_last_generation_comb[i][0:-1])
                sorted_substitute_tuple_list_i = sorted_substitute_tuple_list[:]
                for j in range(len(last_generation_i_comb)):
                    sorted_substitute_tuple_list_i.remove(last_generation_i_comb[j])
                for k in range(len(sorted_substitute_tuple_list_i)):
                    new_word_candidate_set = last_generation_i_comb[:]
                    new_word_candidate_set.append(sorted_substitute_tuple_list_i[k])
                    current_generation_candidate.append(new_word_candidate_set)
            return current_generation_candidate
        else:
            list_comb = list(combinations(sorted_substitute_tuple_list[0:Max_attack_word], 1))
            return list_comb

    heuristic_fn = heuristic_fn or (lambda _, candidate: candidate.similarity_rank)
    halt_condition_fn = halt_condition_fn or (lambda perturbed_text: False)
    delta_P_fn = delta_P_fn
    perturbed_doc = doc
    perturbed_text = perturbed_doc.text

    substitute_count = 0  # calculate how many substitutions used in a doc
    substitute_tuple_list = []  # save the information of substitute word

    word_saliency_array = np.array([word_tuple[2] for word_tuple in word_saliency_list])
    word_saliency_array = softmax(word_saliency_array)

    NE_candidates = NE_list.L[dataset][true_y]

    NE_tags = list(NE_candidates.keys())
    use_NE = True  # whether use NE as a substitute

    max_len = config.word_max_len[dataset]

    #######

    bigram_counts = Counter(ngrams(doc, 2))
    bigram_counts = list(bigram_counts)
    bigram_total = len(bigram_counts)
    connect = '_'
    candidates_bigram = []
    bigram_position_list = []
    for position in range(bigram_total):
        bigram = bigram_counts[position]
        bigram_connect = connect.join([bigram[0].text, bigram[1].text])
        bigram_doc = nlp(bigram_connect)
        bigram_token = bigram_doc[0]
        if bigram_connect in bigrams_have_syns:
            index = bigrams_have_syns.index(bigram_connect)
            candidates_bigram_position = bigram_candidate_list[index][2]
            for candidate_bigram in candidates_bigram_position:
                candidate_bigram_tuple = SubstitutionCandidate(
                    token_position=position,
                    similarity_rank=None,
                    original_token=bigram,
                    candidate_word=candidate_bigram)
                candidates_bigram.append(candidate_bigram_tuple)
            sorted_bigram_candidates = zip(map(partial(heuristic_fn, doc.text), candidates_bigram), candidates_bigram)
            sorted_bigram_candidates = list(sorted(sorted_bigram_candidates, key=lambda t: t[0]))
            delta_p_star_bigram, substitute_bigram = sorted_bigram_candidates.pop()
            substitute_tuple_list.append(
                (position, bigram_connect, substitute_bigram, delta_p_star_bigram, bigram_token.tag_))
            bigram_position_list.append(position)
            bigram_position_list.append(position+1)

    # for each word w_i in x, use WordNet to build a synonym set L_i
    for (position, token, word_saliency, tag) in word_saliency_list:
        if position >= max_len:
            break
        if position in bigram_position_list:
            continue

        candidates = []
        if use_NE:
            NER_tag = token.ent_type_
            if NER_tag in NE_tags:
                candidate = SubstitutionCandidate(position, 0, token, NE_candidates[NER_tag])
                candidates.append(candidate)
            else:
                candidates = _generate_synonym_candidates(token=token, token_position=position, dataset_dict=dataset_dict, word_candidate=word_candidate, rank_fn=rank_fn)
        else:
            candidates = _generate_synonym_candidates(token=token, token_position=position, dataset_dict=dataset_dict, word_candidate=word_candidate, rank_fn=rank_fn)

        if len(candidates) == 0:

            continue
        perturbed_text = perturbed_doc.text

        # The substitute word selection method R(w_i;S_i) defined in Eq.(7)
        sorted_candidates = zip(map(partial(heuristic_fn, doc.text), candidates), candidates)
        # Sorted according to the return value of heuristic_fn function, i.e., candidate importance score Eq.(4)
        sorted_candidates = list(sorted(sorted_candidates, key=lambda t: t[0]))

        # delta_p_star is defined in Eq.(8); substitute is w_i^*
        delta_p_star, substitute = sorted_candidates.pop()
        substitute_tuple_list.append(
            (position, token.text, substitute, delta_p_star, token.tag_))

    # sort all the words w_i in x in descending order based on delta_p_star
    sorted_substitute_tuple_list = sorted(substitute_tuple_list, key=lambda t: t[3], reverse=True)

    Max_attack_word = min(20, len(sorted_substitute_tuple_list))  # upper bound
    last_generation_comb = []
    NE_count = 0
    change_tuple_list = []
    for i in range(1, Max_attack_word+1):
        list_comb = current_generation_comb(last_generation_comb, sorted_substitute_tuple_list, Max_attack_word)
        generation_comb = []
        substitute_count += 1
        change_tuple_list = []
        for j in range(len(list_comb)):
            tuple_comb_j = list_comb[j]
            # Attack_comb = []
            change_tuple_list = []
            NE_count = 0
            perturbed_doc = doc
            perturbed_text = perturbed_doc.text
            candidates_j = []
            for (position, token, substitute, score, tag) in tuple_comb_j:
                # Attack_comb.append((position, token, substitute, score, tag))
                if nlp(token)[0].ent_type_ in NE_tags:
                    NE_count += 1
                if len(substitute.original_token) == 2 and isinstance(substitute.original_token, tuple):
                    bigram_count += 1
                change_tuple_list.append((position, token, substitute, score, tag))
                perturbed_text = ' '.join(_compile_perturbed_tokens(perturbed_doc, [substitute]))
                perturbed_doc = nlp(perturbed_text)
                candidates_j.append(substitute)
            origin_vector, perturbed_vector = origin_perturbed_vector_fn(doc.text, candidates_j)
            prob_shift = delta_P_fn(origin_vector, perturbed_vector)
            list_comb_j = list(tuple_comb_j)
            list_comb_j.append(prob_shift)
            generation_comb.append(tuple(list_comb_j))
            last_generation_comb = generation_comb
            if halt_condition_fn(perturbed_text):
                if verbose:
                    print("use", substitute_count, "substitution; use", NE_count, 'NE')
                sub_word = substitute_count
                sub_rate = substitute_count / len(doc)
                NE_rate = NE_count / substitute_count
                return perturbed_text, sub_word, sub_rate, NE_rate, change_tuple_list

    if verbose:
        print("use", substitute_count, "substitution; use", NE_count, 'NE')
    sub_word = substitute_count
    sub_rate = substitute_count / len(doc)
    if substitute_count == 0:
        NE_rate = 0
    else:
        NE_rate = NE_count / substitute_count

    return perturbed_text, sub_word, sub_rate, NE_rate, change_tuple_list
