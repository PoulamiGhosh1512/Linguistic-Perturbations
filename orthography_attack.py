################### CHARACTER-LEVEL PERTURBATION ####################################
#####################################################################################
import random
from helper import get_homoglyph_dict

virama = '्্੍્୍்్್്'

def get_indices_homoglyph(word, lang):
    indices = []
    homoglyph = get_homoglyph_dict(lang)
    for i,char in enumerate(word):
        if char in homoglyph:
            indices.append(i)
    return indices

def get_homoglyph(char, lang):
    homoglyph = get_homoglyph_dict(lang)
    swap_char_list = homoglyph[char]
    return random.choice(swap_char_list)

def get_indices_virama(word):
    indices,word_len = [],len(word)
    for i,char in enumerate(word):
        if char in virama and i != word_len-1:
            indices.append(i)
    return indices

def get_candidates(word, lang, random_one_swap = False):
    #print('Find candidates for orthography-based attack...')
    if len(word) <= 1: 
        return []

    candidate_words = [] 
    
    homo_indices = get_indices_homoglyph(word, lang)
    
    debug_homo_indices = [(i,word[i]) for i in homo_indices]
    #print(f'Homoglyph indices:{debug_homo_indices}')
    
    if homo_indices:
        if random_one_swap:
            i = random.choice(homo_indices)
            candidate_word = word[:i] + get_homoglyph(word[i],lang) + word[i + 1:]
            candidate_words.append(candidate_word)
        else:
            for i in homo_indices:
                swap_key = get_homoglyph(word[i],lang)
                candidate_word = word[:i] + swap_key + word[i + 1 :]
                candidate_words.append(candidate_word) 
    #print(f'Candidate words:{candidate_words}')
    
    cc_indices = get_indices_virama(word) # cc in cc_indices stands for conjunct consonants
    
    debug_cc_indices = [(i,word[i]) for i in cc_indices]
    #print(f'Virama indices:{debug_cc_indices}')
    
    if cc_indices:
        if random_one_swap:
            i = random.choice(cc_indices)
            candidate_word = word[: i - 1] + word[i + 1] + word[i] + word[i-1] + word[i + 2 :]
            candidate_words.append(candidate_word)
        else:
            for i in cc_indices:
                candidate_word = word[: i - 1] + word[i + 1] + word[i] + word[i-1] + word[i + 2 :]
                candidate_words.append(candidate_word)    
    #print(f'Candidate words:{candidate_words}')
    
    return candidate_words

####################################################################################################################

def get_non_homoglyph(char, lang):
    homoglyph = get_homoglyph_dict(lang)
    swap_char_list = homoglyph[char]
    
    letters =['क','ख','ग','घ','ङ','च','छ','ज','झ','ञ','ट','ठ','ड','ढ','ण','त','थ','द',
              'ध','न','प','फ','ब','भ','म','य','र','ल','व','श','ष','स','ह']
    
    for c in swap_char_list:letters.remove(c)
    return random.choice(letters)

