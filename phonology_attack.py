################### CHARACTER-LEVEL PERTURBATION : PHONOLOGY ATTACK ###################
import random
from indicnlp.transliterate.unicode_transliterate import UnicodeIndicTransliterator

swap_all = {"क":["ख","ग","घ"],"ख":["क","ग","घ"],"ग":["क","ख","घ"],"घ":["क","ख","ग"],
            "च":["छ","ज","झ"],"छ":["च","ज","झ"],"ज":["च","छ","झ"],"झ":["च","छ","ज"],
            "ट":["ठ","ड","ढ"],"ठ":["ट","ड","ढ"],"ड":["ट","ठ","ढ"],"ढ":["ट","ठ","ड"],
            "त":["थ","द","ध"],"थ":["त","द","ध"],"द":["त","थ","ध"],"ध":["त","थ","द"],
            "प":["फ","ब","भ"],"फ":["प","ब","भ"],"ब":["प","फ","भ"],"भ":["प","फ","ब"],                                                       
            "श":["ष","स"],"ष":["श","स"],"स":["ष","श"],
            "अ":["आ"],"आ":["अ"],"इ":["ई"],"ई":["इ"],"उ":["ऊ"],"ऊ":["उ"],"ए":["ऐ"],"ऐ":["ए"],"ओ":["औ"],"औ":["ओ"],
            "ा":[""],"ि":["ी"],"ी":["ि"],"ु":["ू"],"ू":["ु"],"े":["ै"],"ै":["े"],"ो":["ौ"],"ौ":["ो"]}

swap_consonants = {"क":["ख","ग","घ"],"ख":["क","ग","घ"],"ग":["क","ख","घ"],"घ":["क","ख","ग"],
                   "च":["छ","ज","झ"],"छ":["च","ज","झ"],"ज":["च","छ","झ"],"झ":["च","छ","ज"],
                   "ट":["ठ","ड","ढ"],"ठ":["ट","ड","ढ"],"ड":["ट","ठ","ढ"],"ढ":["ट","ठ","ड"],
                   "त":["थ","द","ध"],"थ":["त","द","ध"],"द":["त","थ","ध"],"ध":["त","थ","द"],
                   "प":["फ","ब","भ"],"फ":["प","ब","भ"],"ब":["प","फ","भ"],"भ":["प","फ","ब"]}

swap_vowels = {"अ":["आ"],"आ":["अ"],"इ":["ई"],"ई":["इ"],"उ":["ऊ"],"ऊ":["उ"],"ए":["ऐ"],"ऐ":["ए"],"ओ":["औ"],"औ":["ओ"],
               "ा":[""],"ि":["ी"],"ी":["ि"],"ु":["ू"],"ू":["ु"],"े":["ै"],"ै":["े"],"ो":["ौ"],"ौ":["ो"]}

swap_aspirated = {"क":"ख","ख":"क","ग":"घ","घ":"ग","च":"छ","छ":"च","ज":"झ","झ":"ज",
                  "ट":"ठ","ठ":"ट","ड":"ढ","ढ":"ड","त":"थ","थ":"त","द":"ध","ध":"द",
                  "प":"फ","फ":"प","ब":"भ","भ":"ब"}

swap_voiced = {"क":"ग","ख":"घ","ग":"क","घ":"ख","च":"ज","छ":"झ","ज":"च","झ":"छ",
               "ट":"ड","ठ":"ढ","ड":"ट","ढ":"ठ","त":"द","थ":"ध","द":"त","ध":"थ",
               "प":"ब","फ":"भ","ब":"प","भ":"फ"}

swap_sibilants = {"श":["ष","स"],"ष":["श","स"],"स":["ष","श"]}

def get_char_to_be_swappped(atype):
    if atype == 'all':char_to_be_swappped = swap_all
    elif atype == 'consonants':char_to_be_swappped = swap_consonants
    elif atype == 'vowels':char_to_be_swappped = swap_vowels
    elif atype == 'aspirated':char_to_be_swappped = swap_aspirated
    elif atype == 'voiced':char_to_be_swappped = swap_voiced
    elif atype == 'sibilants':char_to_be_swappped = swap_sibilants
    else:print("Invalid Attack Type")
    return char_to_be_swappped

def get_swap_character(s,char_to_be_swappped):
    if s in char_to_be_swappped:
        swap_char = random.choice(char_to_be_swappped[s])
        return swap_char

def get_indices(word,char_to_be_swappped):
    indices = []
    for i,char in enumerate(word):
        if char in char_to_be_swappped:
            indices.append(i)
    return indices

def get_candidates(word, lang, atype, random_one_swap = False):
    '''
    Generate adversarial candidates for the target word `word`
    atype : attack type (Possible values - vowel, consonants, aspirated, voiced, sibilants, all)
    '''
    if len(word) <= 1:
        return []

    candidate_words = []
    
    if lang!='hi':
        word = UnicodeIndicTransliterator.transliterate(word,lang,"hi")
    
    char_to_be_swappped = get_char_to_be_swappped(atype)
    
    indices = get_indices(word,char_to_be_swappped)
    
    if not indices:
        return []

    if random_one_swap:
        i = random.choice(indices)
        candidate_word = (word[:i] + get_swap_character(word[i],char_to_be_swappped) + word[i + 1:])
        candidate_word = UnicodeIndicTransliterator.transliterate(candidate_word,'hi',lang) if lang!='hi' else candidate_word
        candidate_words.append(candidate_word)
    else:
        for i in indices:
            swap_key = get_swap_character(word[i],char_to_be_swappped)
            candidate_word = word[:i] + swap_key + word[i + 1 :]
            candidate_word = UnicodeIndicTransliterator.transliterate(candidate_word,'hi',lang) if lang!='hi' else candidate_word
            candidate_words.append(candidate_word)            
    return candidate_words
