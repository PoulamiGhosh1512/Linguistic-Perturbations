from indicnlp.script import indic_scripts as isc
from indicnlp.script import phonetic_sim as psim

from sentence_transformers import SentenceTransformer,util
import nltk.translate.chrf_score
from evaluate import load
bertscore = load("bertscore")

import string
punc = string.punctuation
indic_punc = punc + '।৷'

model = SentenceTransformer("sentence-transformers/LaBSE")

def get_semantic_sim(sents1, sents2):
    embeddings1 = model.encode(sents1)
    embeddings2 = model.encode(sents2)
    cosine_score = util.cos_sim(embeddings1, embeddings2)
    return cosine_score

def get_chrf_overlap(transformed_text, reference_text):
    ref = reference_text.split(' ')
    hyp = transformed_text.split(' ')
    chrf = nltk.translate.chrf_score.sentence_chrf(ref, hyp)
    return chrf
    
def get_bert_score(predictions,references):
    results = bertscore.compute(predictions=predictions, references=references, model_type='xlm-roberta-large')
    return results['f1'][0]

def get_phonetic_sim_sent(predictions,references,lang):
    references  = references.replace(' ','') # Remove extra spaces if any
    predictions  = predictions.replace(' ','') # Remove extra spaces if any

    assert len(predictions) == len(references), "Length Mismatch in Phonetic Similarity calculation"

    sentlen = len(references) 
    sims = [] 
    
    for i in range(sentlen):
        if references[i] in indic_punc: continue
        vec1 = isc.get_phonetic_feature_vector(references[i],lang)
        vec2 = isc.get_phonetic_feature_vector(predictions[i],lang)
        sim = psim.cosine(vec1,vec2)
        sims.append(sim)
        
    phonetic_sim = sum(sims)/len(sims) 
    return phonetic_sim

def get_phonetic_sim_word(word1,word2,lang):
    wordlen = len(word1)

    sims = [] 
    
    for i in range(wordlen):
        vec1 = isc.get_phonetic_feature_vector(word1[i],lang)
        vec2 = isc.get_phonetic_feature_vector(word2[i],lang)
        sim = psim.cosine(vec1,vec2)
        sims.append(sim)
        
    phonetic_sim = sum(sims)/len(sims) 
    return phonetic_sim
    
def get_phonetic_sim_changes(changes,lang):
    overall_sim = []
    #AA_vowel_sign = ["ा","া","ਾ","ા","ା","ா","ా","ಾ","ാ"]
    for change in changes:
        word1,word2 = change[0],change[1]
        try:
            sim = get_phonetic_sim_word(word1,word2,lang)
        except AssertionError:
            # Dropping a AA_vowel_sign is equivalent to replacing an AA_vowel with an A_vowel such as आ with अ.
            # Apart from AA_vowel_sign dropping, no other vowel swap causes a decrease in the length of the perturbed word. 
            vec1 = isc.get_phonetic_feature_vector('आ','hi')
            vec2 = isc.get_phonetic_feature_vector('अ','hi')
            sim = psim.cosine(vec1,vec2)            
        overall_sim.append(sim)
    return sum(overall_sim)/len(overall_sim)
        

        
    