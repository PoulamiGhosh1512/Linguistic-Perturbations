import torch
import copy

import imp_score


import orthography_attack
import phonology_attack
import random_char_substitution

from indicnlp.tokenize import indic_tokenize
import string

punc = string.punctuation
indic_punc = punc + '।৷'

def attack(feature, tgt_model, mlm_model, lang, tokenizer, k, batch_size, max_length=512, cos_mat=None, w2i={}, i2w={}):
    
    filter_words = []
    
    words = indic_tokenize.trivial_tokenize(feature.seq)

    # Running prediction on original uncorrupted text
    inputs = tokenizer.encode_plus(feature.seq, None, add_special_tokens=True, truncation=True, max_length=max_length, return_tensors='pt').to('cuda') 
    orig_logits = tgt_model(**inputs)[0].squeeze() # Logits    
    orig_probs = torch.softmax(orig_logits, -1) # Softmax, Logits -> Probability
    orig_label = torch.argmax(orig_probs) # Predicted label for original text sequence
    #print('Softmax probabilities:',orig_probs,'Original Label:',orig_label)
    current_prob = orig_probs.max()
    feature.orig_prob = current_prob.item()

    if orig_label != feature.label:
        feature.success = 3 
        return feature
            
    important_scores = imp_score.get_important_scores(words, tgt_model, current_prob, orig_label, orig_probs, tokenizer, batch_size, max_length)
    list_of_index = sorted(enumerate(important_scores), key=lambda x: x[1], reverse=True) 
    
    word_list = [[words[i],score] for (i,score) in list_of_index]

    feature.query += int(len(words))
    final_words = copy.deepcopy(words)
        
    for top_index in list_of_index: # Iterating over the words in decreasing order of importance    
        if feature.change > int(0.4 * (len(words))):
            feature.success = 1  # exceed
            return feature

        tgt_word = words[top_index[0]] 
        if tgt_word in filter_words:
            continue
            
        if tgt_word in indic_punc:
            continue
        
        substitutes = orthography_attack.get_candidates(tgt_word, lang) 
        feature.candidates[tgt_word] = substitutes
        
        most_gap = 0.0 
        candidate = None

        for substitute_ in substitutes:
            substitute = substitute_ 
            
            if substitute == tgt_word:                
                continue
            if substitute in filter_words:
                continue
                        
            temp_replace = copy.deepcopy(final_words)
            temp_replace[top_index[0]] = substitute 
            temp_text = ' '.join(temp_replace)        
            
            # Running prediction on corrupted text
            inputs = tokenizer.encode_plus(temp_text, None, add_special_tokens=True, truncation=True, max_length=max_length, return_tensors='pt').to('cuda')            
            temp_logits = tgt_model(**inputs)[0].squeeze()
        
            feature.query += 1

            temp_prob = torch.softmax(temp_logits, -1)
            temp_label = torch.argmax(temp_prob) 
            
            if temp_label != orig_label: # Case 1: Perturbation resulted in misclassification
                #print('Attack successful')
                feature.change += 1
                final_words[top_index[0]] = substitute
                feature.changes.append([tgt_word, substitute])
                feature.final_adverse = temp_text
                feature.success = 4
                
                adv_prob = temp_prob.max()
                feature.adv_prob = adv_prob.item()
                feature.adv_label = temp_label
                return feature
            else: # Case 2: Perturbation doesn't change the prediction of the model i.e. the model is still predicting correctly even after perturbation 
                label_prob = temp_prob[orig_label]
                gap = current_prob - label_prob
                if gap > most_gap: # Keep track of the substitute that results in most decrease in original class probability 
                    most_gap = gap
                    candidate = substitute
        
        if most_gap > 0:
            feature.change += 1
            feature.changes.append([tgt_word, candidate])
            current_prob = current_prob - most_gap
            final_words[top_index[0]] = candidate
    
    feature.final_adverse = (' '.join(final_words))
    feature.success = 2
    return feature            
