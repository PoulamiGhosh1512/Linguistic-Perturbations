import constraints
import config as c
from indicnlp.tokenize import indic_tokenize 
import numpy as np

lang = c.lang

def evaluate(features,perturbed_feat):
    sem_sim = [] # For the overall dataset
    overlap_sim = [] # For the overall dataset
    bert_score_sim = []
    phonetic_sim_sent = []
    
    sim_thres = 0.6 # Threshold for semantic similarity
    chrf_thes = 0.6 # Threshold for overlap-based similarity
    bert_score_thres = 0.6
    
    acc = 0
    origin_miss = 0
    total = 0
    total_q = 0
    total_change = 0
    total_word = 0
    skipped_examples = 0 # Examples skipped since they were not meeting the constraints
    
    for i,feat in enumerate(features):
        if i%100 ==0:print(i)
        if feat.success > 2:
            seq = feat.seq[0] if perturbed_feat in ['premise','sentence1'] else feat.seq[1]
            adverse = feat.final_adverse[0] if perturbed_feat in ['premise','sentence1'] else feat.final_adverse[1]
            feat.labse_score = float(constraints.get_semantic_sim([seq], [adverse])) # Semantic similarity
            feat.chrf_score = float(constraints.get_chrf_overlap(adverse,seq)) # Overlap-based similarity
            feat.bert_score = float(constraints.get_bert_score([adverse],[seq]))   
            try:
                psim_score_sent = float(constraints.get_phonetic_sim_sent(adverse,seq,lang)) if lang not in ['bd'] else 0.0 
            except ZeroDivisionError:
                continue
            
            if feat.labse_score < sim_thres or feat.chrf_score < chrf_thes or feat.bert_score < bert_score_thres:
                continue
            
            sem_sim.append(feat.labse_score)
            overlap_sim.append(feat.chrf_score)
            bert_score_sim.append(feat.bert_score)
            phonetic_sim_sent.append(psim_score_sent)
            acc += 1 
            total_q += feat.query
            total_change += feat.change
            total_word += len(indic_tokenize.trivial_tokenize(seq))

            if feat.success == 3: 
                origin_miss += 1 
                
        total += 1
    
    suc = float(acc / total) 

    query = float(total_q / acc) 
    change_rate = float(total_change / total_word)

    origin_acc = 1 - origin_miss / total 
    after_atk = 1 - suc 

    return(origin_acc, after_atk, suc, query, change_rate,np.mean(sem_sim),np.mean(overlap_sim),np.mean(bert_score_sim),np.mean(phonetic_sim_sent),total)