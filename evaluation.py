import constraints
import config as c
from indicnlp.tokenize import indic_tokenize 
import numpy as np

lang = c.lang

def evaluate(features):
    labse_sim,chrf_sim,bert_score_sim,phonetic_sim_sent,phonetic_sim_changes = [],[],[],[],[]
    
    labse_thres,chrf_thres,bert_score_thres = 0.6,0.6,0.6
    
    acc = 0
    origin_miss = 0
    total = 0
    total_q = 0
    total_change = 0
    total_word = 0
    skipped_examples = 0
    
    for i,feat in enumerate(features):
        if feat.success > 2:
            feat.labse_score = float(constraints.get_semantic_sim([feat.seq], [feat.final_adverse])) # Semantic similarity
            feat.chrf_score = float(constraints.get_chrf_overlap(feat.final_adverse,feat.seq)) # Overlap-based similarity
            feat.bert_score = float(constraints.get_bert_score([feat.final_adverse],[feat.seq]))
            psim_score_sent = float(constraints.get_phonetic_sim_sent(feat.final_adverse,feat.seq,lang)) if lang not in ['bd'] else 0.0 # Bodo is not a supported language
        
            if feat.labse_score < labse_thres or feat.chrf_score < chrf_thres or feat.bert_score < bert_score_thres:
                continue
            
            labse_sim.append(feat.labse_score)
            chrf_sim.append(feat.chrf_score)
            bert_score_sim.append(feat.bert_score)
            phonetic_sim_sent.append(psim_score_sent)
            
            total_q += feat.query
            total_change += feat.change
            total_word += len(indic_tokenize.trivial_tokenize(feat.seq))

            if feat.success == 3: 
                origin_miss += 1
                
        total += 1
    
    suc = float(acc / total)

    query = float(total_q / acc) 
    change_rate = float(total_change / total_word)

    origin_acc = 1 - origin_miss / total
    after_atk = 1 - suc

    return(origin_acc, after_atk, suc, query, change_rate,np.mean(labse_sim),np.mean(chrf_sim),np.mean(bert_score_sim),np.mean(phonetic_sim_sent),total)