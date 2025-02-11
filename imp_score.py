import torch

def _get_masked(words, tokenizer):
    len_text = len(words)
    masked_words = []
    if tokenizer.name_or_path.split('/')[1].startswith('xlm'):
        for i in range(len_text):
            masked_words.append(words[0:i] + ['<unk>'] + words[i + 1:])
    else:
        for i in range(len_text):
            masked_words.append(words[0:i] + ['[UNK]'] + words[i + 1:]) 
    return masked_words

def get_important_scores(words, tgt_model, orig_prob, orig_label, orig_probs, tokenizer, batch_size, max_length):
    masked_words = _get_masked(words, tokenizer)
    texts = [' '.join(words) for words in masked_words] 
    inputs = tokenizer.batch_encode_plus(texts, add_special_tokens=True, truncation=True, padding = True, max_length=max_length, return_tensors='pt').to('cuda')
    output = tgt_model(**inputs)
    leave_1_logits = tgt_model(**inputs).logits
    leave_1_probs = torch.softmax(leave_1_logits, -1)    
    leave_1_probs_argmax = torch.argmax(leave_1_probs, dim=-1)
    import_scores = (orig_prob - leave_1_probs[:, orig_label]+ 
                     (leave_1_probs_argmax != orig_label).float()
                     * (leave_1_probs.max(dim=-1)[0] - torch.index_select(orig_probs, 0, leave_1_probs_argmax))
                     ).data.cpu().numpy()

    return import_scores

def get_important_scores_pre(words, hypo ,tgt_model, orig_prob, orig_label, orig_probs, tokenizer, batch_size, max_length):
    masked_words = _get_masked(words, tokenizer)
    texts = [(' '.join(words),hypo) for words in masked_words]  # list of text of masked words
    #print(f'Masked texts: {texts}')
    inputs = tokenizer.batch_encode_plus(texts, add_special_tokens=True, truncation=True, padding = True, max_length=max_length, return_tensors='pt').to('cuda')
    output = tgt_model(**inputs)
    leave_1_logits = tgt_model(**inputs).logits
    leave_1_probs = torch.softmax(leave_1_logits, -1)    
    leave_1_probs_argmax = torch.argmax(leave_1_probs, dim=-1)
    import_scores = (orig_prob - leave_1_probs[:, orig_label]+ 
                     (leave_1_probs_argmax != orig_label).float()
                     * (leave_1_probs.max(dim=-1)[0] - torch.index_select(orig_probs, 0, leave_1_probs_argmax))
                     ).data.cpu().numpy()

    return import_scores

def get_important_scores_hypo(premise, words, tgt_model, orig_prob, orig_label, orig_probs, tokenizer, batch_size, max_length):
    masked_words = _get_masked(words, tokenizer)
    texts = [(premise,' '.join(words)) for words in masked_words]  # list of text of masked words
    #print(f'Masked texts: {texts}')
    inputs = tokenizer.batch_encode_plus(texts, add_special_tokens=True, truncation=True, padding = True, max_length=max_length, return_tensors='pt').to('cuda')
    output = tgt_model(**inputs)
    leave_1_logits = tgt_model(**inputs).logits
    leave_1_probs = torch.softmax(leave_1_logits, -1)    
    leave_1_probs_argmax = torch.argmax(leave_1_probs, dim=-1)
    import_scores = (orig_prob - leave_1_probs[:, orig_label]+ 
                     (leave_1_probs_argmax != orig_label).float()
                     * (leave_1_probs.max(dim=-1)[0] - torch.index_select(orig_probs, 0, leave_1_probs_argmax))
                     ).data.cpu().numpy()

    return import_scores