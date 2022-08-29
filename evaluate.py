import sys, json, spacy, logging
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
from rouge import Rouge
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
import numpy as np
from nltk import word_tokenize
import torch.nn.functional as F
from process_data import ascii_decode, ascii_encode


class QGEvaluator:
    def __init__(self, word_file, prompt_words, generated, reference, difficulty_scores, difficulty_levels):

        self.word_difficulty = {}
        df = pd.read_csv(word_file)
        for idx, row in df.iterrows():
            self.word_difficulty[row['word']] = row['error_rate']
        
        # logging.info('-- word_difficulty {}'.format(self.word_difficulty))
        
        self.prompt_words = prompt_words
        self.generated = generated
        self.reference = reference
        self.difficulty_scores = difficulty_scores
        self.difficulty_levels = difficulty_levels
        self.generated_difficulty_scores = []
        self.generated_difficulty_levels = []

        self.rouge = Rouge()
        self.tokenizer = word_tokenize

        self.rouge_scores = []
        self.coverage = []

    def compute_metrics(self):
        result = {}
        
        rouge_score = self.compute_rouge()
        result.update(rouge_score)
         
        difficulty_consistency = self.compute_difficulty_consistency()
        result.update(difficulty_consistency)

        coverage = self.compute_coverage()
        result.update(coverage)
         
        return result


    def compute_rouge(self):
        for idx in range(len(self.generated)):
            try:
                score = self.rouge.get_scores([self.generated[idx]], [self.reference[idx]])
                self.rouge_scores.append(score[0])
            except Exception:
                self.rouge_scores.append({'rouge-1':{'r':0, 'p':0, 'f':0}, 'rouge-2':{'r':0, 'p':0, 'f':0}, 'rouge-l':{'r':0, 'p':0, 'f':0}})

        rouge_1 = rouge_2 = rouge_l = 0
        for score in self.rouge_scores:
            rouge_1 += score['rouge-1']['f']
            rouge_2 += score['rouge-2']['f']
            rouge_l += score['rouge-l']['f']
       
        return {'rouge-1': rouge_1/len(self.generated), 'rouge-2':rouge_2/len(self.generated), 'rouge-l': rouge_l/len(self.generated)}
       

    def compute_coverage(self):
        cover_cnt = 0
        total_cnt = 0
        for idx in range(len(self.prompt_words)):
            prompt_words = self.tokenizer(self.prompt_words[idx])
            generated_words = self.tokenizer(self.generated[idx])
            total_cnt += len(prompt_words)
            for word in prompt_words:
                if word in generated_words:
                    cover_cnt += 1
        
        return {'coverage': cover_cnt/total_cnt}


    def compute_difficulty_consistency(self):
        for idx in range(len(self.generated)):
            difficulty_score = self.__compute_difficulty_score(self.generated[idx])
            difficulty_level = min(3, difficulty_score//0.5)
            self.generated_difficulty_scores.append(difficulty_score)
            self.generated_difficulty_levels.append(difficulty_level)

        difficulty_pccs = np.corrcoef(self.difficulty_scores, self.generated_difficulty_scores)[0][1] # pearson coefficient  
        difficulty_accuracy = accuracy_score(y_true=self.difficulty_levels, y_pred=self.generated_difficulty_levels)

        return {'difficulty_pccs': difficulty_pccs, 'diffculty_accuarcy': difficulty_accuracy}    


    def __compute_difficulty_score(self, sentence):
        difficulty_score = 0
        words = self.tokenizer(sentence)
        # logging.info('-- sentence {}'.format(sentence))
        # logging.info('-- words {}'.format(words))
        for word in words:
            if word not in self.word_difficulty:
                continue # TODO: how to handle?
            difficulty_score += self.word_difficulty[word]
        return difficulty_score


    def __compute_difficulty_level(self, difficulty_score):
        return min(3, difficulty_score // 0.5)

    
    def output_result(self, filepath):
        with open(filepath, 'w') as fp:
            for idx in range(len(self.prompt_words)):
                fp.write(json.dumps({'prompt_words': self.prompt_words[idx], 'difficulty_score': self.difficulty_scores[idx], 'difficulty_level': self.difficulty_levels[idx], 'generated': self.generated[idx], 'reference': self.reference[idx], 'generated_difficulty_score': self.generated_difficulty_scores[idx], 'generated_difficulty_level': self.generated_difficulty_levels[idx]})+'\n')
        

class KTEvaluator:
    def __init__(self, user_ids=[], user_abilities=[], logits=[], labels=[], states=[], split_ids=[], label_pad_id=-100):
        self.user_ids = user_ids
        self.user_abilities = user_abilities
        self.logits = logits
        self.labels = labels
        self.states = states
        self.split_ids = split_ids
        self.label_pad_id = label_pad_id
    

    def compute_metrics(self):
        # ROC and F1 score
        # logits: [example_num, label_num(2)]
        # labels: [example_num, ]

        results = {
            'train': {'roc':0, 'f1_score':0, 'accuracy':0, 'recall':0, 'precision':0}, 
            'dev':  {'roc':0, 'f1_score':0, 'accuracy':0, 'recall':0, 'precision':0}, 
            'test':  {'roc':0, 'f1_score':0, 'accuracy':0, 'recall':0, 'precision':0}
        }

        total_examples = len(self.user_ids)
        
        print('split_ids', self.split_ids)
        print('logits', self.logits)
        print('labels', self.labels)

        for sid, split in ([(1, 'train'), (2, 'dev'), (3, 'test')]):
            valid_positions = np.where(self.split_ids==sid, True, False) # filter pad positions and other splits
            if not valid_positions.any():
                continue # no such split data
            valid_logits = self.logits[valid_positions] # flat
            valid_labels = self.labels[valid_positions] # flat
            
            pred_labels = np.argmax(valid_logits, axis=-1)
            positive_probs = F.softmax(torch.tensor(valid_logits), dim=-1)[:1].numpy()

            results[split]['roc'] = roc_auc_score(y_true=valid_labels, y_score=positive_probs)
            results[split]['f1_score'] = f1_score(y_true=valid_labels, y_pred=pred_labels)
            results[split]['precision'] = precision(y_true=valid_labels, y_pred=pred_labels)
            results[split]['recall'] = precision(y_true=valid_labels, y_pred=pred_labels)
            results[split]['accuarcy'] = accuracy_score(y_true=valid_labels, y_pred=pred_labels)

        return results

    

    def read(self, file_name):
        with open(file_name, 'r') as fp:
            for line in fp.readlines():
                data = json.loads(line.strip())
                self.user_ids.append(np.array(data['user_id']))
                self.user_abilities.append(data['user_ability'])
                self.logits.append(np.array(data['logits']))
                self.labels.append(np.array(data['labels']))
                self.states.append(np.array(data['states']))
                self.split_ids.append(np.array(data['split_ids']))

    
    def write(self, dir_name):
        
        for idx in range(len(self.user_ids)):
            dump = {
                'user_id': self.user_ids[idx],
                'user_ability': self.user_abilities[idx],
                'logits': self.logits[idx].tolist(),
                'labels': self.labels[idx].tolist(),
                'states': self.states[idx].tolist(),
                'split_ids': self.split_ids[idx].tolist()
            }
            file_name = ascii_decode(self.user[idx])
            with open(os.path.join(dir_name, file_name), 'w') as fp:
                fp.write(json.dumps(file_name+'\n'))
