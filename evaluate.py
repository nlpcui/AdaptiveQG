import sys, json, spacy, logging, torch, os
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
from rouge import Rouge
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
import numpy as np
from nltk import word_tokenize
import torch.nn.functional as F
from process_data import ascii_decode, ascii_encode
from tqdm import tqdm


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

        return {'difficulty_pccs': difficulty_pccs, 'diffculty_accuracy': difficulty_accuracy}    


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
    def __init__(self, num_words, user_ids=[], user_abilities=[], logits=[], labels=[], states=[], split_ids=[], valid_length=[], valid_interactions=[], label_pad_id=-100):
        self.user_ids = user_ids
        self.user_abilities = user_abilities
        self.logits = logits
        self.labels = labels
        self.states = states
        self.split_ids = split_ids
        self.label_pad_id = label_pad_id
        self.valid_length = valid_length
        self.valid_interactions = valid_interactions 
        self.num_words = num_words

    def compute_metrics(self):
        # ROC and F1 score
        # logits: [example_num, label_num(2)]
        # labels: [example_num, ]
        
        total_examples = len(self.user_ids)

        results = {
            'train': {'roc':0, 'f1_score':0, 'accuracy':0, 'recall':0, 'precision':0}, 
            'dev':  {'roc':0, 'f1_score':0, 'accuracy':0, 'recall':0, 'precision':0}, 
            'test':  {'roc':0, 'f1_score':0, 'accuracy':0, 'recall':0, 'precision':0}
        }
        
        collections = {
            'train': {'logits': [], 'labels': [], 'pred_labels': [], 'positive_probs': []},
            'dev': {'logits': [], 'labels': [], 'pred_labels': [], 'positive_probs': []},
            'test': {'logits': [], 'labels': [], 'pred_labels': [], 'positive_probs': []},
        }
        pbar = tqdm(total=len(self.user_ids))
        for example_id in range(total_examples):
            for sid, split in ([(1, 'train'), (2, 'dev'), (3, 'test')]):
                split_positions = np.where(self.split_ids[example_id] ==sid, True, False) # filter other splits
                non_pad_positions = np.where(self.labels[example_id]>=0, True, False)   # filter pad labels
                valid_positions = split_positions & non_pad_positions
                 
                if not valid_positions.any():
                    logging.debug('-- Single: user {} has no data for {} evaluation'.format(ascii_decode(self.user_ids[example_id]), split))
                    continue # no such split data
                

                valid_logits = self.logits[example_id][valid_positions] # flat
                valid_labels = self.labels[example_id][valid_positions] # flat
                
                pred_labels = np.argmax(valid_logits, axis=-1)
                positive_probs = F.softmax(torch.tensor(valid_logits), dim=-1)[:,1].numpy()            

                collections[split]['pred_labels'].extend(pred_labels)
                collections[split]['labels'].extend(valid_labels)
                collections[split]['positive_probs'].extend(positive_probs)
            pbar.update(1)
        pbar.close()
        
        logging.info('-- {} examples for evaluation; TRAIN: {} tokens; DEV: {} tokens; TEST: {} tokens.'.format(len(self.user_ids), len(collections['train']['labels']), len(collections['dev']['labels']), len(collections['test']['labels'])))        
         
        for split in ['train', 'dev', 'test']:
            collections[split]['pred_labels'] = np.array(collections[split]['pred_labels'])
            collections[split]['labels'] = np.array(collections[split]['labels'])
            collections[split]['positive_probs'] = np.array(collections[split]['positive_probs'])

            if len(collections[split]['pred_labels']) == 0 or len(collections[split]['labels']) == 0 or len(collections[split]['positive_probs']) == 0:
                logging.warning('-- Total: no data for {} evaluation'.format(split))
                continue

            results[split]['roc'] = roc_auc_score(y_true=collections[split]['labels'], y_score=collections[split]['positive_probs'])
            results[split]['f1_score'] = f1_score(y_true=collections[split]['labels'], y_pred=collections[split]['pred_labels'])
            results[split]['precision'] = precision_score(y_true=collections[split]['labels'], y_pred=collections[split]['pred_labels'])
            results[split]['recall'] = recall_score(y_true=collections[split]['labels'], y_pred=collections[split]['pred_labels'])
            results[split]['accuracy'] = accuracy_score(y_true=collections[split]['labels'], y_pred=collections[split]['pred_labels'])

        return results

    

    def read(self, dir_name):
        for file_name in os.listdir(dir_name):
            with open(os.path.join(dir_name, file_name), 'r') as fp:
                lines = fp.readlines()
                for line in lines:
                    data = json.loads(line.strip())
                    self.user_ids.append(np.array(data['user_id']))
                    self.user_abilities.append(np.array(data['user_ability']))
                    self.logits.append(np.array(data['logits']))
                    self.labels.append(np.array(data['labels']))
                    self.states.append(np.array(data['states']))
                    self.split_ids.append(np.array(data['split_ids']))
        '''
        self.user_ids = np.array(self.user_ids)
        self.user_abilities = np.array(self.user_abilities)
        self.logits = np.array(self.logits)
        self.labels = np.array(self.labels)
        self.states = np.array(self.states)
        self.split_ids = np.array(self.split_ids)
        '''

    def write(self, filename):
        fp = open(filename, 'w')
        pbar = tqdm(total=len(self.user_ids))         
        for idx in range(len(self.user_ids)):
            
            user_ability = self.user_abilities[idx].tolist()
            user_id = self.user_ids[idx].tolist()
            logits = self.logits[idx].tolist()[:self.valid_length[idx]]
            labels = self.labels[idx].tolist()[:self.valid_length[idx]]
            states = self.states[idx].tolist()[:self.valid_interactions[idx]]
            states = torch.nn.functional.softmax(torch.tensor(states), dim=-1) # int_num, word_num, 2
            states = states[:,:,1].numpy().tolist() # int_num, word_num, 1 
            split_ids = self.split_ids[idx].tolist()[:self.valid_length[idx]]
            ''' 
            logging.info('interaction num raw is {}'.format(valid_int_positions.shape))
            logging.info('logits shape {}'.format(self.logits[idx].shape))
            logging.info('states shape {}'.format(self.states[idx].shape))
            logging.info('logits to list {} {}'.format(self.logits[idx].tolist(), type(self.logits[idx].tolist()), len(self.logits[idx].tolist())))
            logging.info('logits to list cut {} {}'.format(self.logits[idx].tolist()[:seq_len], type(self.logits[idx].tolist()[:seq_len]), len(self.logits[idx].tolist()[:seq_len])))
            for i in range(len(states)):
                logging.info('{}th state is {}'.format(i, states[i]))
            logging.info('-- target logits {}'.format(logits))
            logging.info('-- target labels {}'.format(labels))
            logging.info('-- target splits {}'.format(split_ids))
            logging.info('-- target states {}'.format(states))
            logging.info('-- saving {}, seq_len {}, type {} interaction_num {}, type {}'.format(self.user_ids[idx], seq_len, type(seq_len), interaction_num, type(interaction_num))) 
            logging.info('-- saving {}, logit len {}, labels len {}, states len {}, split_ids len {}'.format(self.user_ids[idx], len(logits), len(labels), len(states), len(split_ids))) 
            new_states = self.states[idx][:self.valid_interactions[idx]]
            logging.info('-- new states shape format{}'.format(new_states.shape))
            new_logits = self.logits[idx][:self.valid_length[idx]]
            logging.info('-- new logits shape format{}'.format(new_logits.shape))
            exit(1)
            seq_len = int(np.where(self.labels[idx]==-200, False, True).sum())
            valid_int_positions = np.where((self.states[idx]==np.zeros([self.num_words, 2])).all(axis=1), False, True)
            interaction_num = int(valid_int_positions.sum())
            '''
            dump = {
                'user_id': user_id,
                'user_ability': user_ability,
                'logits': logits,
                'labels': labels,
                'states': states,
                'split_ids': split_ids
            }
            fp.write(json.dumps(dump)+'\n')
            pbar.update(1)
        pbar.close()
        fp.close()
    
