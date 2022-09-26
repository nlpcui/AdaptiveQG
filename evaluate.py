import sys, json, spacy, logging, torch, os
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
from rouge import Rouge
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
import numpy as np
from nltk import word_tokenize
import torch.nn.functional as F
from utils import *
from tqdm import tqdm


class QGEvaluator:
    def __init__(self, word_file):

        self.word_difficulty = {}
        df = pd.read_csv(word_file)
        for idx, row in df.iterrows():
            self.word_difficulty[row['word']] = row['error_rate']
        
        # logging.info('-- word_difficulty {}'.format(self.word_difficulty))
        
        self.prompt_words = []
        self.generated = []
        self.reference = []
        self.difficulty_scores = []
        self.difficulty_levels = []
        self.generated_difficulty_scores = []
        self.generated_difficulty_levels = []

        self.rouge = Rouge()
        self.tokenizer = word_tokenize

        self.rouge_scores = []
        self.coverage = []


    def read(self, filename):
        with open(filename, 'r') as fp:
            for line in fp.readlines():
                data = json.loads(line.strip())
                self.prompt_words.append(self.tokenizer(data['prompt_words']))
                self.generated.append(data['generated'])
                self.reference.append(data['reference'])
                self.difficulty_levels.append(data['difficulty_level'])
                self.difficulty_scores.append(data['difficulty_score'])
                self.generated_difficulty_scores.append(data['generated_difficulty_score'])
                self.generated_difficulty_levels.append(data['generated_difficulty_level'])


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
        

    def update_difficulty(self):
        # update difficulty using cur word file
        for idx in range(len(self.generated)):
            words = self.tokenizer(self.reference[idx])
            new_diff = 0
            for word in words:
                if word not in self.word_difficulty:
                    continue
                new_diff += self.word_difficulty[word]

            self.difficulty_scores[idx] = new_diff


class KTEvaluator:
    def __init__(self, num_words, label_pad_id=-100):
        self.data = []
        self.num_words = num_words

    def compute_metrics(self):
        # ROC and F1 score
        # logits: [example_num, label_num(2)]
        # labels: [example_num, ]
        
        total_examples = len(self.data)

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
        pbar = tqdm(total=total_examples)
        for data in self.data:
            for sid, split in ([(1, 'train'), (2, 'dev'), (3, 'test')]):
                valid_positions = np.where(data['split_ids']==sid, True, False) # filter other splits
                if not valid_positions.any():
                    logging.info('-- Single: user {} has no data for {} evaluation'.format(ascii_decode(data['user_id']), split))
                    continue # no such split data
                
                valid_logits = data['logits'][valid_positions] # flat
                valid_labels = data['labels'][valid_positions] # flat
                
                pred_labels = np.argmax(valid_logits, axis=-1)
                positive_probs = F.softmax(torch.tensor(valid_logits), dim=-1)[:,1].numpy()            

                collections[split]['pred_labels'].extend(pred_labels)
                collections[split]['labels'].extend(valid_labels)
                collections[split]['positive_probs'].extend(positive_probs)
            pbar.update(1)
        pbar.close()
        
        logging.info('-- {} examples for evaluation; TRAIN: {} tokens; DEV: {} tokens; TEST: {} tokens.'.format(len(self.data), len(collections['train']['labels']), len(collections['dev']['labels']), len(collections['test']['labels'])))        
         
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


    def add(self, user_id, user_ability, logits, labels, split_ids, interaction_ids, memory_states, valid_length, valid_interactions, direction):
        memory_points = []
        cur_iid = -1
        for idx in range(len(interaction_ids)):
            if interaction_ids[idx] == cur_iid:
                memory_points.append(idx)
                cur_iid += 1

        assert len(memory_points) == valid_interactions[0] + 1


        self.data.append({
            'user_id': user_id,
            'user_ability': user_ability,
            'logits': logits[:valid_length[0]] if direction=='right' else logits[-valid_length[0]:],
            'labels': labels[:valid_length[0]] if direction=='right' else labels[-valid_length[0]:],
            'split_ids': split_ids[:valid_length[0]] if direction=='right' else split_ids[-valid_length[0]:],
            'interaction_ids': interaction_ids[:valid_length[0]] if direction=='right' else interaction_ids[-valid_length[0]:],
            'memory_states': memory_states[memory_points], # [interaction_num+1, num_words]
            'mastery_level': np.mean(memory_states, axis=-1) # [interaction_num+1, ]
        })
        


    def save_result(self, dirname):
        pbar = tqdm(total=len(self.data))         
        for data in self.data:
            filename = '-'.join([str(num) for num in data['user_id']])
            save_path = os.path.join(dirname, filename)
            np.savez(save_path, **data)
            pbar.update(1)
        pbar.close()
    


    def load(self, dirname):
        for filename in os.listdir(dirname):
            self.data.append(np.load(os.path.join(dirname, filename)))


    def learning_curve(self, user_selection, min_ratio=0.1, max_steps=1000):
        knowledge_growth = {}

        ## select target users (by abilities)
        user_abilities = [(ascii_decode(data['user_id']), data['user_ability']) for data in self.data]
        user_abilities.sort(key=lambda x:x[1])
        target_users = user_abilities[int(user_selection[0]*len(self.data)): int(user_selection[1]*len(self.data))]

        result = {}

        for step in range(max_steps):
            cur_step_ability = [] # students' abilities in this step
            for data in self.data:
                if ascii_decode(data['user_id']) not in target_users:
                    continue
                
                cur_step_ability.append(data['mastery_level'][step])
            
            if len(cur_step_ability) < len(self.data)*min_ratio:
                pass





def difficulty_calibration(word_file, generated_results, split=0.1, style='broken_line', fitting_degree=5):

    style_config = {
        'gpt-2': {'linestyle': '-', 'color': 'gray', 'marker': 'o'},
        'bart-base': {'linestyle': '-.', 'color': 'red', 'marker': '<'},
        't5-base': {'linestyle': ':', 'color': 'blue', 'marker': '*'}
    }

    for model_name in generated_results:
        evaluator = QGEvaluator(word_file=word_file)
        evaluator.read(generated_results[model_name])
        print(len(evaluator.generated), len(evaluator.reference), len(evaluator.difficulty_scores), len(evaluator.generated_difficulty_scores))
        evaluator.update_difficulty()

        generated_scores = np.array(evaluator.generated_difficulty_scores)
        reference_scores = np.array(evaluator.difficulty_scores)
        
        sorted_idx = np.argsort(generated_scores)

        generated_scores = generated_scores[sorted_idx]
        reference_scores = reference_scores[sorted_idx]


        # print('here', model_name)
        if style == 'fitting_line':
            # plt.plot(generated_scores, reference_scores)
            p= np.polyfit(generated_scores, reference_scores, fitting_degree)     ## fitting line
            p = np.poly1d(p)
            plt.plot(generated_scores, p(generated_scores), color='darkorange')
            plt.plot((0, 3), (0, 3), color='green', linestyle='--')

        elif style == 'broken_line':

            bucket_data = [[] for i in range(30)]
            for idx in range(len(generated_scores)):
                bucket = int(generated_scores[idx] // split)
                if bucket >= len(bucket_data):
                    continue
                bucket_data[bucket].append(idx)

            trunc_pos = len(bucket_data)
            for i in range(5, 30):
                if len(bucket_data[i]) < 10:
                    trunc_pos = i
                    break 

            bucket_data = bucket_data[:trunc_pos]
            # print(len(bucket_data))

            bucket_generated_score = [[generated_scores[idx] for idx in bucket] for bucket in bucket_data]
            bucket_reference_score = [[reference_scores[idx] for idx in bucket] for bucket in bucket_data]

            bucket_generated_score = [sum(scores)/len(scores) for scores in bucket_generated_score]
            bucket_reference_score = [sum(scores)/len(scores) for scores in bucket_reference_score]
            
            # print(bucket_generated_score, )
            # print(bucket_reference_score)
            # print('='*100)
            
            plt.plot(bucket_generated_score, bucket_reference_score, label=model_name, **style_config[model_name])

    plt.xlabel('Required difficulty')
    plt.ylabel('Generated difficulty')
    plt.plot((0, split*30), (0, split*30), color='green', linestyle='--')
    plt.grid()
    plt.legend()
    plt.show()