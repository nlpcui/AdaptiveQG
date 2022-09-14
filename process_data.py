import sys, os, re, json, random, collections
import copy, torch, argparse, configparser, logging, csv, math
from torch.utils.data import Dataset
from typing import List, Dict
from tqdm import tqdm
import numpy as np
import pandas as pd
from decimal import Decimal
from nltk import word_tokenize
from transformers import AutoModelForSeq2SeqLM, BartTokenizer, AutoConfig, AutoTokenizer
from pprint import pprint
from tqdm import tqdm
from scipy.sparse import coo_matrix
from utils import *
from pprint import pprint


class InteractionLog:
    def __init__(self):
        self.user = None
        self.countries = None
        self.prompt = None
        self.days = None
        self.client = None
        self.session = None
        self.format = None
        self.time = None
        self.exercise = [] # list of ExerciseItem

    def parse_from_line(self, line):
        line = re.sub(' +', ' ', line)
        fields = line.split(' ')
        for field_ in fields:
            name_, value = field_.split(':')
            if name_ == 'user':
                self.user = value
            elif name_ == 'countries':
                self.countries = value
            elif name_ == "days":
                self.days = float(value)
            elif name_ == "client":
                self.client = value
            elif name_ == "session":
                self.session = value
            elif name_ == "format":
                self.format = value
            elif name_ == "time":
                self.time = int(value) if type(value)==int else 0


    def to_dict(self):
        dump = copy.deepcopy(self.__dict__)
        dump['exercise'] = [item.__dict__ for item in self.exercise]
        return dump
        

class ExerciseItem:
    def __init__(self):
        self.item_id = None
        self.text = None
        self.pos = None
        self.morph = {}
        self.node = None
        self.edge = None
        self.label = -1

    def parse_from_line(self, line):
        line = re.sub(' +', ' ', line)
        fields = line.strip().split(' ')
        self.item_id, self.text, self.pos = fields[:3]
        for item in fields[3].split('|'):
            key, value = item.split('=')
            self.morph[key] = value
        self.node = fields[4]
        self.edge = int(fields[5])
        
        if len(fields) == 7:
            self.label = int(fields[6])
    
    def to_dict(self):
        return self.__dict__



def auto_convert(x):
    try:
        float_x = float(x)
    except Exception:
        float_x = None
    
    try:
        int_x = int(x)
    except Exception:
        int_x = None
    
    if int_x is not None and int_x == float_x:
        return int_x
    elif float_x is not None and int_x != float_x:
        return float_x
    else:
        return x
        


def build_dataset(train_raw, dev_raw, dev_key_raw, test_raw, test_key_raw, format_output, word_file, w_l_tuple_file, exercise_file, non_adaptive_gen_file, build_kt=True, build_gen_non_adaptive=True, un_cased=True):
    '''
    Input:
    train_raw: train_set (txt)
    dev_raw, dev_key_raw: dev_set (txt)
    test_raw, test_raw_key: test_set (txt)
    Output:
    word_file: word statistics
    exercise_file: exercise statistics
    '''
    if not build_kt and not build_gen:
        return
    
    user_interactions = {}
    word_map = {} 
    w_l_tuple_map = {} # w_l_tuple, id, freq
    exercise_map = {} # exercise, freq, correct_cnt, error_cnt, average_num_errors

    # read train
    logging.info('-- reading train data...')
    train_num = parse_lines(train_raw, user_interactions, 'train', word_map, w_l_tuple_map, exercise_map, label_dict=None, un_cased=un_cased)
    logging.info('-- {} training interactions'.format(train_num))

    # read dev
    logging.info('-- reading dev data...')
    dev_keys = {}
    with open(dev_key_raw, 'r') as fp:
        for line in fp.readlines():
            item_id, label = line.strip().split(' ')
            dev_keys[item_id] = int(label)
    dev_num = parse_lines(dev_raw, user_interactions, 'dev', word_map, w_l_tuple_map, exercise_map, label_dict=dev_keys, un_cased=un_cased)
    logging.info('--{} dev interactions'.format(dev_num))

    # read test 
    logging.info('-- reading test data')
    test_keys = {}
    with open(test_key_raw, 'r') as fp:
        for line in fp.readlines():
            item_id, label = line.strip().split(' ')
            test_keys[item_id] = int(label)
    test_num = parse_lines(test_raw, user_interactions, 'test', word_map, w_l_tuple_map, exercise_map, label_dict=test_keys, un_cased=un_cased)
    logging.info('-- {} test interactions'.format(test_num))
    
    logging.info('-- saving words to {}'.format(word_file))
    for word in word_map:
        word_map[word]['error_rate'] = word_map[word]['error_cnt'] / word_map[word]['cnt'] if word_map[word]['cnt'] > 0 else 0.
    df = pd.DataFrame(word_map.values(), columns=['word', 'word_id', 'cnt', 'error_cnt', 'error_rate'])
    df.to_csv(word_file, index=False)
    
    logging.info('-- saving w_l_tuples to {}'.format(w_l_tuple_file))
    df = pd.DataFrame(w_l_tuple_map.values(), columns=['w_l_tuple', 'w_l_tuple_id', 'cnt'])
    df.to_csv(w_l_tuple_file, index=False)

    logging.info('-- saving exercies to {}'.format(exercise_file))
    for exercise in exercise_map:
        exercise_map[exercise]['exercise_error_rate'] = exercise_map[exercise]['exercise_error_cnt'] / exercise_map[exercise]['cnt']
        exercise_map[exercise]['avg_word_error_cnt'] = exercise_map[exercise]['word_error_cnt'] / exercise_map[exercise]['cnt']
        exercise_map[exercise]['sum_word_error_rate'] = sum([word_map[word]['error_rate'] for word in exercise.split('#')])

    df = pd.DataFrame(exercise_map.values(), columns=['exercise', 'cnt', 'exercise_error_cnt', 'word_error_cnt', 'exercise_error_rate', 'avg_word_error_cnt', 'sum_word_error_rate'])
    df.to_csv(exercise_file, index=False)

    ## compute user_ability -- correct_rate*ave_difficulty
    for user in user_interactions:
        error_cnt = 0
        difficulty_score = 0

        total_cnt = 0
        
        for split in ['train', 'dev', 'test']:
            for interaction in user_interactions[user][split]:
                total_cnt += len(interaction.exercise)
                for item in interaction.exercise:
                    difficulty_score += word_map[item.text]['error_rate']
                    error_cnt += item.label
        
        difficulty_score /= total_cnt
        correct_rate = 1 - (error_cnt / total_cnt)

        user_interactions[user]['user_ability'] = (difficulty_score + correct_rate) / 2 # ave_word_difficulty * ave_word_correct_rate

    ## format data for knowledge tracing
    '''
    format_output: combine train/dev/test in a jsonl file like:
    {
        'user_id': 'XEinXf5+',
        'user_ability': 0.7
        'country': 'CO',
        'train': [{
                'prompt': 'Yo soy un niño.',
                'days': 0.003,,
                'client': 'web',
                'session': 'lesson',
                'format': 'reverse_tap',
                'time': 9,
                'exercise': [{
                    "id": '',
                    "text": '',
                    'tag': '',
                    'morpho': {},
                    "node": 'nsubj',
                    "edge": 4,
                    "label": 0
                }]
        }],
        'dev': [],
        'test': []
    }
    '''
    if build_kt:
        logging.info('-- saving format knowledge_tracing dataset to {}'.format(format_output))
        with open(format_output, 'w') as fp:
            for user in user_interactions:
                interaction = user_interactions[user]['train'][0]
                output_line = json.dumps({
                    'user_id': user,
                    'user_ability': user_interactions[user]['user_ability'],
                    'countries': user_interactions[user]['countries'],
                    'train': [interaction.to_dict() for interaction in user_interactions[user]['train']],
                    'dev': [interaction.to_dict() for interaction in user_interactions[user]['dev']],
                    'test': [interaction.to_dict() for interaction in user_interactions[user]['test']]
                })
                fp.write(output_line+'\n')

    ## construct exercises for non-adaptive generation 8:1:1
    if build_gen_non_adaptive:
        logging.info('-- saving non-adaptive generation dataset to {}.*.jsonl'.format(non_adaptive_gen_file))
        exercise_for_gen = {}
        for user in user_interactions:
            for split in ['train', 'dev', 'test']:
                for interaction in user_interactions[user][split]:
                    exercise = ' '.join([item.text for item in interaction.exercise])
                    if exercise not in exercise_for_gen:
                        exercise_for_gen[exercise] = {
                            'text': exercise, 
                            'tokens': [item.text for item in interaction.exercise],
                            'pos_tags': [item.pos for item in interaction.exercise],
                            'avg_word_error_cnt': exercise_map['#'.join([item.text for item in interaction.exercise])]['avg_word_error_cnt'],
                            'sum_word_error_rate': exercise_map['#'.join([item.text for item in interaction.exercise])]['sum_word_error_rate'],
                        }
        shuffled_keys = list(exercise_for_gen.keys())
        random.shuffle(shuffled_keys)
        splits =  [('train', (0, int(len(shuffled_keys)*0.8))), ('dev', (int(len(shuffled_keys)*0.8), int(len(shuffled_keys)*0.9))), ('test', (int(len(shuffled_keys)*0.9), len(shuffled_keys)))]

        for split, (start, end) in splits:
            with open('{}.{}.jsonl'.format(non_adaptive_gen_file, split), 'w') as fp:
                for idx in range(start, end):
                    fp.write(json.dumps(exercise_for_gen[shuffled_keys[idx]])+'\n')

    

def parse_lines(data_file, user_interactions, split, word_map, w_l_tuple_map, exercise_map, label_dict=None, un_cased=True):
    fp = open(data_file, 'r')
    
    update_cnt = 0

    interaction_log = InteractionLog()
    for line in tqdm(fp.readlines()):
        if line.startswith("# prompt:"): # prompt
            interaction_log.prompt = line.strip()[8:]
        elif line.startswith("# user:"): # meta information (country, time, format, ...)
            interaction_log.parse_from_line(line.strip()[2:])
        elif line.strip() == '': # end of an interaction
            if interaction_log.user not in user_interactions:
                assert split == 'train'
                user_interactions[interaction_log.user] = {
                    'user_id': interaction_log.user,
                    'countries': interaction_log.countries,
                    'train': [],
                    'dev': [],
                    'test': []
                }
            user_interactions[interaction_log.user][split].append(interaction_log)
            
            ## update exercise_map
            exercise_text = '#'.join([item.text for item in interaction_log.exercise])
            error_word_cnt = sum([item.label for item in interaction_log.exercise])
            if exercise_text not in exercise_map:
                exercise_map[exercise_text] = {'exercise': exercise_text, 'cnt': 0, 'exercise_error_cnt': 0, 'word_error_cnt': 0}
            exercise_map[exercise_text]['cnt'] += 1
            exercise_map[exercise_text]['exercise_error_cnt'] += 1 if error_word_cnt > 0 else 0
            exercise_map[exercise_text]['word_error_cnt'] += error_word_cnt

            update_cnt += 1
            interaction_log = InteractionLog()
        else: # exercise_item (word with correctness label)
            exercise_item = ExerciseItem()
            exercise_item.parse_from_line(line)
            
            if un_cased:
                exercise_item.text = exercise_item.text.lower()

            if exercise_item.label == -1 and label_dict:
                exercise_item.label = label_dict[exercise_item.item_id]
            interaction_log.exercise.append(exercise_item)

            ## update word_map
            if exercise_item.text not in word_map:
                word_map[exercise_item.text] = {'word': exercise_item.text, 'word_id': len(word_map), 'cnt': 0, 'error_cnt': 0} # cnt, wrong_cnt
            word_map[exercise_item.text]['cnt'] += 1
            word_map[exercise_item.text]['error_cnt'] += exercise_item.label

            ## update w_l_tuple_map
            w_l_tuple = '{}|{}'.format(exercise_item.text, exercise_item.label)
            if w_l_tuple not in w_l_tuple_map:
                w_l_tuple_map[w_l_tuple] = {'w_l_tuple': w_l_tuple, 'w_l_tuple_id': len(w_l_tuple_map), 'cnt': 0 } #
            w_l_tuple_map[w_l_tuple]['cnt'] += 1


    fp.close()

    return update_cnt


def get_statistics(data_file, target_split=None):

    def format_distribution(distr, total):
        distr = collections.OrderedDict(sorted(distr.items(), key=lambda x:x[0]))
        acc = 0
        for key in distr:
            distr[key] = (distr[key], acc + (distr[key] / total))
            acc = distr[key][1]
        
        return distr

    # TODO: update me
    # 0 train, 1 dev, 2 test, None all
    stats = {
        'user_cnt': 0,
        'interaction_cnt': 0,
        'word_cnt': 0,

        'interaction_num_distribution': {},          # interaction per user
        'max_interaction_num': float('-inf'),
        'min_interaction_num': float('inf'),
        'avg_interaction_num': .0,
        
        'interaction_length_distribution': {},       # wor per interaction 
        'interaction_max_length': float('-inf'),
        'interaction_min_length': float('inf'),
        
        'length_distribution': {},                   # word per user
        'max_length': float('-inf'),                  
        'min_length': float('inf'),
        'avg_length': .0
    }

    with open(data_file, 'r') as fp:
        for line in fp.readlines():
            stats['user_cnt'] += 1
            data = json.loads(line.strip())

            interaction_cnt = 0
            word_cnt = 0

            for split in ['train', 'dev', 'test']:
                if target_split and split != target_split:
                    continue
                interaction_cnt += len(data[split])
                
                for interaction in data[split]:
                    word_cnt += len(interaction['exercise'])

                    ## interaction_length_distribution distribution
                    bucket = len(interaction['exercise'])
                    if bucket not in stats['interaction_length_distribution']:
                        stats['interaction_length_distribution'][bucket] = 0
                    stats['interaction_length_distribution'][bucket] += 1

                    if len(interaction['exercise']) > stats['interaction_max_length']:
                        stats['interaction_max_length'] = len(interaction['exercise'])
                    if len(interaction['exercise']) < stats['interaction_min_length']:
                        stats['interaction_min_length'] = len(interaction['exercise'])
                    

            stats['interaction_cnt'] += interaction_cnt
            stats['word_cnt'] += word_cnt

            # concat length distribution 
            if word_cnt > stats['max_length']:
                stats['max_length'] = word_cnt
            if word_cnt < stats['min_length']:
                stats['min_length'] = word_cnt
            
            bucket = (word_cnt // 100) * 100
            if bucket not in stats['length_distribution']:
                stats['length_distribution'][bucket] = 0
            stats['length_distribution'][bucket] += 1

            # interaction_per_user distribution
            if interaction_cnt > stats['max_interaction_num']:
                stats['max_interaction_num'] = interaction_cnt
            if interaction_cnt < stats['min_interaction_num']:
                stats['min_interaction_num'] = interaction_cnt
            
            bucket = (interaction_cnt // 10)*10
            if bucket not in stats['interaction_num_distribution']:
                stats['interaction_num_distribution'][bucket] = 0
            stats['interaction_num_distribution'][bucket] += 1

                
            

    stats['length_distribution'] = format_distribution(stats['length_distribution'], stats['user_cnt'])
    stats['interaction_num_distribution'] = format_distribution(stats['interaction_num_distribution'], stats['user_cnt'])
    stats['interaction_length_distribution'] = format_distribution(stats['interaction_length_distribution'], stats['interaction_cnt'])
    pprint(stats)



class KTTokenizer:
    def __init__(self, word_file, w_l_tuple_file, max_seq_len, label_pad_id=-100, target_split=['train']):
        
        self.max_seq_len = max_seq_len
        self.label_pad_id = label_pad_id
        self.target_split = target_split

        self.word_map = {}
        df_words = pd.read_csv(word_file)
        for idx, row in df_words.iterrows():
            self.word_map[row['word']] = int(row['word_id'])
        
        self.w_l_tuple_map = {}
        df_w_l_tuples = pd.read_csv(w_l_tuple_file)
        for idx, row in df_w_l_tuples.iterrows():
            self.w_l_tuple_map[row['w_l_tuple']] = int(row['w_l_tuple_id'])
        
        self.task_map = {
            '<pad>': 0,
            'reverse_translate': 1,
            'reverse_tap': 2,
            'listen': 3
        }

        self.split_map = {
            '<pad>': 0,
            'train': 1,
            'dev': 2,
            'test': 3 
        }


        self.word_pad_id = self.word_map['<pad>']
        self.w_l_tuple_pad_id = self.w_l_tuple_map['<pad>']
        self.task_pad_id = self.task_map['<pad>']
        self.split_pad_id = self.split_map['<pad>']

        self.word_sep_id = self.word_map['<sep>']
        self.w_l_tuple_sep_id = self.w_l_tuple_map['<sep>']

        self.word_unk_id = self.word_map['<unk>']
        self.w_l_tuple_unk_id = self.w_l_tuple_map['<unk>']


        self.num_words = len(self.word_map)
        self.num_w_l_tuples = len(self.w_l_tuple_map)
        self.num_tasks = len(self.task_map)


class DuolingoKTDataset(Dataset):
    def __init__(self, raw_data_file, data_dir, word_file, w_l_tuple_file, max_seq_len, target_split, label_pad_id=-100, interaction_pad_id=-100, max_lines=-1):
        
        self.max_seq_len = max_seq_len
        self.label_pad_id = label_pad_id
        self.target_split = target_split
        self.interaction_pad_id = interaction_pad_id

        self.word_map = {
            '<pad>': 0,
            '<bos>': 1,
            '<eos>': 2,
            '<unk>': 3
        }
        special_word_num = len(self.word_map)
        df_words = pd.read_csv(word_file)
        for idx, row in df_words.iterrows():
            self.word_map[row['word']] = int(row['word_id']) + special_word_num
        
        self.w_l_tuple_map = {
            '<pad>': 0,
            '<bos>': 1,
            '<eos>': 2,
            '<unk>': 3
        }
        special_w_l_tuple_num = len(self.w_l_tuple_map)
        df_w_l_tuples = pd.read_csv(w_l_tuple_file)
        for idx, row in df_w_l_tuples.iterrows():
            self.w_l_tuple_map[row['w_l_tuple']] = int(row['w_l_tuple_id']) + special_w_l_tuple_num
        
        self.task_map = {
            '<pad>': 0,
            'reverse_translate': 1,
            'reverse_tap': 2,
            'listen': 3
        }

        self.split_map = {
            '<pad>': 0,
            'train': 1,
            'dev': 2,
            'test': 3 
        }


        self.num_words = len(self.word_map)
        self.num_w_l_tuples = len(self.w_l_tuple_map)
        self.num_tasks = len(self.task_map)

        if raw_data_file:
            self.build_dataset(raw_data_file, data_dir, max_lines)


        self.data = []
        logging.info('loading data from {}'.format(data_dir))
        line_cnt = 0
        for filename in tqdm(os.listdir(data_dir)):
            line_cnt += 1
            if line_cnt >= max_lines > 0:
                break
            data = np.load(os.path.join(data_dir, filename))
            self.data.append({key: data[key] for key in data})
            # print(sizeof(self.data[-1])/1024/1024)
        

    def build_dataset(self, data_file, dirname, max_lines=-1):
        logging.info('read data from {}, serialize data to {}...'.format(data_file, dirname))

        with open(data_file, 'r') as fp:
            pbar = tqdm(total=2593)
            line = fp.readline()
            idx = 0
            while line:
                if idx >= max_lines > 0:
                    break
                idx += 1
                user_log = json.loads(line.strip())
                format_data = self.format(user_log)
                filename = '{}.npz'.format('-'.join([str(num) for num in format_data['user_id']]))
                save_path = os.path.join(dirname, filename)
                np.savez(save_path, **format_data) 
                line = fp.readline()
                pbar.update(1)
            pbar.close() 


    def format(self, user_log, padding=False, truncation=True):
        # format SINGLE user_log to train data
        instance = {
            'user_id': ascii_encode(user_log['user_id']),
            'user_ability': [user_log['user_ability']],
            'word_ids': [],
            'word_attn_mask': None,
            'w_l_tuple_ids': [],
            'w_l_tuple_attn_mask': None,
            'position_ids': [],
            'task_ids': [],
            'interaction_ids': [],
            'labels': [],
            'split_ids': [],
        }

        instance['word_ids'].append(self.word_map['<bos>'])
        instance['w_l_tuple_ids'].append(self.w_l_tuple_map['<bos>'])
        instance['task_ids'].append(self.task_map['<pad>'])
        instance['interaction_ids'].append(-1)
        instance['labels'].append(self.label_pad_id)
        instance['split_ids'].append(self.split_map['<pad>'])

        cur_interaction_id = 0
        # words = []
        for split in ['train', 'dev', 'test']:
            if self.target_split and split not in self.target_split:
                continue 
            for interaction_id, interaction in enumerate(user_log[split]):
                for idx, item in enumerate(interaction['exercise']):
                    # words.append(item['text'])
                    instance['word_ids'].append(self.word_map.get(item['text'], self.word_map['<unk>']))
                    instance['w_l_tuple_ids'].append(self.w_l_tuple_map.get('{}|{}'.format(item['text'], item['label']), self.w_l_tuple_map['<unk>']))
                    instance['task_ids'].append(self.task_map[interaction['format']])
                    instance['labels'].append(item['label'])
                    instance['split_ids'].append(self.split_map[split])
                    instance['interaction_ids'].append(cur_interaction_id)
                
                cur_interaction_id += 1

        instance['word_ids'].append(self.word_map['<eos>'])
        instance['w_l_tuple_ids'].append(self.w_l_tuple_map['<eos>'])
        instance['task_ids'].append(self.task_map['<pad>'])
        instance['interaction_ids'].append(cur_interaction_id)
        instance['split_ids'].append(self.split_map['<pad>'])
        instance['labels'].append(self.label_pad_id)

        instance['position_ids'] = [i for i in range(len(instance['word_ids']))]
        
        instance['word_attn_mask'] = self.__build_word_attn_mask(instance)
        instance['w_l_tuple_attn_mask'] = self.__build_w_l_tuple_attn_mask(instance)
        instance['valid_length'] = [len(instance['word_ids'])]
        instance['valid_interactions'] = [instance['interaction_ids'][-1]]

        for key in instance:
            instance[key] = np.array(instance[key])

        if truncation and len(instance['word_ids']) > self.max_seq_len:
            self.truncate(instance)

        if padding and len(tokenized['word_ids']) < self.max_seq_len:
            self.pad(tokenized)

        return instance


    def __build_word_attn_mask(self, data):
        seq_len = len(data['word_ids'])
        word_attn_mask = [[True for i in range(seq_len)] for j in range(seq_len)]
        
        for target_idx in range(seq_len): # row
            for source_idx in range(seq_len): # column
                if data['interaction_ids'][target_idx] == data['interaction_ids'][source_idx]:
                    word_attn_mask[target_idx][source_idx] = False
                # elif data['interaction_ids'][target_idx] < data['interaction_ids'][source_idx]:
                #     break
        
        return word_attn_mask

    
    def __build_w_l_tuple_attn_mask(self, data):
        seq_len = len(data['word_ids'])
        w_l_tuple_attn_mask = [[True for i in range(seq_len)] for j in range(seq_len)]
        
        for target_idx in range(seq_len): # row
            if target_idx == 0:
                w_l_tuple_attn_mask[0][0] = False
            elif target_idx == seq_len -1:
                w_l_tuple_attn_mask[target_idx] = [False for i in range(seq_len)]
            else:
                for source_idx in range(seq_len): # column
                    if data['interaction_ids'][target_idx] > data['interaction_ids'][source_idx]:
                        w_l_tuple_attn_mask[target_idx][source_idx] = False # attend to previous interactions
                    elif data['interaction_ids'][target_idx] == data['interaction_ids'][source_idx] == self.interaction_pad_id:
                        w_l_tuple_attn_mask[target_idx][source_idx] == False # pad tokens attend to pad tokens

        return w_l_tuple_attn_mask


    def __build_memory_update_mask(self, data):
        memory_update_mask = [[0 for i in range(len(data['word_ids']))] for j in range(len(data['word_ids']))]
        for column_id in range(data['sep_indices']):
            if data['sep_indices'][column_id] == 0:
                continue
            
            memory_update_mask[column_id][column_id] == 1
            
            for row_id in range(column_id):
                if data['sep_indices'] == 1:
                    memory_update_mask[row_id][column_id] == 1
                    
        return memory_update_mask


    def __getitem__(self, idx):
        return self.data[idx] # user_id, user_ability, word_ids, word_attn_mask, w_l_tuple_ids, x_w_l_tuple_attn_mask, position_ids, task_ids, interaction_ids, labels, split_ids, valid_length, valid_interactions
        

    def pad(self, tokenized, max_seq_len=None, direction='left'):
        # only pad word_ids and w_l_tuple_ids, not to pad sep_indices
        max_seq_len = max_seq_len if max_seq_len else self.max_seq_len
        
        seq_pad_length = max_seq_len - len(tokenized['word_ids'])

        if seq_pad_length <= 0:
            return 
        
        pad_word_ids = [self.word_map['<pad>'] for i in range(seq_pad_length)]
        pad_w_l_tuple_ids = [self.w_l_tuple_map['<pad>'] for i in range(seq_pad_length)]
        pad_task_ids = [self.task_map['<pad>'] for i in range(seq_pad_length)]
        pad_labels = [self.label_pad_id for i in range(seq_pad_length)]
        pad_split_ids = [self.split_map['<pad>'] for i in range(seq_pad_length)]
        pad_interaction_ids = [self.interaction_pad_id for i in range(seq_pad_length)] # TODO: 是否-100 做填充？

        if direction == 'right':
            pad_width = (0, seq_pad_length)
            attn_pad_width = ((0, seq_pad_length), (0, seq_pad_length))
        elif direction == 'left':
            pad_width = (seq_pad_length, 0)
            attn_pad_width = ((seq_pad_length, 0), (seq_pad_length, 0))
        
        tokenized['word_ids'] = np.pad(tokenized['word_ids'], pad_width, constant_values=(self.word_map['<pad>'], self.word_map['<pad>']))
        tokenized['w_l_tuple_ids'] = np.pad(tokenized['w_l_tuple_ids'], pad_width, constant_values=(self.w_l_tuple_map['<pad>'], self.w_l_tuple_map['<pad>']))
        tokenized['task_ids'] = np.pad(tokenized['task_ids'], pad_width, constant_values=(self.task_map['<pad>'], self.task_map['<pad>']))
        tokenized['labels'] = np.pad(tokenized['labels'], pad_width, constant_values=(self.label_pad_id, self.label_pad_id))
        tokenized['split_ids'] = np.pad(tokenized['split_ids'], pad_width, constant_values=(self.split_map['<pad>'], self.split_map['<pad>']))
        tokenized['interaction_ids'] = np.pad(tokenized['split_ids'], pad_width, constant_values=(self.interaction_pad_id, self.interaction_pad_id))

        #TODO: pad mask
        tokenized['word_attn_mask'] = np.pad(tokenized['word_attn_mask'], attn_pad_width, constant_values=((False, False), (True, True)))
        tokenized['w_l_tuple_attn_mask'] = np.pad(tokenized['w_l_tuple_attn_mask'], attn_pad_width, constant_values=((False, False), (True, True)))
    
        tokenized['position_ids'] = [i for i in range(len(tokenized['position_ids']))]
        

    def truncate(self, tokenized, max_seq_len=None, direction='left'):
        max_seq_len = max_seq_len if max_seq_len else self.max_seq_len
        
        trunc_length = len(tokenized['word_ids']) - max_seq_len
        if trunc_length <= 0:
            return

        if direction == 'right':
            tokenized['word_ids'] = np.concatenate([tokenized['word_ids'][:max_seq_len-1], np.array([self.word_map['<eos>']])]) 
            tokenized['w_l_tuple_ids'] = np.concatenate([tokenized['w_l_tuple_ids'][:max_seq_len-1], np.array([self.w_l_tuple_map['<eos>']])]) 
            tokenized['task_ids'] = np.concatenate([tokenized['task_ids'][:max_seq_len-1], np.array([self.task_map['<pad>']])]) 
            tokenized['labels'] = np.concatenate([tokenized['labels'][:max_seq_len-1], np.array([self.label_pad_id])])
            tokenized['split_ids'] = np.concatenate([tokenized['split_ids'][:max_seq_len-1], np.array([self.split_map['<pad>']])])
            tokenized['interaction_ids'] = np.concatenate([tokenized['interaction_ids'][:max_seq_len-1], np.array([tokenized['interaction_ids'][-1]+1])]) 
            tokenized['position_ids'] = tokenized['position_ids'][:max_seq_len]

            # truncate attn mask
            tokenized['word_attn_mask'] = tokenized['word_attn_mask'][:seq_len,:seq_len]
            tokenized['word_attn_mask'][-1, :] = True
            tokenized['word_attn_mask'][:, -1] = True
            tokenized['word_attn_mask'][-1, -1] = False

            tokenized['w_l_tuple_attn_mask'] = tokenized['w_l_tuple_attn_mask'][:seq_len,:seq_len]
            tokenized['w_l_tuple_attn_mask'][:, -1] = True
            tokenized['w_l_tuple_attn_mask'][-1, :] = False
            ## end mask truncation

        elif direction == 'left':
            tokenized['word_ids'] = np.concatenate([np.array([self.word_map['<bos>']]), tokenized['word_ids'][trunc_length+1:]])
            tokenized['w_l_tuple_ids'] = np.concatenate([np.array([self.w_l_tuple_map['<bos>']]), tokenized['w_l_tuple_ids'][trunc_length+1:]])
            tokenized['task_ids'] = np.concatenate([np.array([self.task_map['<pad>']]), tokenized['task_ids'][trunc_length+1:]])
            tokenized['labels'] = np.concatenate([np.array([self.label_pad_id]), tokenized['labels'][trunc_length+1:]]) 
            tokenized['split_ids'] = np.concatenate([np.array([self.split_map['<pad>']]), tokenized['split_ids'][trunc_length+1:]])
            tokenized['position_ids'] = [i for i in range(max_seq_len)]
            
            start_interaction = tokenized['interaction_ids'][trunc_length+1]
            tokenized['interaction_ids'] = np.concatenate([np.array([-1]), tokenized['interaction_ids'][trunc_length+1:]-start_interaction])

            # truncate attn mask
            tokenized['word_attn_mask'] = tokenized['word_attn_mask'][trunc_length:,trunc_length:]
            tokenized['word_attn_mask'][:, 0] = True
            tokenized['word_attn_mask'][0, :] = True
            tokenized['word_attn_mask'][0, 0] = False

            tokenized['w_l_tuple_attn_mask'] = tokenized['w_l_tuple_attn_mask'][trunc_length:, trunc_length:]
            tokenized['w_l_tuple_attn_mask'][:, 0] = True
            tokenized['w_l_tuple_attn_mask'][0, :] = True
            tokenized['w_l_tuple_attn_mask'][0, 0] = False 
            # end mask truncation




    def __len__(self):
        return len(self.data)


    def construct_collate_fn(self, direction='left', max_seq_len=None):
        
        def collate_fn(batch_data):
            batch_max_seq_len = max([len(data['word_ids']) for data in batch_data])

            # logging.info('-- collate, {} examples, batch_max_seq_len {}'.format(len(batch_data), batch_max_seq_len))
            if max_seq_len and max_seq_len < batch_max_seq_len:
                batch_max_seq_len = max_seq_len

            # align batch length
            for data in batch_data:
                # TODO: align/truncate attn_mask
                self.truncate(data, max_seq_len=batch_max_seq_len, direction=direction)
                self.pad(data, max_seq_len=batch_max_seq_len, direction=direction)
            
            x_user_ids = torch.tensor(np.stack([data['user_id'] for data in batch_data], axis=0))
            x_user_abilities = torch.tensor(np.stack([data['user_ability'] for data in batch_data], axis=0))
            x_word_ids = torch.tensor(np.stack([data['word_ids'] for data in batch_data], axis=0))
            x_word_attn_masks = torch.tensor(np.stack([data['word_attn_mask'] for data in batch_data], axis=0))
            x_w_l_tuple_ids = torch.tensor(np.stack([data['w_l_tuple_ids'] for data in batch_data], axis=0))
            x_w_l_tuple_attn_masks = torch.tensor(np.stack([data['w_l_tuple_attn_mask'] for data in batch_data], axis=0))
            x_position_ids = torch.tensor(np.stack([data['position_ids'] for data in batch_data], axis=0))
            x_task_ids = torch.tensor(np.stack([data['task_ids'] for data in batch_data], axis=0))
            x_interaction_ids = torch.tensor(np.stack([data['interaction_ids'] for data in batch_data], axis=0))
            y_labels = torch.tensor(np.stack([data['labels'] for data in batch_data], axis=0))
            split_ids = torch.tensor(np.stack([data['split_ids'] for data in batch_data], axis=0)) # 0 train, 1 dev, 2 test
            x_valid_lengths = torch.tensor(np.stack([data['valid_length'] for data in batch_data], axis=0))
            x_valid_interactions = torch.tensor(np.stack([data['valid_interactions'] for data in batch_data], axis=0))

            return x_user_ids, x_user_abilities, x_word_ids, x_word_attn_masks, x_w_l_tuple_ids, x_w_l_tuple_attn_masks, x_position_ids, x_task_ids, x_interaction_ids, y_labels, split_ids, x_valid_lengths, x_valid_interactions


        return collate_fn


class DuolingoPersonalziedQGDataset(Dataset):
    def __init__(self, data_file, split, word_file, sample_rate):
        '''
        construct three inputs: 
        '''
        assert split in ['train', 'dev', 'test']

        self.vocab = {}

        self.user_ids = []
        self.difficulty = []
        self.x_keywords = []
        self.y_exercises = []

        df = pd.read_csv(word_file)
        for index, row in df.iterrows():
            self.vocab[row['word']] = len(self.vocab)
            
        with open(data_file, 'r') as fp:
            for line in fp.readlines():
                user_log = json.loads(line.strip())
                for interaction_log in user_log[split]:
                    keywords = self.sample_by_pos(interaction_log['exercise'])
                    exercise_words = [item['text'] for item in interaction_log['exercise']]
                    error_cnt = sum([item['label'] for item in interaction_log['exercise']])
                    if keywords:
                        self.user_ids.append(user_log['user'])
                        self.difficulty.append(error_cnt/math.sqrt(len(interaction_log['exercise'])))
                        self.x_keywords.append(' '.join(keywords))
                        self.y_exercises.append(' '.join(exercise_words))
                        

    def __getitem__(self, index):
        return self.user_ids[index], self.difficulty[index], self.x_keywords[index], self.y_exercises[index]

    def __len__(self):
        return len(self.data)


    def calc_word_coverage(self):
        # word coverage of sample strategy
        covered = set()
        for keywords in self.x_keywords:
            for word in keywords:
                covered.add(word)
        
        return len(covered) / len(self.vocab)  


    def construct_collate_fn(self, model, tokenizer, x_max_length, y_max_length, padding='max_length', truncation=True, label_pad_token_id=-100, return_tensors='pt'):

        def collate_fn(batch_data):
            user_ids = [data[0] for data in batch_data]
            difficulties = [data[1] for data in batch_data]
            keywords_encoded = self.tokenizer(
                [data[2] for data in batch_data], 
                max_length=self.x_max_length, 
                padding=self.padding,
                truncation=self.truncation,
                return_tensors=self.return_tensors
            )
            x_keyword_ids = keywords_encoded['input_ids']
            x_attention_mask = keywords_encoded['attention_mask']
            y_exercise_labels = self.tokenizer(
                [data[3] for data in batch_data],
                max_length=self.y_max_length,
                padding=self.padding,
                truncation=self.truncation,
                return_tensors=self.return_tensors
            )['input_ids']

            y_exercise_labels[y_exercise_labels==self.tokenizer.pad_token_id] = self.label_pad_token_id
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(y_exercise_labels)

            return user_ids, difficulties, x_keyword_ids, x_attention_mask, y_exercise_labels, decoder_input_ids 

        return collate_fn



class DuolingoNonAdaptiveGenDataset(Dataset):
    def __init__(self, data_file, tokenizer, model, sampler, enable_difficulty):
        '''
        non-adaptive QG, two inputs: prompt words/difficulty score
        '''
        
        self.enable_difficulty = enable_difficulty
        self.sampler = sampler
        
        self.x_difficulty_scores = []
        self.x_difficulty_levels = []
        self.x_prompt_words = []
        self.y_exercises = []

        self.exercise_tokens = []
        self.token_pos_tags = []

        self.difficulty_distributions = {}

        with open(data_file, 'r') as fp:
            for line in fp.readlines():
                data = json.loads(line.strip())
                
                # prompt words
                sampled_words = self.sampler.sample(tokens=data['tokens'], pos_tags=data['pos_tags'])
                if not sampled_words:
                    continue # discard one-word sentence
                self.x_prompt_words.append(' '.join(sampled_words))

                # difficulty score
                self.x_difficulty_scores.append(data['sum_word_error_rate'])
                
                # discretized difficulty level
                difficulty_level = self.convert_difficulty(data['sum_word_error_rate'])
                self.x_difficulty_levels.append(difficulty_level)

                # question text
                if self.enable_difficulty:
                    difficulty_control_token = tokenizer.additional_special_tokens[difficulty_level]
                    self.y_exercises.append('{} {}'.format(difficulty_control_token, data['text']))
                else:
                    self.y_exercises.append(data['text'])
                


        # logging.info(sorted(self.difficulty_distributions.items(), key=lambda x:x[0]))
        ## difficulty level distributions. (//0.5)
        ## train: [(0, 1660), (1, 3663), (2, 1489), (3, 279), (4, 44), (5, 12), (6, 8), (7, 2)]
        ## dev:
        ## test: 

    def __getitem__(self, idx):
        return self.x_difficulty_scores[idx], self.x_difficulty_levels[idx], self.x_prompt_words[idx],  self.y_exercises[idx]

    
    def construct_collate_fn(self, tokenizer, model, x_max_length, y_max_length, padding='max_length', truncation=True, return_tensors='pt', label_pad_token_id=-100):
        
        def collate_fn(batch_data):    

            x_difficulty_scores = torch.tensor([data[0] for data in batch_data])
            x_difficulty_levels = torch.tensor([data[1] for data in batch_data])
            
            x_encoded = tokenizer([data[2] for data in batch_data], max_length=x_max_length, padding=padding, truncation=truncation, return_tensors=return_tensors)

            x_prompt_word_ids = x_encoded['input_ids']
            x_attention_mask = x_encoded['attention_mask']
            y_exercise_labels = tokenizer(
                [data[3] for data in batch_data],
                max_length=y_max_length,
                padding=padding,
                truncation=truncation,
                return_tensors=return_tensors
            )['input_ids']

            y_exercise_labels[y_exercise_labels==tokenizer.pad_token_id] = label_pad_token_id
            y_decoder_input_ids = model.prepare_decoder_input_ids_from_labels(y_exercise_labels)

            return x_difficulty_scores, x_difficulty_levels, x_prompt_word_ids, x_attention_mask, y_exercise_labels, y_decoder_input_ids 

        
        return collate_fn


    def __len__(self):
        return len(self.x_prompt_words)


    def convert_difficulty(self, raw_score):
        level = int(raw_score // 0.5)
        if level not in self.difficulty_distributions:
            self.difficulty_distributions[level] = 0
        self.difficulty_distributions[level] += 1
        
        if level <= 2:
            return level
        else:
            return 3



class WordSampler:
    def __init__(self, sample_rate):
        # sample prompt words from a sentence
        self.sample_priority = [
            ['NOUN', 'VERB'],
            ['ADJ', 'ADV'],
            ['PUNCT', 'SYM', 'X', 'ADP', 'AUX', 'INTJ', 'CCONJ', 'DET', 'PROPN', 'NUM', 'PART', 'SCONJ', 'PRON']
        ]

        self.sample_rate = sample_rate

    
    def sample(self, tokens, pos_tags):
        if len(tokens) == 1:
            return None
        
        assert len(tokens) == len(pos_tags)
        total_num = int(Decimal(len(tokens)*self.sample_rate).quantize(Decimal("1."), rounding = "ROUND_HALF_UP"))
        
        # seperate words
        source_lists = [[] for i in range(len(self.sample_priority))]
        for idx in range(len(tokens)):
            
            if tokens[idx] in ['am', 'is', 'are']:
                source_lists[-1].append(tokens[idx])
                continue
            for i, pos_list in enumerate(self.sample_priority):
                if pos_tags[idx] in pos_list:
                    source_lists[i].append(tokens[idx])

        sampled = set([])
        for source in source_lists:
            sampled.update(random.sample(source, min(max(total_num-len(sampled), 0), len(source))))
            if len(sampled) >= total_num:
                break
        
        sampled = list(sampled)
        random.shuffle(sampled)

        return sampled



def compute_difficulties(reference_file, sentence):
    # split sentence 
    words = nltk.word_tokenize(sentence.lower())
    for word in words:
        pass
    pass



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='local_conf.ini')
    
    args, remaining_argv = parser.parse_known_args()
    config = configparser.ConfigParser()
    config.read(args.conf)

    for section in config:
        if section == 'DEFAULT':
            continue
        for option in config.options(section):
            parser.add_argument('--{}'.format(option), default=config.get(section, option))
    
    args = parser.parse_args(remaining_argv)
    
    logging.basicConfig(
        format='%(asctime)s %(message)s', 
        datefmt='%Y-%d-%m %I:%M:%S %p', 
        # filename=args.duolingo_en_es_train_log, 
        level=logging.INFO, 
        filemode='w'
    )  

    # stats = get_statistics('/Users/cuipeng/Documents/Datasets/duolingo_2018_shared_task/data_en_es/en_es_format.jsonl', target_split=None)
    
    # build_dataset(
    #     train_raw=args.duolingo_en_es_train_raw, 
    #     dev_raw=args.duolingo_en_es_dev_raw, 
    #     dev_key_raw=args.duolingo_en_es_dev_key_raw, 
    #     test_raw=args.duolingo_en_es_test_raw, 
    #     test_key_raw=args.duolingo_en_es_test_key_raw, 
    #     format_output=args.duolingo_en_es_format, 
    #     w_l_tuple_file=args.duolingo_en_es_w_l_tuple_file, 
    #     word_file=args.duolingo_en_es_word_file, 
    #     exercise_file=args.duolingo_en_es_exercise_file, 
    #     non_adaptive_gen_file=args.duolingo_en_es_non_adaptive_exercise_gen,
    #     build_kt=True,
    #     build_gen_non_adaptive=False
    # )
