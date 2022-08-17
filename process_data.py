import sys, os, re, json, random, torch, argparse, configparser, logging, csv, math
from torch.utils.data import Dataset
from typing import List, Dict
from tqdm import tqdm
import numpy as np
import pandas as pd
from transformers import BartTokenizer

        
class DuolingoDatasetBuilder(Dataset):
    def __init__(self, filepath, max_len, vocab_save_path):
        super(DuolingoDatasetBuilder, self).__init__()
        self.records = []

        # embedding_matrix: #formats#students#tokens#interactions#pad
        self.vocab = {
            '<listen>': (0, -1),
            '<reverse_translate>': (1, -1),
            '<reverse_tap>': (2, -1)
        } # three task formats: listn/reverse_translation/reverse_tap
        self.exercise_map = {}
        
        self.x_exercises = [] # words
        self.x_interactions = [] # <word, label> pairs
        self.x_exercise_attn_masks = [] # mask other exercises
        self.x_interaction_attn_masks = [] # mask subsequent interactions  
        self.y_labels = []
        
        self.max_len = max_len
        
        # statistics
        label_0 = 0
        label_1 = 0

        print('read data ...')
        with open(filepath, 'r') as fp:
            cur_record = Record()
            for line in tqdm(fp.readlines()):
                if line.startswith("# prompt:"): # prompt
                    cur_record.prompt = line.strip()[8:]
                elif line.startswith("# user:"):
                    cur_record.parse(line.strip()[2:])
                elif line.strip() == '':
                    self.records.append(cur_record)
                    cur_record.sequence = ' '.join([token.text for token in cur_record.tokens])
                    if cur_record.sequence not in self.exercise_map:
                        self.exercise_map[cur_record.sequence] = Exercise(len(self.exercise_map), 0)
                    self.exercise_map[cur_record.sequence].cnt += 1
                    cur_record = Record()
                else:
                    token = Token()
                    token.parse(line)
                    cur_record.tokens.append(token)
                    if  'w#{}'.format(token.text) not in self.vocab:
                        self.vocab['w#{}'.format(token.text)] = [len(self.vocab), 0] # id, cnt
                    self.vocab['w#{}'.format(token.text)][1] += 1
                    if 'i#{}|{}'.format(token.text, token.label) not in self.vocab:
                        self.vocab['i#{}|{}'.format(token.text, token.label)] = [len(self.vocab), 0]
                    self.vocab['i#{}|{}'.format(token.text, token.label)][1] += 1
                    if token.label == 0:
                        label_0 += 1
                    else:
                        label_1 += 1
        print('label cnt, 1: {}, 0:{}'.format(label_1, label_0))
        print(len(self.vocab))
        self.vocab['<pad>'] = [len(self.vocab), -1]
        self.vocab['<sep>'] = [len(self.vocab), -1]
        self.vocab['<unk>'] = [len(self.vocab), -1]
        self.save_vocab(vocab_save_path)

        user_records = {}
        for record in self.records:
            if record.user not in user_records:
                user_records[record.user] = []
            user_records[record.user].append(record)

        print('process data')
        for user in tqdm(user_records):
            x_exercise = []
            x_interaction = []
            x_exercise_attn_mask = [[True for i in range(self.max_len)] for j in range(self.max_len)]
            x_exercise_attn_mask[0][0] = False
            x_interaction_attn_mask = [[True for i in range(self.max_len)] for j in range(self.max_len)]
            x_interaction_attn_mask[0][0] = False
            y_labels = [] # 0, 1, -1 for padding

            x_exercise.append(self.vocab['<sep>'][0]) # <sep>
            x_interaction.append(self.vocab['<sep>'][0]) # <sep>
            y_labels.append(-1)

            for i, record in enumerate(user_records[user]):
                x_exercise.append(self.vocab['<{}>'.format(record.task)][0]) # task_format
                x_interaction.append(self.vocab['<{}>'.format(record.task)][0]) # task_format
                y_labels.append(-1)
                
                start = len(x_exercise)
                
                for j, token in enumerate(record.tokens):
                    x_exercise.append(self.vocab["w#{}".format(token.text)][0]) # words
                    x_interaction.append(self.vocab['i#{}|{}'.format(token.text, token.label)][0]) # interactions
                    y_labels.append(token.label)
                
                x_exercise.append(self.vocab['<sep>'][0])
                x_interaction.append(self.vocab['<sep>'][0])
                y_labels.append(token.label)
                
                end = min(len(x_exercise), self.max_len)
                if start <= self.max_len:
                    for s in range(start-1, end):
                        for t in range(start-1, end):
                            x_exercise_attn_mask[s][t] = False
                    
                    for s in range(start-1, end):
                        for t in range(0, start):
                            x_interaction_attn_mask[s][t] = False

            # avoid nan
            for i in range(self.max_len):
                if all(x_exercise_attn_mask[i]):
                    x_exercise_attn_mask[i][0] = False
                if all(x_interaction_attn_mask[i]):
                    x_interaction_attn_mask[i][0] = False
                assert not all(x_exercise_attn_mask[i])
                assert not all(x_interaction_attn_mask[i])

            assert(len(x_exercise)==len(x_interaction)==len(y_labels))
            self.x_exercises.append(self.pad(x_exercise, pad_token=self.vocab['<pad>'][0]))
            self.x_interactions.append(self.pad(x_interaction, pad_token=self.vocab['<pad>'][0]))
            self.x_exercise_attn_masks.append(x_exercise_attn_mask)
            self.x_interaction_attn_masks.append(x_interaction_attn_mask)
            self.y_labels.append(self.pad(y_labels, pad_token=-1))


        self.x_exercises = np.array(self.x_exercises)
        self.x_interactions = np.array(self.x_interactions)
        self.x_exercise_attn_masks = np.array(self.x_exercise_attn_masks)
        self.x_interaction_attn_masks = np.array(self.x_interaction_attn_masks)
        self.y_labels = np.array(self.y_labels)


    def word2id(self, token):
        if token in self.vocab:
            return self.vocab[token][0]
        else:
            return len(self.vocab[token])


    def pad(self, sequence, pad_token):
        if len(sequence) > self.max_len:
            return sequence[:self.max_len]
        else:
            return sequence + [pad_token for i in range(self.max_len-len(sequence))]

    def __len__(self):
        return len(self.x_exercises)

    def __getitem__(self, idx):
        return self.x_exercises[idx], self.x_interactions[idx], self.x_exercise_attn_masks[idx], self.x_interaction_attn_masks[idx], self.y_labels[idx]
    

    def save_vocab(self, save_path):
        total = 0
        word_cnt = 0
        interaction_cnt = 0
        for token in self.vocab:
            if token.startswith('i#'):
                interaction_cnt += 1
            elif token.startswith('w#'):
                word_cnt += 1
        print('{} words, {} interactions in total'.format(
            word_cnt,
            interaction_cnt
        ))

        with open(save_path, 'w') as fp:
            for token in self.vocab:
                fp.write('{}\t{}\t{}\n'.format(token, self.vocab[token][0], self.vocab[token][1])) 


def compute_difficulty(exercise_words, error_cnt):
    return error_cnt / math.sqrt(len(exercise_words))


class DuolingoKTDataset(Dataset):
    def __init__(self, data_file, vocab_file, word_file, exercise_file, max_len):
        self.max_len = max_len
        self.vocab = {}
        self.word_map = {}
        self.exercise_map = {}
        self.mode = 'train'

        df_vocab = pd.read_csv(vocab_file)
        for index, row in df_vocab.iterrows():
            self.vocab[row['token']] = row['token_id']
        
        self.word_map = {}
        df_word = pd.read_csv(word_file)
        for index, row in df_word.iterrows():
            self.word_map[row['word']] = (row['cnt'], row['wrong_cnt'], row['difficulty'])

        self.exercise_map = {}
        df_exercise = pd.read_csv(exercise_file)
        for index, row in df_exercise.iterrows():
            self.exercise_map[row['exercise']] = (row['cnt'], row['wrong_cnt'], row['difficulty'])

        self.user_ids = []
        self.interactions = []
        self.exercises = []
        self.split_labels = [] # 0train 1dev 2test
        self.answer_labels = []

        logging.info('vectorizing examples')
        with open(data_file, 'r') as fp:
            for line in tqdm(fp.readlines()):
                user_interactions = json.loads(line.strip())
                
                interaction_seq = [self.vocab['<sep>']]
                exercise_seq = [self.vocab['<sep>']]
                label_seq = [-1]
                split_seq = [0]

                for s, split in enumerate(['train', 'dev', 'test']):
                    for interaction_log in user_interactions[split]:

                        interaction_seq.append(self.vocab['<{}>'.format(interaction_log['format'])])
                        exercise_seq.append(self.vocab['<{}>'.format(interaction_log['format'])])
                        label_seq.append(-1)
                        split_seq.append(s)

                        for exercise_item in interaction_log['exercise']:
                            word = 'w#{}'.format(exercise_item['text'])
                            exercise_seq.append(self.vocab.get(word, self.vocab['<unk>']))

                            item = 'i#{}|{}'.format(exercise_item['text'], exercise_item['label'])
                            interaction_seq.append(self.vocab.get(item, self.vocab['<unk>']))

                            label_seq.append(exercise_item['label'])
                            split_seq.append(s)
                        
                        interaction_seq.append(self.vocab['<sep>'])
                        exercise_seq.append(self.vocab['<sep>'])
                        label_seq.append(-1)
                        split_seq.append(s)
                
                self.interactions.append(interaction_seq)
                self.exercises.append(exercise_seq)
                self.user_ids.append(user_interactions['user'])
                self.split_labels.append(split_seq)
                self.answer_labels.append(label_seq)


    def __len__(self):
        return len(self.user_ids)


    def __getitem__(self, idx):
        return self.user_ids[idx], self.interactions[idx], self.exercises[idx], self.answer_labels[idx], self.split_labels[idx]


    def get_statistics(self, split=None):
        # 0 train, 1 dev, 2 test, None all
        stats = {
            'user_cnt': len(self.user_ids),
            'interaction_cnt': 0,
            'word_cnt': 0,
            'interaction_per_user': 0,
            'word_per_user': 0,
            'word_per_interaction': 0,
            'uniq_exercise_cnt': 0,
            'uniq_word_cnt': 0,
            'uniq_exercise_per_user': 0,
            'uniq_word_per_user': 0,
            'x_max_length': float('-inf'),
            'x_min_length': float('inf'),
            'x_avg_length': 0,
            'x_length_distribution': [0 for i in range(50)]
        }
        
        uniq_words = set()
        uniq_exercises = set()

        for i in range(self.__len__()):
            user_uniq_words = set()
            user_uniq_exercises = set()
            exercise = []
            x_length = 0
            
            for j in range(1, len(self.exercises[i])):
                cur_split = self.split_labels[i][j]
                if split is not None and cur_split != split:
                    continue
                
                x_length += 1 
                if self.exercises[i][j] == self.vocab['<sep>']:
                    stats['interaction_cnt'] += 1
                    user_uniq_exercises.add('#'.join(exercise))
                    uniq_exercises.add('#'.join(exercise))
                    exercise = []
                elif self.exercises[i][j] > 5:
                    stats['word_cnt'] += 1
                    exercise.append(str(self.exercises[i][j]))
                    user_uniq_words.add(self.exercises[i][j])
                    uniq_words.add(self.exercises[i][j])

            stats['uniq_exercise_per_user'] += len(user_uniq_exercises)
            stats['uniq_word_per_user'] += len(user_uniq_words)

            if x_length > stats['x_max_length']:
                stats['x_max_length'] = x_length
            if x_length < stats['x_min_length']:
                stats['x_min_length'] = x_length
            
            stats['x_length_distribution'][min(x_length//100, len(stats['x_length_distribution'])-1)] += 1
            stats['x_avg_length'] += x_length
        
        stats['interaction_per_user'] = stats['interaction_cnt'] / stats['user_cnt']
        stats['word_per_user'] = stats['word_cnt'] / stats['user_cnt']
        stats['uniq_exercise_cnt'] = len(uniq_exercises) 
        stats['uniq_word_cnt'] = len(uniq_words)
        stats['uniq_exercise_per_user'] /= stats['user_cnt']
        stats['uniq_word_per_user'] /= stats['user_cnt']
        stats['word_per_interaction'] = stats['word_cnt'] / stats['interaction_cnt']
        stats['x_avg_length'] /= stats['user_cnt']

        return stats


    @classmethod
    def collate_fn(cls, batch_data):
        # [user_ids, interactions, exercises, labels, splits]
        # max_len, pad_token, sep_token, 
        max_len = max([len(b[1]) for b in batch_data])

        user_ids = [b[0] for b in batch_data]
        interactions = torch.tensor([cls.pad_trunc(b[1]) for b in batch_data])
        exercises = torch.tensor([cls.pad_trunc(b[2]) for b in batch_data])
        answer_labels = torch.tensor([cls.pad_trunc(b[3]) for b in batch_data])
        split_labels = torch.tensor([cls.pad_trunc(b[4]) for b in batch_data])

        return user_ids, interactions, exercises, interaction_attn_masks, exercise_attn_masks, answer_labels, split_labels


    @classmethod
    def pad_trunc(cls, seq):
        pass

    @classmethod
    def build_attn_mask(cls, seq):
        pass

    @classmethod
    def parse_lines(cls, data_file, user_interactions, split, vocab, word_map, exercise_map, label_dict=None):
        fp = open(data_file, 'r')
        
        update_cnt = 0

        interaction_log = InteractionLog()
        for line in tqdm(fp.readlines()):
            if line.startswith("# prompt:"): # prompt
                interaction_log.prompt = line.strip()[8:]
            elif line.startswith("# user:"):
                interaction_log.parse_from_line(line.strip()[2:])
            elif line.strip() == '':
                if interaction_log.user not in user_interactions:
                    assert split == 'train'
                    user_interactions[interaction_log.user] = {
                        'user': interaction_log.user,
                        'countries': interaction_log.countries,
                        'train': [],
                        'dev': [],
                        'test': []
                    }
                user_interactions[interaction_log.user][split].append(interaction_log)
                ## update interaction
                exercise_text = ' '.join([item.text for item in interaction_log.exercise])
                wrong_cnt = sum([item.label for item in interaction_log.exercise])
                if exercise_text not in exercise_map:
                    exercise_map[exercise_text] = [len(interaction_log.exercise), 0, 0, 0] # length, cnt, wrong_words_cnt, wrong_cnt
                exercise_map[exercise_text][1] += 1
                exercise_map[exercise_text][2] += wrong_cnt
                exercise_map[exercise_text][3] += 1 if wrong_cnt > 0 else 0

                update_cnt += 1
                interaction_log = InteractionLog()
            else:
                exercise_item = ExerciseItem()
                exercise_item.parse_from_line(line)
                if exercise_item.label == -1 and label_dict:
                    exercise_item.label = label_dict[exercise_item.item_id]
                interaction_log.exercise.append(exercise_item)
                
                # vocabulary
                if split == 'train':
                    ## update vocab
                    if  'w#{}'.format(exercise_item.text) not in vocab:
                        vocab['w#{}'.format(exercise_item.text)] = len(vocab)

                    if 'i#{}|{}'.format(exercise_item.text, exercise_item.label) not in vocab:
                        vocab['i#{}|{}'.format(exercise_item.text, exercise_item.label)] = len(vocab) # id, cnt

                    ## word
                    if exercise_item.text not in word_map:
                        word_map[exercise_item.text] = [0, 0] # cnt, wrong_cnt
                    word_map[exercise_item.text][0] += 1
                    word_map[exercise_item.text][1] += exercise_item.label  

        fp.close()

        return update_cnt

    @classmethod
    def get_format_datat(cls, train_raw, dev_raw, dev_key_raw, test_raw, test_key_raw, format_output, vocab_file, word_file, exercise_file):
        '''
        {
            'user': 'XEinXf5+',
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
        user_interactions = {}
        vocab = {
            '<sep>': 0,
            '<pad>': 1,
            '<unk>': 2,
            '<listen>': 3,
            '<reverse_translate>': 4,
            '<reverse_tap>': 5
        }
        word_map = {} # freq, difficulty(wrong cnt)
        exercise_map = {} # length, freq, difficulty(wrong cnt)

        # read train
        train_num = cls.parse_lines(train_raw, user_interactions, 'train', vocab, word_map, exercise_map, label_dict=None)
        logging.info('{} training interactions'.format(train_num))

        # read dev
        dev_keys = {}
        with open(dev_key_raw, 'r') as fp:
            for line in fp.readlines():
                item_id, label = line.strip().split(' ')
                dev_keys[item_id] = int(label)
        dev_num = cls.parse_lines(dev_raw, user_interactions, 'dev', vocab, word_map, exercise_map, label_dict=dev_keys)
        logging.info('{} dev interactions'.format(dev_num))

        # read test 
        test_keys = {}
        with open(test_key_raw, 'r') as fp:
            for line in fp.readlines():
                item_id, label = line.strip().split(' ')
                test_keys[item_id] = int(label)
        test_num = cls.parse_lines(test_raw, user_interactions, 'test', vocab, word_map, exercise_map, label_dict=test_keys)
        logging.info('{} test interactions'.format(test_num))
        
        with open(format_output, 'w') as fp:
            for user in user_interactions:
                output_line = json.dumps({
                    'user': user,
                    'countries': user_interactions[user]['countries'],
                    'train': [interaction.to_dict() for interaction in user_interactions[user]['train']],
                    'dev': [interaction.to_dict() for interaction in user_interactions[user]['dev']],
                    'test': [interaction.to_dict() for interaction in user_interactions[user]['test']]
                })
                fp.write(output_line+'\n')

        logging.info('saving vocab ...')
        vocab_table = [{
            'token': item[0],
            'token_id': item[1]
        } for item in vocab.items()]
        df = pd.DataFrame(vocab_table, columns=['token', 'token_id'])
        df.to_csv(vocab_file, index=False)
        
        logging.info('saving words ...')
        word_table = [{
            'word': item[0],
            'cnt': item[1][0],
            'wrong_cnt': item[1][1],
            'difficulty': item[1][1]/item[1][0]
        } for item in word_map.items()]
        df = pd.DataFrame(word_table, columns=['word', 'cnt', 'wrong_cnt', 'difficulty'])
        df.to_csv(word_file, index=False)
        
        logging.info('saving exercies ...')
        exercise_table = []
        for item in exercise_map.items():
            if item[1][1] >10:
                exercise_table.append({
            'exercise': item[0], 
            'word_num': item[1][0], 
            'cnt': item[1][1], 
            'word_wrong_cnt': item[1][2],
            'exercise_wrong_cnt': item[1][3], 
            'word_wrong_rate': item[1][2]/(item[1][0]*item[1][1]),
            'exercise_wrong_rate': item[1][3]/(item[1][1])
            })

        df = pd.DataFrame(exercise_table, columns=['exercise', 'word_num', 'cnt', 'word_wrong_cnt', 'exercise_wrong_cnt', 'word_wrong_rate', 'exercise_wrong_rate'])
        df.to_csv(exercise_file, index=False)


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
        self.exercise = []

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
        dump = self.__dict__
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



class DuolingoDataset(Dataset):
    def __init__(self, dirname, shuffle):
        super(DuolingoDataset, self).__init__()
        self.base_dir = dirname
        self.files = os.listdir(dirname)

        if shuffle:
            random.shuffle(self.files)
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        with open(os.path.join(self.base_dir, self.files[idx]), 'r') as fp:
            example_json = json.loads(fp.readlines()[0])
            return (
                np.array(example_json['x_exercise']), 
                np.array(example_json['x_interaction']), 
                np.array(example_json['x_exercise_attn_mask']),
                np.array(example_json['x_interaction_attn_mask']),
                np.array(example_json['y_labels'])
            )


class DuolingoTokenizer:
    def __init__(self, vocab_save_path):
        self.vocab = {}
        self.reverse_vocab = {}

        with open(vocab_save_path, 'r') as fp:
            for line in fp.readlines():
                fields = line.strip().split('\t')
                self.vocab[fields[0]] = [int(fields[1]), int(fields[2])]
                self.reverse_vocab[int(fields[1])] = [fields[0], int(fields[2])]

    def __len__(self):
        return len(self.vocab)

    def tokenize(self, words):
        encoded = []
        for word in words:
            encoded.append(self.vocab.get(word, '<unk>')[0])
        return encoded
    
    def detokenize(self, words):
        decoded = []
        for word in words:
            decoded.append(self.reverse_vocab.get(int(word))[0])
        return decoded
    
    def batch_tokenize(self, sequences):
        result = []
        for sequence in sequences:
            result.append(self.tokenize(sequence))
        return result

    def batch_detokenize(self, sequences):
        result = []
        for sequence in sequences:
            result.append(self.detokenize(sequence))
        return result


class DuolingoGenDataset(Dataset):
    def __init__(self, data_file, split, word_file, sample_rate):
        assert split in ['train', 'dev', 'test']
        self.sample_priority = [
            ['NOUN', 'VERB'],
            ['ADJ', 'ADV'],
            # ['PUNCT', 'SYM', 'X', 'ADP', 'AUX', 'INTJ', 'CCONJ', 'DET'. 'PROPN', 'NUM', 'PART', 'SCONJ', 'PRON']
        ]
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
        return len(self.x_keywords)

    def sample_by_pos(self, exercise, rate=0.5):
        # hierarchical sample
        if len(exercise) == 1:
            return None
        
        total_num = math.ceil(len(exercise)*rate)
        sampled = []

        # seperate words
        source_lists = [[] for i in range(len(self.sample_priority)+1)]
        for item in exercise:
            if item['text'] in ['am', 'is', 'are']:
                source_lists[-1].append(item['text'])
                continue
            flag = False
            for i, pos_list in enumerate(self.sample_priority):
                if item['pos'] in pos_list:
                    source_lists[i].append(item['text'])
                    flag = True
                    break
            if not flag:
                source_lists[-1].append(item['text'])

        sampled = set([])
        for source in source_lists:
            sampled.update(random.sample(source, min(max(total_num-len(sampled), 0), len(source))))
            if len(sampled) >= total_num:
                break
        
        sampled = list(sampled)
        random.shuffle(sampled)

        return sampled
    
    def sample_by_difficulty(self, exercise):
        pass

    
    def sample_random(self, exercise):
        pass

    def calc_word_coverage(self):
        # word coverage of sample strategy
        covered = set()
        for keywords in self.x_keywords:
            for word in keywords:
                covered.add(word)
        
        return len(covered) / len(self.vocab)  



class QGDataCollator:
    def __init__(self, model, tokenizer, x_max_length, y_max_length, padding='max_length', truncation='pt', label_pad_token_id=-100, return_tensors=True):
        self.model = model
        self.padding = padding
        self.x_max_length = x_max_length
        self.y_max_length = y_max_length
        self.label_pad_token_id = label_pad_token_id
        self.return_tensors = return_tensors
        self.truncation = truncation
        self.tokenizer = tokenizer

    def __call__(self, batch_data):

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
        filename=args.duolingo_en_es_train_log, 
        level=logging.INFO, 
        filemode='w'
    )  
