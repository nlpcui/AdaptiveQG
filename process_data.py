import sys, os, re, json, random
from torch.utils.data import Dataset
from dataclasses import dataclass, field
from typing import List, Dict
from tqdm import tqdm
import numpy as np


@dataclass
class SimpleToken:
    uid: int
    cnt: int

@dataclass
class Exercise:
    uid: int
    cnt: int

@dataclass
class Token:
    uid: str = ''
    text: str = ''
    pos_label: str = ''
    morpho_labels: dict = field(default_factory=dict)
    dep_label: str = ''
    dep_pos: int = ''
    label: int = -1

    def parse(self, line):
        line = re.sub(' +', ' ', line)
        fields = line.strip().split(' ')
        self.uid, self.text, self.pos_label = fields[:3]
        for item in fields[3].split('|'):
            key, value = item.split('=')
            self.morpho_labels[key] = value
        self.dep_label = fields[4]
        self.dep_pos = int(fields[5])
        
        if len(fields) == 7:
            self.label = int(fields[6])

            
@dataclass
class Record:
    user: str = ''
    countries: str = ''
    days: float = .0
    client: str = ''
    session: str = ''
    task: str = ''
    time: int = 0
    prompt: str = ''
    tokens: List[Token] = field(default_factory=list)
    sequence: str = ''
    
    def parse(self, line):
        line = re.sub(' +', ' ', line)
        fields = line.split(' ')
        for field_ in fields:
            name_, value = field_.split(':')
            if name_ == 'user':
                self.user = value
            elif name_ == 'countries':
                self.countries = value
            elif name_ == "session":
                self.session = value
            elif name_ == "days":
                self.days = float(value)
            elif name_ == "client":
                self.client = value
            elif name_ == "format":
                self.task = value
            elif name_ == "time":
                self.time = int(value) if type(value)==int else 0
            
    
class DuolingoDatasetBuilder(Dataset):
    def __init__(self, filepath, max_len, vocab_save_path):
        super(DuolingoDatasetBuilder, self).__init__()
        self.records = []

        # embedding_matrix: #formats#students#tokens#interactions#pad
        self.vocab = {
            '<listen>': (0, -1),
            '<reverse_translate>': (1, -1),
            '<reverse_tap>': (2, -1)
        } # listn/reverse_translation/reverse_tap
        self.exercise_map = {}
        
        self.x_exercises = [] 
        self.x_interactions = [] # e+a
        self.x_exercise_attn_masks = [] # mask other exercises
        self.x_interaction_attn_masks = [] # mask subsequent interactions  
        self.y_labels = []
        
        self.max_len = max_len
        
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

    def get_statistics(self, filters):
        # data_en_es_train: {'user_cnt': 2593, 'record_cnt': 824012, 'token_cnt': 2622957, 'record_per_user': 317.7832626301581, 'token_per_user': 1011.5530273814115, 'token_per_record': 3.1831538861084545, 'uniq_exercise_cnt': 7841, 'uniq_token_cnt': 2226, 'uniq_exercise_per_user': 257.8700347088315, 'uniq_token_per_user': 314.1519475510991}
        # 1:330788, 0:2292169
        user_records = {}
        uniq_exercises = {}
        uniq_tokens = {}
        
        for record in self.records:
            if record.user not in user_records:
                user_records[record.user] = []
            user_records[record.user].append(record)
            
            if record.sequence not in uniq_exercises:
                uniq_exercises[record.sequence] = 0
            uniq_exercises[record.sequence] += 1
            
            for token in record.tokens:
                if token.text not in uniq_tokens:
                    uniq_tokens[token.text] = 0
                uniq_tokens[token.text] += 1
            
        user_cnt = len(user_records)
        record_cnt = len(self.records)
        token_cnt = sum([len(record.tokens) for record in self.records])
        
        record_per_user = record_cnt / user_cnt
        token_per_user = token_cnt / user_cnt
        token_per_record = token_cnt / record_cnt
        
        uniq_exercise_cnt = len(uniq_exercises)
        uniq_token_cnt = len(uniq_tokens)
        
        uniq_exercise_per_user = 0
        uniq_token_per_user = 0
        for user in user_records:
            user_uniq_tokens = set()
            user_uniq_exercises = set()
            for record in user_records[user]:
                user_uniq_exercises.add(record.sequence)
                for token in record.tokens:
                    user_uniq_tokens.add(token.text)
                    
            uniq_exercise_per_user += len(user_uniq_exercises)
            uniq_token_per_user += len(user_uniq_tokens)
        
        uniq_exercise_per_user /= user_cnt
        uniq_token_per_user /= user_cnt
                    
        print({
            'user_cnt': user_cnt, 
            'record_cnt': record_cnt,
            'token_cnt': token_cnt, 
            'record_per_user': record_per_user, 
            'token_per_user': token_per_user,
            'token_per_record': token_per_record, 
            'uniq_exercise_cnt': uniq_exercise_cnt,
            'uniq_token_cnt': uniq_token_cnt,
            'uniq_exercise_per_user': uniq_exercise_per_user, 
            'uniq_token_per_user': uniq_token_per_user 
        })


    def save_by_stu(self, dirname):
        print('saving to {} ...'.format(dirname))
        for i in tqdm(range(len(self.x_exercises))):
            with open(os.path.join(dirname, '{}.json'.format(i)), 'w') as fp:
                example = json.dumps({
                    'x_exercise': self.x_exercises[i].tolist(),
                    'x_interaction': self.x_interactions[i].tolist(),
                    'x_exercise_attn_mask': self.x_exercise_attn_masks[i].tolist(),
                    'x_interaction_attn_mask': self.x_interaction_attn_masks[i].tolist(),
                    'y_labels': self.y_labels[i].tolist()
                })
                fp.write(example)
    

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



if __name__ == '__main__':
    pass

