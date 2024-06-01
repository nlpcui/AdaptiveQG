import sys, os, re, json, random, collections, nltk
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


def get_user_map(user_file):
    user_map = {}
    user_df = pd.read_csv(user_file)
    for _, row in user_df.iterrows():
        user_map[row['user_name']] = int(row['user_id'])
    return user_map


def get_word_map(word_file):
    vocab = {'<pad>': 0, '<bos>': 1, '<eos>': 2, '<unk>': 3}
    special_word_num = len(vocab)
    df = pd.read_csv(word_file)
    for idx, row in df.iterrows():
        vocab[row['word']] = int(row['word_id']) + special_word_num
    return vocab


def get_student_by_rank(student_file):
    students = []
    for idx, row in pd.read_csv(student_file).iterrows():
        students.append(row['student_id'])
    return students


def get_kt_word_difficulty(word_file):
    word_difficulty = {}
    for idx, row in pd.read_csv(word_file).iterrows():
        word_difficulty[row['word']] = row['difficulty']
    return word_difficulty


def get_vocab_difficulty(word_file):
    vocab_difficulty = {'<pad>': 0., '<bos>': 0., '<eos>': 0., '<unk>': 0.}
    df = pd.read_csv(word_file)
    for idx, row in df.iterrows():
        vocab_difficulty[row['word']] = row['error_rate']
    return vocab_difficulty


def get_w_l_tuple_map(w_l_tuple_file):
    w_l_tuple_map = {'<pad>': 0, '<bos>': 1, '<eos>': 2, '<unk>': 3}
    special_w_l_tuple_num = len(w_l_tuple_map)
    df_w_l_tuples = pd.read_csv(w_l_tuple_file)
    for idx, row in df_w_l_tuples.iterrows():
        w_l_tuple_map[row['w_l_tuple']] = int(row['w_l_tuple_id']) + special_w_l_tuple_num
    return w_l_tuple_map


def get_pos_tag_map(pos_tag_file):
    pos_tag_map = {'<pad>': 0}
    df_pos_tag = pd.read_csv(pos_tag_file)
    for _, row in df_pos_tag.iterrows():
        pos_tag_map[row['pos_tag']] = len(pos_tag_map)

    return pos_tag_map


def get_exercise(exercise_file):
    exercises = []
    for _, row in pd.read_csv(exercise_file).iterrows():
        exercises.append(row['exercise'])
    return exercises


def get_sampled_users(user_file):
    with open(user_file, 'r') as fp:
        return json.loads(fp.readlines()[0])


def get_train_exercise(train_exercise_file):
    train_exercises = set()
    with open(train_exercise_file, 'r') as fp:
        for line in fp.readlines():
            example = json.loads(line.strip())
            train_exercises.add(example['text'])
    return train_exercises


def build_reverse_index(exercise_file, save_path):
    index = {}
    for _, row in pd.read_csv(exercise_file).iterrows():
        words = row['exercise'].split('#')
        for word in words:
            if word not in index:
                index[word] = []
            index[word].append(row['exercise'].replace('#', ' '))
    with open(save_path, 'w') as fp:
        fp.write(json.dumps(index))


def read_inverse_index(index_file):
    with open(index_file, 'r') as fp:
        index = json.loads(fp.readlines()[0])

    for key in index:
        index[key] = set(index[key])
    return index

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
        self.exercise = []  # list of ExerciseItem

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
                self.time = int(value) if is_int(value) else 0

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


def build_dataset(train_raw, dev_raw, dev_key_raw, test_raw, test_key_raw, format_output, word_file, w_l_tuple_file, exercise_file, user_file, non_adaptive_gen_file, build_kt=True,
                  build_gen_non_adaptive=True, un_cased=True):
    '''
    Input:
    train_raw: train_set (txt)
    dev_raw, dev_key_raw: dev_set (txt)
    test_raw, test_raw_key: test_set (txt)
    Output:
    word_file: word statistics
    exercise_file: exercise statistics
    '''
    if not build_kt and not build_gen_non_adaptive:
        return

    user_interactions = {}
    word_map = {}
    w_l_tuple_map = {}  # w_l_tuple, id, freq
    exercise_map = {}  # exercise, freq, correct_cnt, error_cnt, average_num_errors
    user_map = {}

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

    # save user_map
    logging.info('-- saving users to {}'.format(user_file))
    user_map = {user_name: {'user_name': user_name, 'user_id': i} for (i, user_name) in enumerate(user_interactions.keys())}
    df = pd.DataFrame(user_map.values(), columns=['user_name', 'user_id'])
    df.to_csv(user_file, index=False)

    # compute user_ability: correct_rate * ave_difficulty
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

        user_interactions[user]['user_ability'] = (difficulty_score * correct_rate) / 2  # ave_word_difficulty * ave_word_correct_rate

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
                fp.write(output_line + '\n')

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
        splits = [('train', (0, int(len(shuffled_keys) * 0.8))), ('dev', (int(len(shuffled_keys) * 0.8), int(len(shuffled_keys) * 0.9))), ('test', (int(len(shuffled_keys) * 0.9), len(shuffled_keys)))]

        for split, (start, end) in splits:
            with open('{}.{}.jsonl'.format(non_adaptive_gen_file, split), 'w') as fp:
                for idx in range(start, end):
                    fp.write(json.dumps(exercise_for_gen[shuffled_keys[idx]]) + '\n')


def parse_lines(data_file, user_interactions, split, word_map, w_l_tuple_map, exercise_map, label_dict=None, un_cased=True):
    fp = open(data_file, 'r')

    update_cnt = 0

    interaction_log = InteractionLog()
    for line in tqdm(fp.readlines()):
        if line.startswith("# prompt:"):  # prompt
            interaction_log.prompt = line.strip()[8:]
        elif line.startswith("# user:"):  # meta information (country, time, format, ...)
            interaction_log.parse_from_line(line.strip()[2:])
        elif line.strip() == '':  # end of an interaction
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
        else:  # exercise_item (word with correctness label)
            exercise_item = ExerciseItem()
            exercise_item.parse_from_line(line)

            if un_cased:
                exercise_item.text = exercise_item.text.lower()

            if exercise_item.label == -1 and label_dict:
                exercise_item.label = label_dict[exercise_item.item_id]
            interaction_log.exercise.append(exercise_item)

            ## update word_map
            if exercise_item.text not in word_map:
                word_map[exercise_item.text] = {'word': exercise_item.text, 'word_id': len(word_map), 'cnt': 0, 'error_cnt': 0}  # cnt, wrong_cnt
            word_map[exercise_item.text]['cnt'] += 1
            word_map[exercise_item.text]['error_cnt'] += exercise_item.label

            ## update w_l_tuple_map
            w_l_tuple = '{}|{}'.format(exercise_item.text, exercise_item.label)
            if w_l_tuple not in w_l_tuple_map:
                w_l_tuple_map[w_l_tuple] = {'w_l_tuple': w_l_tuple, 'w_l_tuple_id': len(w_l_tuple_map), 'cnt': 0}  #
            w_l_tuple_map[w_l_tuple]['cnt'] += 1

    fp.close()

    return update_cnt


def get_statistics(data_file, target_split=None):
    def format_distribution(distr, total):
        distr = collections.OrderedDict(sorted(distr.items(), key=lambda x: x[0]))
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

        'interaction_num_distribution': {},  # interaction per user
        'max_interaction_num': float('-inf'),
        'min_interaction_num': float('inf'),
        'avg_interaction_num': .0,

        'interaction_length_distribution': {},  # wor per interaction
        'interaction_max_length': float('-inf'),
        'interaction_min_length': float('inf'),

        'length_distribution': {},  # word per user
        'max_length': float('-inf'),
        'min_length': float('inf'),
        'avg_length': .0,

        'days_distribution': {},
        'time_distribution': {},
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
                    if interaction['days'] not in stats['days_distribution']:
                        stats['days_distribution'][interaction['days']] = 0
                    stats['days_distribution'][interaction['days']] += 1

                    if interaction['time'] not in stats['time_distribution']:
                        stats['time_distribution'][interaction['time']] = 0
                    stats['time_distribution'][interaction['time']] += 1

                    word_cnt += len(interaction['exercise'])

                    # interaction_length_distribution distribution
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

            bucket = (interaction_cnt // 10) * 10
            if bucket not in stats['interaction_num_distribution']:
                stats['interaction_num_distribution'][bucket] = 0
            stats['interaction_num_distribution'][bucket] += 1

    stats['length_distribution'] = format_distribution(stats['length_distribution'], stats['user_cnt'])
    stats['interaction_num_distribution'] = format_distribution(stats['interaction_num_distribution'], stats['user_cnt'])
    stats['interaction_length_distribution'] = format_distribution(stats['interaction_length_distribution'], stats['interaction_cnt'])
    stats['days_distribution'] = format_distribution(stats['days_distribution'], stats['interaction_cnt'])
    stats['time_distribution'] = format_distribution(stats['time_distribution'], stats['interaction_cnt'])
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


class AdaptiveQGDataset(Dataset):
    def __init__(self, data_file):
        super(AdaptiveQGDataset, self).__init__()

        self.x_word_ids = []
        self.x_tuple_ids = []

        with open(data_file, 'r') as fp:
            for line in fp.readlines():
                pass

    def __getitem__(self, item):
        return

    def __len__(self):
        return len(self)

class NonAdaptiveGenDataset(Dataset):
    def __init__(self, data_file, tokenizer, model_name, sample_rate, idx, prepare_decoder_input_ids, use_difficulty=True, use_skill=True, x_max_length=15, y_max_length=30):
        if not use_difficulty and not use_skill:
            logging.error('Dataset build error')
            exit(1)

        # consider add example questions
        self.use_difficulty = use_difficulty
        self.use_skill = use_skill

        self.x_max_length = x_max_length
        self.y_max_length = y_max_length
        # self.sampler = sampler
        # self.x_difficulty_levels = []

        self.x_target_word_ids = []
        self.x_input_ids = []
        self.x_attention_mask = []
        self.x_difficulties = []
        # self.x_difficulty_positions = []
        self.y_decoder_input_ids = []
        self.y_labels = []

        self.tokenizer = tokenizer
        if not self.tokenizer.pad_token:  # for gpt2
            self.tokenizer.pad_token = self.tokenizer.eos_token

        with open(data_file, 'r') as fp:
            for line in fp.readlines():
                instance = json.loads(line)

                x_input = None
                if self.use_difficulty and self.use_skill:  # <pad> + target_words (skills)
                    x_input = tokenizer.pad_token + ' '.join(instance['keywords'][str(sample_rate)][idx])
                elif self.use_difficulty:  # <pad>
                    x_input = tokenizer.pad_token
                elif self.use_skill:  # target_words
                    x_input = ' '.join(instance['keywords'][str(sample_rate)][idx])
                y_output = instance['text']

                x_tokenized = self.tokenizer(x_input, padding='max_length', return_tensors='pt', max_length=self.x_max_length)
                self.x_target_word_ids.append(
                    self.tokenizer(' '.join(instance['keywords'][str(sample_rate)][idx]), padding='max_length', return_tensors='pt', max_length=self.x_max_length)['input_ids'])
                self.x_input_ids.append(x_tokenized['input_ids'])
                self.x_attention_mask.append(x_tokenized['attention_mask'])
                self.x_difficulties.append(torch.tensor([[instance['sum_word_error_rate']]]))

                y_tokenized = self.tokenizer(y_output, padding='max_length', return_tensors='pt', max_length=self.y_max_length)
                y_tokenized['input_ids'][y_tokenized['input_ids'] == self.tokenizer.pad_token_id] = -100
                self.y_labels.append(y_tokenized['input_ids'])

                if prepare_decoder_input_ids:
                    self.y_decoder_input_ids.append(prepare_decoder_input_ids(y_tokenized['input_ids']))

        self.x_input_ids = torch.cat(self.x_input_ids, dim=0)
        self.x_target_word_ids = torch.cat(self.x_target_word_ids, dim=0)
        self.x_attention_mask = torch.cat(self.x_attention_mask, dim=0)
        self.x_difficulties = torch.cat(self.x_difficulties, dim=0)
        self.y_decoder_input_ids = torch.cat(self.y_decoder_input_ids, dim=0)
        self.y_labels = torch.cat(self.y_labels, dim=0)

    def __getitem__(self, idx):
        return {
            'x_target_word_ids': self.x_target_word_ids[idx],
            'x_difficulties': self.x_difficulties[idx],
            'x_input_ids': self.x_input_ids[idx],
            'x_attention_mask': self.x_attention_mask[idx],
            'y_decoder_input_ids': self.y_decoder_input_ids[idx],
            'y_labels': self.y_labels[idx]
        }

    def __len__(self):
        return self.x_input_ids.size(0)


class NonIndividualizedQGDataset(Dataset):
    def __init__(self, data_file, tokenizer, sample_rate, idx, difficulty_bucket, difficulty_max_label, d_template, d_source,
                 prepare_decoder_input_ids, d_type, use_difficulty=True, use_skill=True, x_max_length=15, y_max_length=30):
        if not use_difficulty and not use_skill:
            logging.error('Dataset build error')
            exit(1)
        assert d_type in ['continuous', 'discrete']
        assert d_source in ['kt', 'gd']
        self.use_difficulty = use_difficulty
        self.use_skill = use_skill

        self.d_type = d_type
        self.d_source = d_source

        self.x_max_length = x_max_length
        self.y_max_length = y_max_length
        # self.sampler = sampler
        # self.x_difficulty_levels = []

        self.x_target_word_ids = []
        self.x_input_ids = []
        self.x_attention_mask = []
        self.x_difficulties = []

        self.y_decoder_input_ids = []
        self.y_labels = []

        self.difficult_bucket = difficulty_bucket
        self.difficulty_max_label = difficulty_max_label
        self.d_template = d_template

        self.tokenizer = tokenizer

        with open(data_file, 'r') as fp:
            for line in fp.readlines():
                instance = json.loads(line)
                # print('here', self.d_source, 'kt:', instance['kt_estimated_difficulty'], 'gd:', instance['sum_word_error_rate'])
                if self.d_source == 'kt':
                    avg_difficulty = instance['kt_estimated_difficulty']
                else:
                    avg_difficulty = instance['sum_word_error_rate']

                discrete_difficulty = int(min(avg_difficulty // self.difficult_bucket, self.difficulty_max_label-1))
                x_input = ''

                if self.use_difficulty:  # use difficulty only C_d
                    if self.d_type == 'continuous':
                        x_input += self.tokenizer.pad_token  # continuous d will be replaced by d_embedding
                    else:
                        x_input += self.d_template.format(discrete_difficulty)

                if self.use_skill:  # use skills only C_s
                    x_input += ' '.join(instance['keywords'][str(sample_rate)][idx])

                y_output = instance['text']

                x_tokenized = self.tokenizer(x_input, padding='max_length', return_tensors='pt', max_length=self.x_max_length)
                self.x_target_word_ids.append(
                    self.tokenizer(' '.join(instance['keywords'][str(sample_rate)][idx]), padding='max_length', return_tensors='pt', max_length=self.x_max_length)['input_ids'])
                self.x_input_ids.append(x_tokenized['input_ids'])
                self.x_attention_mask.append(x_tokenized['attention_mask'])
                self.x_difficulties.append(torch.tensor([[instance['sum_word_error_rate']]]))

                y_tokenized = self.tokenizer(y_output, padding='max_length', return_tensors='pt', max_length=self.y_max_length)
                y_tokenized['input_ids'][y_tokenized['input_ids'] == self.tokenizer.pad_token_id] = -100
                self.y_labels.append(y_tokenized['input_ids'])

                if prepare_decoder_input_ids:
                    self.y_decoder_input_ids.append(prepare_decoder_input_ids(y_tokenized['input_ids']))

        self.x_input_ids = torch.cat(self.x_input_ids, dim=0)
        self.x_target_word_ids = torch.cat(self.x_target_word_ids, dim=0)
        self.x_attention_mask = torch.cat(self.x_attention_mask, dim=0)
        self.x_difficulties = torch.cat(self.x_difficulties, dim=0)
        self.y_decoder_input_ids = torch.cat(self.y_decoder_input_ids, dim=0)
        self.y_labels = torch.cat(self.y_labels, dim=0)

    def __getitem__(self, idx):
        return {
            'x_target_word_ids': self.x_target_word_ids[idx],
            'x_difficulties': self.x_difficulties[idx],
            'x_input_ids': self.x_input_ids[idx],
            'x_attention_mask': self.x_attention_mask[idx],
            'y_decoder_input_ids': self.y_decoder_input_ids[idx],
            'y_labels': self.y_labels[idx]
        }

    def __len__(self):
        return self.x_input_ids.size(0)


class JointKTQGDataset(Dataset):
    # pair KT sequence with single questions. Only works with batch_size=1 (student-wise batch)
    def __init__(self, data_dir, word_map, pos_tag_map, target_split, word_sampler, qg_tokenizer_left, qg_tokenizer_right, prepare_decoder_input_ids_from_labels, sample_rate,
                 difficulty_bucket, difficulty_max_label, d_template, d_type, d_source, qg_use_history=False, qg_use_difficulty=False, qg_use_state=False, qg_use_skills=False,
                 return_kt=False, return_aqg=False, pad_label_id=-100, kt_trunc_direction='left', kt_pad_direction='right', max_seq_length=1024, max_keyword_length=15,
                 max_question_length=30, max_examples=-1, qg_max_history_length=256, target_example=None, user_list=None):
        super(JointKTQGDataset, self).__init__()

        assert d_type in ['discrete', 'continuous']
        assert d_source in ['kt', 'gd']

        if not return_kt and not return_aqg:
            logging.error('must set return data')
            exit(1)

        if not qg_use_history and not qg_use_difficulty and not qg_use_state and not qg_use_skills:
            logging.error('at least one type input')
            exit(1)

        self.user_list = user_list

        self.qg_use_difficulty = qg_use_difficulty
        self.qg_use_skills = qg_use_skills
        self.qg_use_state = qg_use_state
        self.qg_use_history = qg_use_history

        self.difficulty_bucket = difficulty_bucket
        self.difficulty_max_label = difficulty_max_label
        self.d_template = d_template
        self.d_type = d_type
        self.d_source = d_source

        self.return_kt = return_kt
        self.return_aqg = return_aqg
        self.target_split = target_split
        self.qg_max_history_length = qg_max_history_length
        self.max_seq_length = max_seq_length
        self.max_keyword_length = max_keyword_length
        self.max_question_length = max_question_length
        self.pad_label_id = pad_label_id
        self.sample_rate = sample_rate
        self.kt_pad_direction = kt_pad_direction
        self.kt_trunc_direction = kt_trunc_direction
        self.vocab = {'<pad>': 0, '<bos>': 1, '<eos>': 2, '<unk>': 3}
        self.w_l_tuple_map = {'<pad>': 0, '<bos>': 1, '<eos>': 2, '<unk>': 3}
        self.vocab_difficulty = {'<pad>': 0., '<bos>': 0., '<eos>': 0., '<unk>': 0.}
        self.pos_tag_map = {'<pad>': 0}
        self.word_sampler = word_sampler
        self.qg_tokenizer_left = qg_tokenizer_left  # truncate from left
        self.qg_tokenizer_right = qg_tokenizer_right  # truncate from right
        self.prepare_decoder_input_ids_from_labels = prepare_decoder_input_ids_from_labels

        self.word_map = word_map
        self.inverse_word_map = {self.word_map[word]: word for word in self.word_map}

        self.pos_tag_map = pos_tag_map
        self.inverse_pos_tag_map = {self.pos_tag_map[pos_tag]: pos_tag for pos_tag in self.pos_tag_map}

        self.x_user_ascii = []
        self.x_user_ids = []
        self.x_adaptive_difficulties = []  # [num_questions, ]
        self.x_non_adaptive_difficulties = []  # [num_questions, ]
        self.x_kt_valid_length = []
        self.x_kt_valid_interactions = []
        self.x_kt_word_ids = []  # [max_seq_length, ]
        self.x_kt_word_pos_tag_ids = []  # [max_seq_length, ]
        self.x_kt_w_l_tuple_ids = []  # [max_seq_length, ]
        self.x_kt_interaction_ids = []  # [max_seq_length, ]
        self.x_kt_split_ids = []  # [max_seq_length, ]
        self.y_kt_labels = []  # [max_seq_length, ]

        self.x_qg_state_positions = []  # [num_questions, ]
        self.x_qg_split_ids = []  # [num_questions, ]  # num_questions varies
        self.x_qg_knowledge_states_ids = []  # [num_questions, ]
        self.x_qg_input_ids = []  # [num_questions, max_keywords_length]
        self.x_qg_keyword_ids = []  # [num_questions, max_keywords_length]
        self.x_qg_attention_masks = []  # [num_questions, max_keywords_length]
        self.y_qg_decoder_input_ids = []  # [num_questions, max_question_length]
        self.y_qg_labels = []  # [num_questions, max_question_length]

        if d_type == 'discrete':
            difficulty_control_tokens = {'additional_special_tokens': [self.d_template.format(i) for i in range(self.difficulty_max_label)]}
            self.qg_tokenizer_right.add_special_tokens(difficulty_control_tokens)
            self.qg_tokenizer_left.add_special_tokens(difficulty_control_tokens)

        num_examples = 0
        for filename in tqdm(os.listdir(data_dir)):
            num_examples += 1
            if target_example is not None and num_examples != target_example:
                continue
            if num_examples > max_examples > 0:
                break
            with np.load(os.path.join(data_dir, filename)) as user_log:
                if user_log['user_id'] == 2537:
                    continue  # this user has no dev data thus will cause bug.

                if self.user_list and ascii_decode(user_log['user_ascii']) not in self.user_list:
                    continue

                if len(user_log['word_ids']) > 2500:
                    continue  # no enough training data

                if len(user_log['word_ids']) > self.max_seq_length:
                    user_log = self.truncate(user_log)
                # check truncation
                try:
                    assert len(user_log['word_ids']) == len(user_log['w_l_tuple_ids']) == len(user_log['word_pos_tag_ids']) == \
                           len(user_log['interaction_ids']) == len(user_log['labels']) == len(user_log['split_ids']) <= self.max_seq_length
                except AssertionError:
                    logging.error('Truncation Error: word_ids {}, w_l_tuple_ids {}, word_pos_tag_ids {}, interaction_ids {}, labels {}, split_ids {}'.format(
                        len(user_log['word_ids']), len(user_log['w_l_tuple_ids']), len(user_log['word_pos_tag_ids']),
                        len(user_log['interaction_ids']), len(user_log['labels']), len(user_log['split_ids'])
                    ))
                    exit(1)

                # check padding
                if len(user_log['word_ids']) < self.max_seq_length:
                    user_log = self.pad(user_log)
                try:
                    assert len(user_log['word_ids']) == self.max_seq_length
                    assert len(user_log['w_l_tuple_ids']) == self.max_seq_length
                    assert len(user_log['word_pos_tag_ids']) == self.max_seq_length
                    assert len(user_log['interaction_ids']) == self.max_seq_length
                    assert len(user_log['labels']) == self.max_seq_length
                    assert len(user_log['split_ids']) == self.max_seq_length
                except AssertionError:
                    logging.error('Padding Error: word_ids {}, w_l_tuple_ids {}, word_pos_tag_ids {}, interaction_ids {}, labels {}, split_ids {}'.format(
                        len(user_log['word_ids']), len(user_log['w_l_tuple_ids']), len(user_log['word_pos_tag_ids']),
                        len(user_log['interaction_ids']), len(user_log['labels']), len(user_log['split_ids'])
                    ))
                    exit(1)

                self.x_user_ascii.append(user_log['user_ascii'])
                self.x_user_ids.append(user_log['user_id'])
                self.x_adaptive_difficulties.append(torch.tensor(user_log['adaptive_difficulties']).float())
                self.x_non_adaptive_difficulties.append(torch.tensor(user_log['non_adaptive_difficulties']).float())
                self.x_kt_word_ids.append(user_log['word_ids'])
                self.x_kt_word_pos_tag_ids.append(user_log['word_pos_tag_ids'])
                self.x_kt_w_l_tuple_ids.append(user_log['w_l_tuple_ids'])
                self.x_kt_split_ids.append(user_log['split_ids'])
                self.x_kt_interaction_ids.append(user_log['interaction_ids'])
                self.x_kt_valid_length.append(np.sum(np.where(user_log['word_ids'] != 0, 1, 0)))
                self.x_kt_valid_interactions.append(np.max(user_log['interaction_ids'])+1)
                self.y_kt_labels.append(user_log['labels'])

                # create QG data
                # split sequence to questions/tags(for sampling)
                all_questions = []
                all_pos_tags = []
                end_positions = []  # end point of each question, used to get knowledge_states for question generation
                last_id = 0

                question_word_ids = []  # temp container
                question_pos_tag_ids = []  # temp container
                question_split_ids = []  # temp container

                for i, interaction_id in enumerate(user_log['interaction_ids']):
                    # print(i, interaction_id)
                    if interaction_id == -1:
                        continue  # skip <bos> and <eos>
                    # the first knowledge_state is from the pad token before first word, i.e., <bos>
                    if len(end_positions) == 0:
                        end_positions.append(i - 1)
                    if interaction_id != last_id:
                        all_questions.append(copy.deepcopy(question_word_ids))
                        all_pos_tags.append(copy.deepcopy(question_pos_tag_ids))
                        question_split_ids.append(user_log['split_ids'][i - 1])
                        question_word_ids.clear()
                        question_pos_tag_ids.clear()
                        end_positions.append(i - 1)
                        last_id = interaction_id
                    question_word_ids.append(user_log['word_ids'][i])
                    question_pos_tag_ids.append(user_log['word_pos_tag_ids'][i])

                if len(question_word_ids) > 0:
                    all_questions.append(copy.deepcopy(question_word_ids))
                    all_pos_tags.append(copy.deepcopy(question_pos_tag_ids))
                    question_split_ids.append(user_log['split_ids'][-1])

                history = []  # history questions
                for i in range(len(all_questions)):
                    if i == 0:
                        history.append([])  # no history context
                    else:
                        # a hacky way to insert "." as separator
                        history.append(history[-1]+all_questions[i-1]+[len(self.word_map)-1])  # TODO: separator between historical questions?

                assert len(history) == len(all_questions)

                qg_input_ids = []
                qg_keyword_ids = []
                qg_attention_masks = []
                qg_decoder_input_ids = []
                qg_labels = []
                for i in range(len(all_questions)):
                    history_words = [self.inverse_word_map[word_id] for word_id in history[i]]
                    question_words = [self.inverse_word_map[word_id] for word_id in all_questions[i]]
                    question_word_pos_tags = [self.inverse_pos_tag_map[pos_tag_id] for pos_tag_id in all_pos_tags[i]]
                    sampled_keywords = self.word_sampler.sample(question_words, question_word_pos_tags, sample_rate=self.sample_rate)

                    # construct question input ids
                    non_adaptive_difficulty_label = int(min(user_log['non_adaptive_difficulties'][i]//self.difficulty_bucket, self.difficulty_max_label-1))
                    qg_input = ''

                    if self.qg_use_history and i != 0:
                        qg_input += ' '.join(history_words)
                        if self.qg_use_state or self.qg_use_difficulty or self.qg_use_skills:
                            qg_input += self.qg_tokenizer_left.bos_token
                    if self.qg_use_state:
                        qg_input += self.qg_tokenizer_right.pad_token
                    if self.qg_use_difficulty:
                        if self.d_type == 'discrete':
                            qg_input += self.d_template.format(non_adaptive_difficulty_label)   # use non-adaptive difficulty for Controllable baseline
                        else:
                            qg_input += self.qg_tokenizer_right.pad_token
                    if self.qg_use_skills:
                        qg_input += ' ' + ' '.join(sampled_keywords)

                    # print(i, 'raw', qg_input)

                    target_words_tokenized = self.qg_tokenizer_right(' '.join(sampled_keywords), return_tensors='pt', max_length=self.max_keyword_length, padding='max_length', truncation=True)

                    if self.qg_use_history:
                        input_tokenized = self.qg_tokenizer_left(qg_input, return_tensors='pt', max_length=self.qg_max_history_length, padding='max_length', truncation=True)
                    else:
                        input_tokenized = self.qg_tokenizer_right(qg_input, return_tensors='pt', max_length=self.max_keyword_length, padding='max_length', truncation=True)
                    question_tokenized = self.qg_tokenizer_right(' '.join(question_words), return_tensors='pt', max_length=self.max_question_length, padding='max_length', truncation=True)

                    # print('recovered', self.qg_tokenizer_right.decode(input_tokenized['input_ids'][0]))

                    qg_input_ids.append(input_tokenized['input_ids'])
                    qg_attention_masks.append(input_tokenized['attention_mask'])

                    qg_keyword_ids.append(target_words_tokenized['input_ids'])

                    question_tokenized['input_ids'][question_tokenized['input_ids'] == self.qg_tokenizer_right.pad_token_id] = self.pad_label_id
                    qg_labels.append(question_tokenized['input_ids'])
                    qg_decoder_input_ids.append(self.prepare_decoder_input_ids_from_labels(question_tokenized['input_ids']))

                try:
                    assert len(end_positions) == len(question_split_ids) == len(user_log['adaptive_difficulties']) \
                           == len(user_log['non_adaptive_difficulties']) == len(qg_keyword_ids) == len(qg_attention_masks) \
                           == len(qg_labels) == len(qg_decoder_input_ids) == len(qg_input_ids)
                except AssertionError:
                    logging.error('Question length not aligned: end_positions {}, question_split_ids {}, adaptive_difficulties {}, non_adaptive_difficulties {}, qg_keyword_ids {}, qg_attention_masks {}, '.format(
                        len(end_positions), len(question_split_ids), len(user_log['adaptive_difficulties']), len(user_log['non_adaptive_difficulties']),
                        len(qg_keyword_ids), len(qg_attention_masks), len(qg_labels), len(qg_decoder_input_ids), len(qg_input_ids)
                    ))
                    exit(1)

                # print('non_adaptive_difficulties', user_log['non_adaptive_difficulties'])
                # exit(1)

                self.x_qg_state_positions.append(torch.tensor(end_positions))
                self.x_qg_split_ids.append(torch.tensor(question_split_ids))
                self.x_qg_knowledge_states_ids.append(torch.tensor(end_positions))
                self.x_qg_keyword_ids.append(torch.cat(qg_keyword_ids, dim=0))
                self.x_qg_input_ids.append(torch.cat(qg_input_ids, dim=0))
                self.x_qg_attention_masks.append(torch.cat(qg_attention_masks, dim=0))
                self.y_qg_labels.append(torch.cat(qg_labels, dim=0))
                self.y_qg_decoder_input_ids.append(torch.cat(qg_decoder_input_ids, dim=0))

        self.x_user_ascii = torch.tensor(np.array(self.x_user_ascii))
        self.x_user_ids = torch.tensor(np.array(self.x_user_ids))
        self.x_kt_word_ids = torch.tensor(np.array(self.x_kt_word_ids))
        self.x_kt_word_pos_tag_ids = torch.tensor(np.array(self.x_kt_word_pos_tag_ids))
        self.x_kt_w_l_tuple_ids = torch.tensor(np.array(self.x_kt_w_l_tuple_ids))
        self.x_kt_interaction_ids = torch.tensor(np.array(self.x_kt_interaction_ids))
        self.x_kt_split_ids = torch.tensor(np.array(self.x_kt_split_ids))
        self.y_kt_labels = torch.tensor(np.array(self.y_kt_labels))

    def __len__(self):
        return len(self.x_user_ascii)

    def __getitem__(self, idx):
        kt_item = {
            'x_user_ascii': self.x_user_ascii[idx],
            'x_user_ids': self.x_user_ids[idx],
            'x_kt_word_ids': self.x_kt_word_ids[idx],
            'x_kt_word_pos_tag_ids': self.x_kt_word_pos_tag_ids[idx],
            'x_kt_w_l_tuple_ids': self.x_kt_w_l_tuple_ids[idx],
            'x_kt_interaction_ids': self.x_kt_interaction_ids[idx],
            'x_kt_split_ids': self.x_kt_split_ids[idx],
            'y_kt_labels': self.y_kt_labels[idx],
            'x_kt_valid_length': self.x_kt_valid_length[idx],
            'x_kt_valid_interactions': self.x_kt_valid_interactions[idx]
        }
        pqg_item = {
            # 'x_user_ascii': self.x_user_ascii[idx],
            # 'x_user_ids': self.x_user_ids[idx],
            'x_adaptive_difficulties': self.x_adaptive_difficulties[idx],
            'x_non_adaptive_difficulties': self.x_non_adaptive_difficulties[idx],
            'x_qg_state_positions': self.x_qg_state_positions[idx],
            'x_qg_split_ids': self.x_qg_split_ids[idx],
            'x_qg_input_ids': self.x_qg_input_ids[idx],
            'x_qg_keyword_ids': self.x_qg_keyword_ids[idx],
            'x_qg_attention_masks': self.x_qg_attention_masks[idx],
            'y_qg_decoder_input_ids': self.y_qg_decoder_input_ids[idx],
            'y_qg_labels': self.y_qg_labels[idx]
        }

        if self.return_kt and self.return_aqg:
            return kt_item | pqg_item
        elif self.return_kt:
            return kt_item
        elif self.return_aqg:
            return pqg_item

    def get_target_split(self, user_log):

        start_point = np.where(user_log['split_ids'] == self.target_split[0])[0][0]
        end_point = np.where(user_log['split_ids'] == self.target_split[-1])[0][-1]
        print('start point', start_point, 'end point', end_point)
        exit(1)
        start_interaction = user_log['interaction_ids'][start_point]
        end_interaction = user_log['interaction_ids'][end_point]
        index = np.zeros_like(user_log['split_ids']).astype(bool)
        for split in self.target_split:
            index = index | np.where(user_log['split_ids'] == split, True, False)

        user_log_split = {
            'user_ascii': user_log['user_ascii'],
            'user_id': np.array([user_log['user_id']]),
            'user_ability': user_log['user_ability'],
            'word_ids': np.concatenate([user_log['word_ids'][0:1], user_log['word_ids'][index], user_log['word_ids'][-1:]], axis=0),
            'word_pos_tag_ids': np.concatenate([user_log['word_pos_tag_ids'][0:1], user_log['word_pos_tag_ids'][index], user_log['word_pos_tag_ids'][-1:]], axis=0),
            'w_l_tuple_ids': np.concatenate([user_log['w_l_tuple_ids'][0:1], user_log['w_l_tuple_ids'][index], user_log['w_l_tuple_ids'][-1:]], axis=0),
            'interaction_ids': np.concatenate([user_log['interaction_ids'][0:1], user_log['interaction_ids'][index], user_log['interaction_ids'][-1:]], axis=0),
            'labels': np.concatenate([user_log['labels'][0:1], user_log['labels'][index], user_log['labels'][-1:]], axis=0),
            'split_ids': np.concatenate([user_log['split_ids'][0:1], user_log['split_ids'][index], user_log['split_ids'][-1:]], axis=0),
            'adaptive_difficulties': user_log['adaptive_difficulties'][start_interaction: end_interaction + 1],
            'non_adaptive_difficulties': user_log['non_adaptive_difficulties'][start_interaction: end_interaction + 1],
        }

        return user_log_split

    def truncate(self, user_log, direction='left'):
        start_pos = end_pos = None
        start_interaction_id = end_interaction_id = None
        if direction == 'left':
            start_interaction_id = user_log['interaction_ids'][-self.max_seq_length + 1]  # reserve a place for <bos>
            if user_log['interaction_ids'][-self.max_seq_length] == start_interaction_id:
                start_interaction_id += 1  # keep whole interactions
            end_interaction_id = len(user_log['adaptive_difficulties'])
            start_pos = np.where(user_log['interaction_ids'] == start_interaction_id)[0][0]
            end_pos = len(user_log['word_ids'])

        elif direction == 'right':
            end_interaction_id = user_log['interaction_ids'][self.max_seq_length]
            start_interaction_id = 0
            if user_log['interaction_ids'][self.max_seq_length + 1] == end_interaction_id:
                end_interaction_id -= 1
            start_pos = 0
            end_pos = np.where(user_log['interaction_ids'] == end_interaction_id)[0][-1] + 1
            end_interaction_id += 1

        user_log_truncated = {
            'user_ascii': user_log['user_ascii'],
            'user_id': user_log['user_id'],
            'user_ability': user_log['user_ability'],
            'word_ids': np.append(np.array([self.vocab['<bos>']]), user_log['word_ids'][start_pos: end_pos], axis=0),
            'word_pos_tag_ids': np.append(np.array([self.pos_tag_map['<pad>']]), user_log['word_pos_tag_ids'][start_pos: end_pos]),
            'w_l_tuple_ids': np.append(np.array([self.w_l_tuple_map['<bos>']]), user_log['w_l_tuple_ids'][start_pos: end_pos], axis=0),
            'interaction_ids': np.append(np.array([-1]), user_log['interaction_ids'][start_pos: end_pos] - start_interaction_id, axis=0),
            'labels': np.append(np.array([self.pad_label_id]), user_log['labels'][start_pos: end_pos], axis=0),
            'split_ids': np.append(np.array([0]), user_log['split_ids'][start_pos: end_pos], axis=0),
            'non_adaptive_difficulties': user_log['non_adaptive_difficulties'][start_interaction_id: end_interaction_id],
            'adaptive_difficulties': user_log['adaptive_difficulties'][start_interaction_id: end_interaction_id],
            'valid_length': np.array([end_pos - start_pos + 1]),
            'valid_interactions': np.array([end_interaction_id - start_interaction_id])  # interaction_ids starts with -1
        }

        user_log_truncated['interaction_ids'][-1] = -1
        return user_log_truncated

    def pad(self, user_log, direction='right'):
        pad_length = self.max_seq_length - len(user_log['word_ids'])
        pad_width = None
        if direction == 'right':
            pad_width = ((0, pad_length),)
        elif direction == 'left':
            pad_width = ((pad_length, 0),)

        user_log_pad = {
            'user_ascii': user_log['user_ascii'],
            'user_id': user_log['user_id'],
            'user_ability': user_log['user_ability'],
            'word_pos_tag_ids': np.pad(user_log['word_pos_tag_ids'], pad_width, constant_values=((self.pos_tag_map['<pad>'], self.pos_tag_map['<pad>']),)),
            'non_adaptive_difficulties': user_log['non_adaptive_difficulties'],
            'adaptive_difficulties': user_log['adaptive_difficulties'],
            'word_ids': np.pad(user_log['word_ids'], pad_width, constant_values=((self.vocab['<pad>'], self.vocab['<pad>']),)),
            'w_l_tuple_ids': np.pad(user_log['w_l_tuple_ids'], pad_width, constant_values=((self.w_l_tuple_map['<pad>'], self.w_l_tuple_map['<pad>']),)),
            'interaction_ids': np.pad(user_log['interaction_ids'], pad_width, constant_values=((-1, -1),)),
            'labels': np.pad(user_log['labels'], pad_width, constant_values=((self.pad_label_id, self.pad_label_id),)),
            'split_ids': np.pad(user_log['split_ids'], pad_width, constant_values=((0, 0),))
        }
        return user_log_pad

    def construct_collate_fn(self):
        def collate_fn(batch_data):
            # pad num_questions
            pass

        return collate_fn


class DuolingoKTDataset(Dataset):

    def __init__(self, raw_data_file, data_dir, word_file, w_l_tuple_file, user_file, pos_tag_vocab_file, max_seq_len, target_split, label_pad_id=-100, interaction_pad_id=-100, time_pad_id=0,
                 days_pad_id=0, max_lines=-1, discard_rate=0.5):

        self.max_seq_len = max_seq_len
        self.label_pad_id = label_pad_id
        self.days_pad_id = days_pad_id
        self.time_pad_id = time_pad_id
        self.target_split = target_split
        self.interaction_pad_id = interaction_pad_id

        self.discard_rate = discard_rate

        self.num_days = 23
        self.num_time = 9

        self.word_map = {'<pad>': 0, '<bos>': 1, '<eos>': 2, '<unk>': 3}
        self.difficulty_map = {'<pad>': 0., '<bos>': 0., '<eos>': 0., '<unk>': 0.}

        special_word_num = len(self.word_map)
        df_words = pd.read_csv(word_file)
        for idx, row in df_words.iterrows():
            self.word_map[row['word']] = int(row['word_id']) + special_word_num
            self.difficulty_map[row['word']] = float(row['error_rate'])

        self.inverse_word_map = {self.word_map[word]: word for word in self.word_map}

        self.w_l_tuple_map = {'<pad>': 0, '<bos>': 1, '<eos>': 2, '<unk>': 3}
        special_w_l_tuple_num = len(self.w_l_tuple_map)
        df_w_l_tuples = pd.read_csv(w_l_tuple_file)
        for idx, row in df_w_l_tuples.iterrows():
            self.w_l_tuple_map[row['w_l_tuple']] = int(row['w_l_tuple_id']) + special_w_l_tuple_num

        self.inverse_w_l_tuple_map = {self.w_l_tuple_map[w_l_tuple]: w_l_tuple for w_l_tuple in self.w_l_tuple_map}

        self.task_map = {'<pad>': 0, 'reverse_translate': 1, 'reverse_tap': 2, 'listen': 3}
        self.split_map = {'<pad>': 0, 'train': 1, 'dev': 2, 'test': 3}

        self.pos_tag_map = {'<pad>': 0}
        df = pd.read_csv(pos_tag_vocab_file)
        for _, row in df.iterrows():
            self.pos_tag_map[row['pos_tag']] = len(self.pos_tag_map)

        self.user_map = {}
        df_users = pd.read_csv(user_file)
        for idx, row in df_users.iterrows():
            self.user_map[row['user_name']] = row['user_id']
        self.num_users = len(self.user_map)

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
            train_steps = np.where(data['split_ids'] == 1, True, False).sum()
            if train_steps < self.discard_rate * data['valid_length'][0]:
                continue
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
                filename = '{}.npz'.format('-'.join([str(num) for num in format_data['user_ascii']]))
                save_path = os.path.join(dirname, filename)
                np.savez(save_path, **format_data)
                line = fp.readline()
                pbar.update(1)
            pbar.close()

    def format(self, user_log, padding=False, truncation=False):
        # format SINGLE user_log to train data
        instance = {
            'user_ascii': ascii_encode(user_log['user_id']),
            'user_id': self.user_map[user_log['user_id']],
            'user_ability': [user_log['user_ability']],
            'word_ids': [],
            # 'word_attn_mask': None,
            'w_l_tuple_ids': [],
            # 'w_l_tuple_attn_mask': None,
            # 'position_ids': [],
            'task_ids': [],
            'interaction_ids': [],
            'labels': [],
            'split_ids': [],
            'non_adaptive_difficulties': [],
            'adaptive_difficulties': [],
            'word_pos_tag_ids': [],
            # 'days': [],
            # 'time': []
        }

        instance['word_ids'].append(self.word_map['<bos>'])
        instance['w_l_tuple_ids'].append(self.w_l_tuple_map['<bos>'])
        instance['task_ids'].append(self.task_map['<pad>'])
        instance['interaction_ids'].append(-1)
        instance['labels'].append(self.label_pad_id)
        instance['split_ids'].append(self.split_map['<pad>'])
        instance['word_pos_tag_ids'].append(self.pos_tag_map['<pad>'])
        # instance['days'].append(self.days_pad_id)
        # instance['time'].append(self.time_pad_id)

        cur_interaction_id = 0
        # words = []
        for split in ['train', 'dev', 'test']:
            if self.target_split and split not in self.target_split:
                continue
            for interaction_id, interaction in enumerate(user_log[split]):
                # days = self.discretize_days(interaction['days'])
                # time = self.discretize_time(interaction['time'])
                adaptive_difficulty = non_adaptive_difficulty = 0

                for idx, item in enumerate(interaction['exercise']):
                    # words.append(item['text'])
                    instance['word_ids'].append(self.word_map.get(item['text'], self.word_map['<unk>']))
                    instance['w_l_tuple_ids'].append(self.w_l_tuple_map.get('{}|{}'.format(item['text'], item['label']), self.w_l_tuple_map['<unk>']))
                    instance['task_ids'].append(self.task_map[interaction['format']])
                    instance['labels'].append(item['label'])
                    instance['split_ids'].append(self.split_map[split])
                    instance['interaction_ids'].append(cur_interaction_id)
                    instance['word_pos_tag_ids'].append(self.pos_tag_map[item['pos']])
                    # instance['days'].append(days)
                    # instance['time'].append(time)
                    adaptive_difficulty += item['label']
                    non_adaptive_difficulty += self.difficulty_map[item['text']]

                instance['adaptive_difficulties'].append(adaptive_difficulty)
                instance['non_adaptive_difficulties'].append(non_adaptive_difficulty)

                cur_interaction_id += 1

        instance['word_ids'].append(self.word_map['<eos>'])
        instance['w_l_tuple_ids'].append(self.w_l_tuple_map['<eos>'])
        instance['task_ids'].append(self.task_map['<pad>'])
        instance['interaction_ids'].append(-1)
        instance['split_ids'].append(self.split_map['<pad>'])
        instance['labels'].append(self.label_pad_id)
        instance['word_pos_tag_ids'].append(self.pos_tag_map['<pad>'])

        # instance['days'].append(self.days_pad_id)
        # instance['time'].append(self.time_pad_id)

        # instance['position_ids'] = [i for i in range(len(instance['word_ids']))]

        # instance['word_attn_mask'] = self.__build_word_attn_mask(instance)
        # instance['w_l_tuple_attn_mask'] = self.__build_w_l_tuple_attn_mask(instance)
        instance['valid_length'] = [len(instance['word_ids'])]
        instance['valid_interactions'] = [instance['interaction_ids'][-1] + 1]

        for key in instance:
            instance[key] = np.array(instance[key])

        if truncation and len(instance['word_ids']) > self.max_seq_len:
            self.truncate(instance)

        if padding and len(instance['word_ids']) < self.max_seq_len:
            self.pad(instance)

        return instance

    def check(self, mask):
        for idx, row in enumerate(mask):
            if not (row == False).any():
                return idx, row.shape
        return None

    def discretize_time(self, time):
        # size 9
        if time < 0:
            return 0  # invalid pad
        elif time <= 3:
            return 1
        elif 3 < time <= 5:
            return 2
        elif 5 < time <= 10:
            return 3
        elif 10 < time <= 15:
            return 4
        elif 15 < time <= 20:
            return 5
        elif 20 < time <= 30:
            return 6
        elif 30 < time <= 50:
            return 7
        else:
            return 8

    def discretize_days(self, days):
        # size 23
        if days < 0:
            return 0  # pad
        elif days < 20:
            return int(days) + 1
        elif 20 <= days < 25:
            return 21
        else:
            return 22

    def __build_word_attn_mask(self, data):
        seq_len = len(data['word_ids'])
        word_attn_mask = [[True for i in range(seq_len)] for j in range(seq_len)]

        for target_idx in range(seq_len):  # row
            for source_idx in range(seq_len):  # column
                if data['interaction_ids'][target_idx] == data['interaction_ids'][source_idx]:
                    word_attn_mask[target_idx][source_idx] = False
                    # (1) special_tokens attend to special tokens, and (2) words attend within the same interaction
                # elif data['interaction_ids'][target_idx] < data['interaction_ids'][source_idx]:
                #     break

        return word_attn_mask

    def __build_w_l_tuple_attn_mask(self, data):
        seq_len = len(data['word_ids'])
        w_l_tuple_attn_mask = [[True for i in range(seq_len)] for j in range(seq_len)]

        for target_idx in range(seq_len):  # row
            if target_idx == 0:
                w_l_tuple_attn_mask[0][0] = False  # <bos> attends only <bos>
            elif target_idx == seq_len - 1:
                w_l_tuple_attn_mask[target_idx] = [False for i in range(seq_len)]  # <eos> attends all
            else:
                for source_idx in range(seq_len):  # column
                    if data['interaction_ids'][target_idx] > data['interaction_ids'][source_idx] > self.interaction_pad_id:
                        w_l_tuple_attn_mask[target_idx][source_idx] = False  # attend to previous interactions
                    elif data['interaction_ids'][target_idx] == data['interaction_ids'][source_idx] == self.interaction_pad_id:
                        w_l_tuple_attn_mask[target_idx][source_idx] = False  # pad tokens attend to pad tokens

        return w_l_tuple_attn_mask

    def __build_memory_update_mask(self, data):
        memory_update_mask = [[0 for i in range(len(data['word_ids']))] for j in range(len(data['word_ids']))]
        for column_id in range(data['sep_indices']):
            if data['sep_indices'][column_id] == 0:
                continue

            memory_update_mask[column_id][column_id] = 1

            for row_id in range(column_id):
                if data['sep_indices'] == 1:
                    memory_update_mask[row_id][column_id] = 1

        return memory_update_mask

    def __getitem__(self, idx):
        return self.data[idx]
        # user_id, user_ability, word_ids, word_attn_mask, w_l_tuple_ids, x_w_l_tuple_attn_mask, position_ids, task_ids, interaction_ids, labels, split_ids, valid_length, valid_interactions

    def pad(self, tokenized, max_seq_len=None, direction='left'):
        # only pad word_ids and w_l_tuple_ids, not to pad sep_indices
        max_seq_len = max_seq_len if max_seq_len else self.max_seq_len

        seq_pad_length = max_seq_len - len(tokenized['word_ids'])

        if seq_pad_length <= 0:
            return

            # pad_word_ids = [self.word_map['<pad>'] for i in range(seq_pad_length)]
        # pad_w_l_tuple_ids = [self.w_l_tuple_map['<pad>'] for i in range(seq_pad_length)]
        # pad_task_ids = [self.task_map['<pad>'] for i in range(seq_pad_length)]
        # pad_labels = [self.label_pad_id for i in range(seq_pad_length)]
        # pad_split_ids = [self.split_map['<pad>'] for i in range(seq_pad_length)]
        # pad_interaction_ids = [self.interaction_pad_id for i in range(seq_pad_length)] # TODO: 是否-100 做填充？

        if direction == 'right':
            pad_width = (0, seq_pad_length)
            attn_pad_width = ((0, seq_pad_length), (0, seq_pad_length))
            tokenized['position_ids'] = np.array([i for i in range(max_seq_len - seq_pad_length)] + [0 for i in range(seq_pad_length)])
        elif direction == 'left':
            pad_width = (seq_pad_length, 0)
            attn_pad_width = ((seq_pad_length, 0), (seq_pad_length, 0))
            tokenized['position_ids'] = np.array([0 for i in range(seq_pad_length)] + [i for i in range(max_seq_len - seq_pad_length)])

        tokenized['word_ids'] = np.pad(tokenized['word_ids'], pad_width, constant_values=(self.word_map['<pad>'], self.word_map['<pad>']))
        tokenized['w_l_tuple_ids'] = np.pad(tokenized['w_l_tuple_ids'], pad_width, constant_values=(self.w_l_tuple_map['<pad>'], self.w_l_tuple_map['<pad>']))
        tokenized['task_ids'] = np.pad(tokenized['task_ids'], pad_width, constant_values=(self.task_map['<pad>'], self.task_map['<pad>']))
        tokenized['labels'] = np.pad(tokenized['labels'], pad_width, constant_values=(self.label_pad_id, self.label_pad_id))
        tokenized['split_ids'] = np.pad(tokenized['split_ids'], pad_width, constant_values=(self.split_map['<pad>'], self.split_map['<pad>']))
        tokenized['interaction_ids'] = np.pad(tokenized['interaction_ids'], pad_width, constant_values=(self.interaction_pad_id, self.interaction_pad_id))
        tokenized['days'] = np.pad(tokenized['days'], pad_width, constant_values=(self.days_pad_id, self.days_pad_id))
        tokenized['time'] = np.pad(tokenized['time'], pad_width, constant_values=(self.time_pad_id, self.time_pad_id))

        tokenized['word_attn_mask'] = np.pad(tokenized['word_attn_mask'], attn_pad_width, constant_values=((False, False), (True, True)))
        tokenized['w_l_tuple_attn_mask'] = np.pad(tokenized['w_l_tuple_attn_mask'], attn_pad_width, constant_values=((False, False), (True, True)))

    def truncate(self, tokenized, max_seq_len=None, direction='left'):
        max_seq_len = max_seq_len if max_seq_len else self.max_seq_len
        seq_len = len(tokenized['word_ids'])
        trunc_length = len(tokenized['word_ids']) - max_seq_len
        if trunc_length <= 0:
            return

        tokenized['valid_length'] = np.array([max_seq_len])

        if direction == 'right':
            tokenized['word_ids'] = np.concatenate([tokenized['word_ids'][:max_seq_len - 1], np.array([self.word_map['<eos>']])])
            tokenized['w_l_tuple_ids'] = np.concatenate([tokenized['w_l_tuple_ids'][:max_seq_len - 1], np.array([self.w_l_tuple_map['<eos>']])])
            tokenized['task_ids'] = np.concatenate([tokenized['task_ids'][:max_seq_len - 1], np.array([self.task_map['<pad>']])])
            tokenized['labels'] = np.concatenate([tokenized['labels'][:max_seq_len - 1], np.array([self.label_pad_id])])
            tokenized['split_ids'] = np.concatenate([tokenized['split_ids'][:max_seq_len - 1], np.array([self.split_map['<pad>']])])
            tokenized['interaction_ids'] = np.concatenate([tokenized['interaction_ids'][:max_seq_len - 1], np.array([tokenized['interaction_ids'][-1] + 1])])
            tokenized['valid_interactions'][0] = tokenized['interaction_ids'][-1] + 1
            tokenized['days'] = tokenized['days'][:max_seq_len]
            tokenized['time'] = tokenized['time'][:max_seq_len]
            tokenized['position_ids'] = tokenized['position_ids'][:max_seq_len]

            # truncate attn mask
            tokenized['word_attn_mask'] = tokenized['word_attn_mask'][:seq_len, :seq_len]
            tokenized['word_attn_mask'][-1, :] = True
            tokenized['word_attn_mask'][:, -1] = True
            tokenized['word_attn_mask'][-1, -1] = False

            tokenized['w_l_tuple_attn_mask'] = tokenized['w_l_tuple_attn_mask'][:seq_len, :seq_len]
            tokenized['w_l_tuple_attn_mask'][:, -1] = True
            tokenized['w_l_tuple_attn_mask'][-1, :] = False

        elif direction == 'left':
            tokenized['word_ids'] = np.concatenate([np.array([self.word_map['<bos>']]), tokenized['word_ids'][trunc_length + 1:]])
            tokenized['w_l_tuple_ids'] = np.concatenate([np.array([self.w_l_tuple_map['<bos>']]), tokenized['w_l_tuple_ids'][trunc_length + 1:]])
            tokenized['task_ids'] = np.concatenate([np.array([self.task_map['<pad>']]), tokenized['task_ids'][trunc_length + 1:]])
            tokenized['labels'] = np.concatenate([np.array([self.label_pad_id]), tokenized['labels'][trunc_length + 1:]])
            tokenized['split_ids'] = np.concatenate([np.array([self.split_map['<pad>']]), tokenized['split_ids'][trunc_length + 1:]])
            tokenized['position_ids'] = np.array([i for i in range(max_seq_len)])
            tokenized['days'] = np.concatenate([np.array([self.days_pad_id]), tokenized['days'][trunc_length + 1:]])
            tokenized['time'] = np.concatenate([np.array([self.time_pad_id]), tokenized['time'][trunc_length + 1:]])

            start_interaction = tokenized['interaction_ids'][trunc_length + 1]
            tokenized['interaction_ids'] = np.concatenate([np.array([-1]), tokenized['interaction_ids'][trunc_length + 1:] - start_interaction])
            tokenized['valid_interactions'][0] = tokenized['interaction_ids'][-1] + 1

            # truncate attn mask
            tokenized['word_attn_mask'] = tokenized['word_attn_mask'][trunc_length:, trunc_length:]
            tokenized['word_attn_mask'][:, 0] = True
            tokenized['word_attn_mask'][0, :] = True
            tokenized['word_attn_mask'][0, 0] = False

            tokenized['w_l_tuple_attn_mask'] = tokenized['w_l_tuple_attn_mask'][trunc_length:, trunc_length:]
            tokenized['w_l_tuple_attn_mask'][0, :] = True
            tokenized['w_l_tuple_attn_mask'][:, 0] = False
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
                # print('before', data['interaction_ids'].shape)
                self.truncate(data, max_seq_len=batch_max_seq_len, direction=direction)
                # print('after truncate', data['interaction_ids'].shape)
                self.pad(data, max_seq_len=batch_max_seq_len, direction=direction)
            #     print('after truncate and pad', data['interaction_ids'].shape)
            # exit(1)
            x_user_ascii = torch.tensor(np.stack([data['user_ascii'] for data in batch_data], axis=0))
            x_user_ids = torch.tensor(np.stack([data['user_id'] for data in batch_data], axis=0))
            x_user_abilities = torch.tensor(np.stack([data['user_ability'] for data in batch_data], axis=0))
            x_word_ids = torch.tensor(np.stack([data['word_ids'] for data in batch_data], axis=0))
            x_word_attn_masks = torch.tensor(np.stack([data['word_attn_mask'] for data in batch_data], axis=0))
            x_w_l_tuple_ids = torch.tensor(np.stack([data['w_l_tuple_ids'] for data in batch_data], axis=0))
            x_w_l_tuple_attn_masks = torch.tensor(np.stack([data['w_l_tuple_attn_mask'] for data in batch_data], axis=0))
            x_position_ids = torch.tensor(np.stack([data['position_ids'] for data in batch_data], axis=0))
            x_task_ids = torch.tensor(np.stack([data['task_ids'] for data in batch_data], axis=0))
            x_days = torch.tensor(np.stack([data['days'] for data in batch_data], axis=0))
            x_time = torch.tensor(np.stack([data['time'] for data in batch_data], axis=0))
            x_interaction_ids = torch.tensor(np.stack([data['interaction_ids'] for data in batch_data], axis=0))
            y_labels = torch.tensor(np.stack([data['labels'] for data in batch_data], axis=0))
            split_ids = torch.tensor(np.stack([data['split_ids'] for data in batch_data], axis=0))  # 0 train, 1 dev, 2 test
            x_valid_lengths = torch.tensor(np.stack([data['valid_length'] for data in batch_data], axis=0))
            x_valid_interactions = torch.tensor(np.stack([data['valid_interactions'] for data in batch_data], axis=0))

            return x_user_ascii, x_user_ids, x_user_abilities, x_word_ids, x_word_attn_masks, x_w_l_tuple_ids, x_w_l_tuple_attn_masks, x_position_ids, x_task_ids, x_days, x_time, x_interaction_ids, y_labels, split_ids, x_valid_lengths, x_valid_interactions

        return collate_fn


class LMKTDataset(Dataset):
    def __init__(self, data_file, vocab_file, tokenizer, max_length, max_span, target_split, max_lines=-1):
        self.tokenizer = tokenizer
        self.x_word_ids = []
        self.x_input_ids = []
        self.x_attention_mask = []
        self.x_span_ids = []
        self.y_labels = []
        self.y_splits = []
        self.x_valid_length = []
        self.max_length = max_length
        self.max_span = max_span
        self.target_split = target_split

        self.word_map = {}
        for _, row in pd.read_csv(vocab_file).iterrows():
            self.word_map[row['word']] = len(self.word_map)

        self.num_words = len(self.word_map)

        with open(data_file, 'r') as fp:
            line_cnt = 0
            for line in tqdm(fp.readlines()):
                line_cnt += 1
                if line_cnt > max_lines > 0:
                    break
                x_input_ids = [tokenizer.cls_token_id]
                x_attention_mask = [1]
                x_span_ids = []
                y_labels = []
                y_splits = []
                x_word_ids = []

                user_log = json.loads(line.strip())
                for split_id, split in [(1, 'train'), (2, 'dev'), (3, 'test')]:
                    if self.target_split and split not in self.target_split:
                        continue
                    for interaction_id, interaction in enumerate(user_log[split]):
                        for idx, item in enumerate(interaction['exercise']):
                            if item == 0:
                                ids = self.tokenizer(item['text'])['input_ids'][1:-1]
                            else:
                                ids = self.tokenizer(' ' + item['text'])['input_ids'][1:-1]

                            x_attention_mask.extend([1 for i in range(len(ids))])
                            x_span_ids.append([len(x_input_ids) - 1, len(x_input_ids) + len(ids) - 1])
                            x_input_ids.extend(ids)
                            y_labels.append(item['label'])
                            y_splits.append(split_id)
                            x_word_ids.append(self.word_map[item['text']])

                x_input_ids.append(tokenizer.sep_token_id)
                x_attention_mask.append(1)
                # x_word_ids.append(0)
                # y_labels.append(-100)
                # y_splits.append(0)

                if len(x_input_ids) > self.max_length:
                    x_input_ids, x_span_ids, x_attention_mask, x_word_ids, y_splits, y_labels = self.truncate(x_input_ids, x_span_ids, x_attention_mask, x_word_ids, y_splits, y_labels)
                elif len(x_input_ids) < self.max_length:
                    x_input_ids, x_attention_mask = self.pad(x_input_ids, x_attention_mask)

                self.x_valid_length.append([len(x_word_ids)])
                x_span_ids, x_word_ids, y_splits, y_labels = self.pad_span(x_span_ids, x_word_ids, y_splits, y_labels)

                try:
                    assert len(x_input_ids) == len(x_attention_mask) == self.max_length
                    assert len(x_span_ids) == len(x_word_ids) == len(y_splits) == len(y_labels) == self.max_span
                except AssertionError:
                    logging.error('fail to align length: x_input_ids {}, x_span_ids {}, x_word_ids {}, y_splits {}, y_labels {}, x_attention_mask {}'.format(
                        len(x_input_ids), len(x_span_ids), len(x_word_ids), len(y_splits), len(y_labels), len(x_attention_mask)
                    ))
                    exit(1)
                self.x_input_ids.append(x_input_ids)
                self.x_span_ids.append(x_span_ids)
                self.x_attention_mask.append(x_attention_mask)
                self.x_word_ids.append(x_word_ids)
                self.y_splits.append(y_splits)
                self.y_labels.append(y_labels)

    def truncate(self, x_input_ids, x_span_ids, x_attention_mask, x_word_ids, y_splits, y_labels):
        out = len(x_input_ids) - self.max_length
        x_input_ids = [self.tokenizer.cls_token_id] + x_input_ids[out + 1:]
        x_attention_mask = x_attention_mask[out:]

        start = None
        for i, span in enumerate(x_span_ids):
            if out >= span[1]:
                continue
            else:
                start = i
                break

        x_span_ids = x_span_ids[start:]
        for i, span in enumerate(x_span_ids):
            x_span_ids[i][0] -= out
            x_span_ids[i][1] -= out
        x_span_ids[0][0] = max(0, x_span_ids[0][0])

        y_splits = y_splits[start:]
        y_labels = y_labels[start:]
        x_word_ids = x_word_ids[start:]
        try:
            assert len(x_span_ids) == len(x_word_ids) == len(y_splits) == len(y_labels)
            assert x_span_ids[-1][0] < self.max_length
        except AssertionError:
            logging.error('truncation error: x_span {}, x_word_ids {} y_splits {}, y_labels {}, last span {} first span {}'.format(
                len(x_span_ids), len(x_word_ids), len(y_splits), len(y_labels), x_span_ids[-1], x_span_ids[0]
            ))
            exit(1)
        return x_input_ids, x_span_ids, x_attention_mask, x_word_ids, y_splits, y_labels

    def pad(self, x_input_ids, x_attention_mask):
        short = self.max_length - len(x_input_ids)
        x_input_ids.extend([self.tokenizer.pad_token_id for i in range(short)])
        x_attention_mask.extend([0 for i in range(short)])

        return x_input_ids, x_attention_mask

    def pad_span(self, x_span_ids, x_word_ids, y_splits, y_labels):
        x_span_ids = [[self.max_length - 2, self.max_length - 1] for i in range(self.max_span - len(x_span_ids))] + x_span_ids
        y_labels = [-100 for i in range(self.max_span - len(y_labels))] + y_labels
        y_splits = [0 for i in range(self.max_span - len(y_splits))] + y_splits
        x_word_ids = [0 for i in range(self.max_span - len(x_word_ids))] + x_word_ids
        try:
            assert len(x_span_ids) == len(x_word_ids) == len(y_labels) == len(y_splits) == self.max_span
        except AssertionError:
            logging.error('pad span: x_span_ids {}, x_word_ids {}, y_splits {}, y_labels {}'.format(
                len(x_span_ids), len(x_word_ids), len(y_splits), len(y_labels)
            ))
            exit(1)
        return x_span_ids, x_word_ids, y_splits, y_labels

    def __getitem__(self, item):
        return torch.tensor(self.x_input_ids[item]), torch.tensor(self.x_span_ids[item]), torch.tensor(self.x_attention_mask[item]), torch.tensor(self.x_word_ids[item]), torch.tensor(
            self.x_valid_length[item]), torch.tensor(self.y_splits[item]), torch.tensor(self.y_labels[item])

    def __len__(self):
        return len(self.x_input_ids)


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
                        self.difficulty.append(error_cnt / math.sqrt(len(interaction_log['exercise'])))
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

            y_exercise_labels[y_exercise_labels == self.tokenizer.pad_token_id] = self.label_pad_token_id
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(y_exercise_labels)

            return user_ids, difficulties, x_keyword_ids, x_attention_mask, y_exercise_labels, decoder_input_ids

        return collate_fn


class DuolingoNonAdaptiveGenDataset(Dataset):
    def __init__(self, data_file, tokenizer, model_name, sample_rate, idx, prepare_decoder_input_ids, x_max_length=15, y_max_length=30, enable_difficulty=True):
        self.enable_difficulty = enable_difficulty
        self.x_max_length = x_max_length
        self.y_max_length = y_max_length
        # self.sampler = sampler
        # self.x_difficulty_levels = []

        self.x_input_ids = []
        self.x_attention_mask = []
        self.x_difficulties = []
        self.x_difficulty_positions = []
        self.y_decoder_input_ids = []
        self.y_labels = []

        self.tokenizer = tokenizer
        if not self.tokenizer.pad_token:  # for gpt2
            self.tokenizer.pad_token = self.tokenizer.eos_token

        with open(data_file, 'r') as fp:
            for line in fp.readlines():
                instance = json.loads(line)
                if model_name == 'gpt2':
                    x_input = '{}{} = {}'.format(tokenizer.pad_token, ' '.join(instance['keywords'][str(sample_rate)][idx]), instance['text'])
                    y_output = '{} = {}'.format(''.join(instance['keywords'][str(sample_rate)][idx]), instance['text'])
                else:
                    x_input = tokenizer.pad_token + ' '.join(instance['keywords'][str(sample_rate)][idx])
                    y_output = instance['text']

                x_tokenized = self.tokenizer(x_input, padding='max_length', return_tensors='pt', max_length=self.x_max_length)
                self.x_input_ids.append(x_tokenized['input_ids'])
                self.x_difficulty_positions.append(torch.tensor([[1]]))
                self.x_attention_mask.append(x_tokenized['attention_mask'])
                self.x_difficulties.append(torch.tensor([[instance['sum_word_error_rate']]]))

                y_tokenized = self.tokenizer(y_output, padding='max_length', return_tensors='pt', max_length=self.y_max_length)
                y_tokenized['input_ids'][y_tokenized['input_ids'] == self.tokenizer.pad_token_id] = -100
                self.y_labels.append(y_tokenized['input_ids'])

                if prepare_decoder_input_ids:
                    self.y_decoder_input_ids.append(prepare_decoder_input_ids(y_tokenized['input_ids']))

        self.x_input_ids = torch.cat(self.x_input_ids, dim=0)
        self.x_attention_mask = torch.cat(self.x_attention_mask, dim=0)
        self.x_difficulties = torch.cat(self.x_difficulties, dim=0)
        self.y_decoder_input_ids = torch.cat(self.y_decoder_input_ids, dim=0)
        self.y_labels = torch.cat(self.y_labels, dim=0)

    def __getitem__(self, idx):
        return self.x_input_ids[idx], self.x_attention_mask[idx], self.x_difficulties[idx], self.y_decoder_input_ids[idx], self.y_labels[idx]

    def __len__(self):
        return self.x_input_ids.size(0)

    # def convert_difficulty(self, raw_score):
    #     level = int(raw_score // 0.5)
    #     if level not in self.difficulty_distributions:
    #         self.difficulty_distributions[level] = 0
    #     self.difficulty_distributions[level] += 1
    #
    #     if level <= 2:
    #         return level
    #     else:
    #         return 3

    @staticmethod
    def create_dataset(data_files, sampler, rounds=3):
        sample_rates = [0.0, 0.25, 0.5, 0.75, 1.0]
        for fn in data_files:
            print('processing {}'.format(fn))
            new_fp = open(fn + 'new', 'w')
            with open(fn, 'r') as fp:
                for line in fp.readlines():
                    data = json.loads(line.strip())
                    keywords = {rate: [] for rate in sample_rates}
                    # sample n times for each rate 
                    for sr in sample_rates:
                        for i in range(rounds):
                            if sr == 0.:
                                keywords[sr].append([])
                            elif sr == 1.:
                                tokens_ = copy.deepcopy(data['tokens'])
                                random.shuffle(tokens_)
                                keywords[sr].append(tokens_)
                            else:
                                sampled_words = sampler.sample(tokens=data['tokens'], pos_tags=data['pos_tags'], sample_rate=sr)
                                keywords[sr].append(sampled_words)
                    data['keywords'] = keywords
                    out = json.dumps(data)
                    # for sr in data['keywords']:
                    #     print(type(sr))
                    # for sr in json.loads(out)['keywords']:
                    #     print(type(sr))
                    # exit(1)
                    new_fp.write(out + '\n')


class WordSampler:
    def __init__(self):
        # sample prompt words from a sentence
        self.sample_priority = [
            ['NOUN', 'VERB'],
            ['ADJ', 'ADV'],
            ['PUNCT', 'SYM', 'X', 'ADP', 'AUX', 'INTJ', 'CCONJ', 'DET', 'PROPN', 'NUM', 'PART', 'SCONJ', 'PRON']
        ]

    def sample(self, tokens, pos_tags, sample_rate):
        # if len(tokens) == 1:
        #     return None

        assert len(tokens) == len(pos_tags)
        total_num = int(Decimal(len(tokens) * sample_rate).quantize(Decimal("1."), rounding="ROUND_HALF_UP"))

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
            sampled.update(random.sample(source, min(max(total_num - len(sampled), 0), len(source))))
            if len(sampled) >= total_num:
                break

        sampled = list(sampled)
        random.shuffle(sampled)

        return sampled
