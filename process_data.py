import sys, os, re, json, random
import copy, torch, argparse, configparser, logging, csv, math
from torch.utils.data import Dataset
from typing import List, Dict
from tqdm import tqdm
import numpy as np
import pandas as pd
from decimal import Decimal
from nltk import word_tokenize
from transformers import AutoModelForSeq2SeqLM, BartTokenizer



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



def build_dataset(train_raw, dev_raw, dev_key_raw, test_raw, test_key_raw, format_output, vocab_file, word_file, exercise_file, non_adaptive_gen_file, build_kt=True, build_gen=True, un_cased=True):
    '''
    Input:
    train_raw: train_set (txt)
    dev_raw, dev_key_raw: dev_set (txt)
    test_raw, test_raw_key: test_set (txt)
    Output:
    vocab_file: words + <word,label> pairs + 
    word_file: word statistics
    exercise_file: exercise statistics
    '''
    if not build_kt and not build_gen:
        return
    
    user_interactions = {}
    vocab = {
        '<sep>': 0,
        '<pad>': 1,
        '<unk>': 2,
        '<listen>': 3,
        '<reverse_translate>': 4,
        '<reverse_tap>': 5
    }
    word_map = {} # word, freq, correct_cnt, error_cnt 
    exercise_map = {} # exercise, freq, correct_cnt, error_cnt, average_num_errors

    # read train
    logging.info('-- reading train data...')
    train_num = parse_lines(train_raw, user_interactions, 'train', vocab, word_map, exercise_map, label_dict=None, un_cased=un_cased)
    logging.info('-- {} training interactions'.format(train_num))

    # read dev
    logging.info('-- reading dev data...')
    dev_keys = {}
    with open(dev_key_raw, 'r') as fp:
        for line in fp.readlines():
            item_id, label = line.strip().split(' ')
            dev_keys[item_id] = int(label)
    dev_num = parse_lines(dev_raw, user_interactions, 'dev', vocab, word_map, exercise_map, label_dict=dev_keys, un_cased=un_cased)
    logging.info('--{} dev interactions'.format(dev_num))

    # read test 
    logging.info('-- reading test data')
    test_keys = {}
    with open(test_key_raw, 'r') as fp:
        for line in fp.readlines():
            item_id, label = line.strip().split(' ')
            test_keys[item_id] = int(label)
    test_num = parse_lines(test_raw, user_interactions, 'test', vocab, word_map, exercise_map, label_dict=test_keys, un_cased=un_cased)
    logging.info('-- {} test interactions'.format(test_num))

    logging.info('-- saving vocab to {}'.format(vocab_file))
    vocab_table = [{
        'token': item[0],
        'token_id': item[1]
    } for item in vocab.items()]
    df = pd.DataFrame(vocab_table, columns=['token', 'token_id'])
    df.to_csv(vocab_file, index=False)
    
    logging.info('-- saving words to {}'.format(word_file))
    for word in word_map:
        word_map[word]['error_rate'] = word_map[word]['error_cnt'] / word_map[word]['cnt']
    df = pd.DataFrame(word_map.values(), columns=['word', 'cnt', 'error_cnt', 'error_rate'])
    df.to_csv(word_file, index=False)
    
    logging.info('saving exercies to {}'.format(exercise_file))

    for exercise in exercise_map:
        exercise_map[exercise]['exercise_error_rate'] = exercise_map[exercise]['exercise_error_cnt'] / exercise_map[exercise]['cnt']
        exercise_map[exercise]['avg_word_error_cnt'] = exercise_map[exercise]['word_error_cnt'] / exercise_map[exercise]['cnt']
        exercise_map[exercise]['sum_word_error_rate'] = sum([word_map[word]['error_rate'] for word in exercise.split('#')])



    df = pd.DataFrame(exercise_map.values(), columns=['exercise', 'cnt', 'exercise_error_cnt', 'word_error_cnt', 'exercise_error_rate', 'avg_word_error_cnt', 'sum_word_error_rate'])
    df.to_csv(exercise_file, index=False)

    ## format data for knowledge tracing
    '''
    format_output: combine train/dev/test in a jsonl file like:
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
    if build_kt:
        logging.info('-- saving format knowledge_tracing dataset to {}'.format(format_output))
        with open(format_output, 'w') as fp:
            for user in user_interactions:
                interaction = user_interactions[user]['train'][0]
                output_line = json.dumps({
                    'user': user,
                    'countries': user_interactions[user]['countries'],
                    'train': [interaction.to_dict() for interaction in user_interactions[user]['train']],
                    'dev': [interaction.to_dict() for interaction in user_interactions[user]['dev']],
                    'test': [interaction.to_dict() for interaction in user_interactions[user]['test']]
                })
                fp.write(output_line+'\n')

    ## construct exercises for non-adaptive generation 8:1:1
    if build_gen:
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

    

def parse_lines(data_file, user_interactions, split, vocab, word_map, exercise_map, label_dict=None, un_cased=True):
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
                    'user': interaction_log.user,
                    'countries': interaction_log.countries,
                    'train': [],
                    'dev': [],
                    'test': []
                }
            user_interactions[interaction_log.user][split].append(interaction_log)
            
            ## update exercise
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
                word_map[exercise_item.text] = {'word': exercise_item.text, 'cnt': 0, 'error_cnt': 0} # cnt, wrong_cnt
            word_map[exercise_item.text]['cnt'] += 1
            word_map[exercise_item.text]['error_cnt'] += exercise_item.label  

            # update vocabulary
            if split == 'train' or split == 'dev':
                ## update vocab
                if  'w#{}'.format(exercise_item.text) not in vocab:
                    vocab['w#{}'.format(exercise_item.text)] = len(vocab)

                if 'i#{}|{}'.format(exercise_item.text, exercise_item.label) not in vocab:
                    vocab['i#{}|{}'.format(exercise_item.text, exercise_item.label)] = len(vocab) # id, cnt


    fp.close()

    return update_cnt


def get_statistics(data_file, split=None):
    # TODO: update me
    # 0 train, 1 dev, 2 test, None all
    stats = {
        'user_cnt': 0,
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
        return len(self.x_keywords)


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
                
                # prompt words
                self.x_prompt_words.append(self.sampler.sample(tokens=data['tokens'], pos_tags=data['pos_tags']))


        logging.info(sorted(self.difficulty_distributions.items(), key=lambda x:x[0]))
        ## difficulty level distributions. (//0.5)
        ## train: [(0, 1660), (1, 3663), (2, 1489), (3, 279), (4, 44), (5, 12), (6, 8), (7, 2)]
        ## dev:
        ## test: 

    def __getitem__(self, idx):
        return self.x_difficulty_scores[idx], self.x_difficulty_levels, self.x_prompt_words[idx],  self.y_exercises[idx]

    
    def construct_collate_fn(self, tokenizer, x_max_length, y_max_length, padding='max_length', truncation=True, return_tensors='pt', label_pad_token_id=-100):
        
        def collate_fn(batch_data):    

            x_difficulty_scores = torch.tensor([data[0] for data in batch_data])
            x_difficulty_levels = torch.tenssor([data[1]] for data in batch_data)

            x_encoded = tokenizer([data[2] for data in batch_data], max_length=x_max_length, padding=padding, truncation=truncation, return_tensors=return_tensors)

            x_prompt_word_ids = x_encoded['input_ids']
            x_attention_mask = x_encoded['attention_mask']
            y_exercise_labels = self.tokenizer(
                [data[3] for data in batch_data],
                max_length=self.y_max_length,
                padding=self.padding,
                truncation=self.truncation,
                return_tensors=self.return_tensors
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


    # build_dataset(
    #     train_raw=args.duolingo_en_es_train_raw, 
    #     dev_raw=args.duolingo_en_es_dev_raw, 
    #     dev_key_raw=args.duolingo_en_es_dev_key_raw, 
    #     test_raw=args.duolingo_en_es_test_raw, 
    #     test_key_raw=args.duolingo_en_es_test_key_raw, 
    #     format_output=args.duolingo_en_es_format, 
    #     vocab_file=args.duolingo_en_es_vocab, 
    #     word_file=args.duolingo_en_es_words, 
    #     exercise_file=args.duolingo_en_es_exercises, 
    #     non_adaptive_gen_file=args.duolingo_en_es_non_adaptive_exercise_gen,
    #     build_kt=True,
    #     build_gen=True
    # )

    # sampler = WordSampler(sample_rate=0.5)


