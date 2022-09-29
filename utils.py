import os, sys, json, math, collections, torch
import numpy as np
import pandas as pd
from pprint import pprint
import matplotlib.pyplot as plt
import seaborn as sns


def sizeof(obj, unit='m'):
    size = 0
    if type(obj) == dict:
        for key in obj.keys():
            size += sizeof(key)
            size += sizeof(obj[key])
    
    elif type(obj) in [list, tuple]:
        for element in obj:
            size += sizeof(element)
    
    else:
        size = sys.getsizeof(obj) 
    
    return size


def ascii_encode(x):
    # str => int list 
    result = []
    for char in x:
        result.append(ord(char))
    return result


def ascii_decode(x):
    # int list => str
    result = []
    for number in x:
        result.append(chr(number))
    return ''.join(result)


def shift_sequence(sequence, offset):
    seq_len = sequence.size(1)
    if offset > 0: # shift right
        sequence = torch.cat([sequence[:, seq_len-offset:], sequence[:,:seq_len-offset]], dim=-1)
    elif offset < 0: # shift left
        sequence = torch.cat([sequence[:, -offset:], sequence[:, :-offset]], dim=-1)
    
    return sequence


def check_binary_matrix(matrix):
    records = []

    for i, row in enumerate(matrix):
        row_short = []
        pre_value = None
        cnt = 0
        for j, value in enumerate(row):
            if pre_value is not None and pre_value != value:
                row_short.append('{}*{}'.format(pre_value, cnt))
                cnt = 0
            cnt += 1
            pre_value = value

        row_short.append('{}*{}'.format(pre_value, cnt))
        
        records.append(row_short)

    return records


def is_int(number):
    try:
        int(number)
        return True
    except:
        return False

def format_distribution(distr, total):
    distr = collections.OrderedDict(sorted(distr.items(), key=lambda x:x[0]))
    acc = 0
    for key in distr:
        distr[key] = (distr[key], acc + (distr[key] / total), (distr[key] / total)) # bucket, cnt, acc_portion 
        acc = distr[key][1]
    
    return distr


def calc_word_error_rate(data_file, output_file, selection=[0, 1]):
    ## get word difficulty (non-adaptive)
    abilities = []

    word_error_rate = {}
    with open(data_file, 'r') as fp:
        lines = fp.readlines()
        for line in lines:
            user_log = json.loads(line.strip())
            abilities.append([user_log['user_id'], float(user_log['user_ability'])])
            
        abilities.sort(key=lambda x:x[1])

        target_students = set([item[0] for item in abilities[int(len(abilities)*selection[0]): int(len(abilities)*selection[1])]])

        for line in lines:
            user_log = json.loads(line.strip())
            if user_log['user_id'] not in target_students:
                continue
            for split in ['train', 'dev', 'test']:
                for interaction in user_log[split]:
                    for item in interaction['exercise']:
                        if item['text'] not in word_error_rate:
                            word_error_rate[item['text']] = [0, 0]
                        word_error_rate[item['text']][0] += item['label']
                        word_error_rate[item['text']][1] += 1
        
    for word in word_error_rate:
        word_error_rate[word] = [word, word_error_rate[word][1], word_error_rate[word][0] / word_error_rate[word][1]]


    df = pd.DataFrame(word_error_rate.values(), columns=['word', 'cnt', 'error_rate'])
    df.to_csv(output_file)



def calc_student_ability_distribution(data_file, split=1000):
    '''
    distribution of students' abilities (defined as correct_rate*ave_difficulty)
    '''
    ability_dist = {}
    total = 0
    with open(data_file, 'r') as fp:
        for line in fp.readlines():
            total += 1
            user_log = json.loads(line.strip())
            ability_score = user_log['user_ability']
            bucket = ability_score // split
            if bucket not in ability_dist:
                ability_dist[bucket] = 0
            ability_dist[bucket] += 1

    ability_dist = format_distribution(ability_dist, total)
    pprint(ability_dist)
    target_keys = [i for i in range(41, 57)]
    values = []

    buckets = list(ability_dist.keys())

    for key in target_keys:
        values.append(ability_dist[key][2])

    plt.bar(np.arange(len(values)), values, facecolor='lightskyblue')
    plt.xticks(np.arange(len(target_keys)), )
    plt.yticks([0.05*i for i in range(5)], labels=['{}%'.format(i*5) for i in range(5)])
    plt.show()

    # print(values)
    # pprint(ability_dist)



def calc_difficulty_trend(data_file, word_map_file, start_step=100, max_step=600, min_rate=0.1, fitting_line=True, average=True):
    '''
    the difficulty changing trend of exercises.
    '''
    word_error_rate = {}
    df = pd.read_csv(word_map_file)
    for idx, row in df.iterrows():
        word_error_rate[row['word']] = row['error_rate']


    step_ids = []
    step_absolute_difficulties = []
    step_relative_difficulties = []


    user_step_difficulty_collections = [] #[user_cnt, user_steps]
    with open(data_file, 'r') as fp:
        for line in fp.readlines():
            user_log = json.loads(line.strip())
            difficulty_trend = []
            for split in ['train', 'test', 'dev']:
                for interaction in user_log[split]:
                    absolute_difficulty = 0
                    relative_difficulty = 0
                    
                    for item in interaction['exercise']:
                        absolute_difficulty += word_error_rate[item['text']]
                        relative_difficulty += item['label']
                    
                    # absolute_difficulty /= len(interaction['exercise'])
                    # relative_difficulty = len(interaction['exercise'])
                    difficulty_trend.append([absolute_difficulty, relative_difficulty])
            
            user_step_difficulty_collections.append(difficulty_trend)


    for step in range(start_step, max_step): 
        cur_step_absolute_difficulties = []
        cur_step_relative_difficulties = []

        for user_steps in user_step_difficulty_collections:
            if step >= len(user_steps):
                continue
            cur_step_absolute_difficulties.append(user_steps[step][0])
            cur_step_relative_difficulties.append(user_steps[step][1])
        
        assert len(cur_step_relative_difficulties) == len(cur_step_absolute_difficulties)

        if len(cur_step_relative_difficulties) > min_rate * len(user_step_difficulty_collections): # minimum statistical base
            if average:
                step_ids.append(step)
                step_absolute_difficulties.append(sum(cur_step_absolute_difficulties)/len(cur_step_absolute_difficulties))
                step_relative_difficulties.append(sum(cur_step_relative_difficulties)/len(cur_step_relative_difficulties))
            else:
                step_ids.extend([step for i in range(len(cur_step_relative_difficulties))])
                step_absolute_difficulties.extend(cur_step_absolute_difficulties)
                step_relative_difficulties.extend(cur_step_relative_difficulties)
        else:
            break

    # plot absolute
    plt.plot(step_ids, step_absolute_difficulties, linewidth='1.5')
    if fitting_line:
        p_absolute = np.polyfit(step_ids, step_absolute_difficulties, 3)     ## fitting line
        p_absolute = np.poly1d(p_absolute)
        plt.plot(step_ids, p_absolute(step_ids), color='green', linestyle='--')
    plt.show()





def calc_ability_trend(result_dir, word_map_file, max_steps, group_size=50, min_rate=0.1, selection=None):

    word_difficulty_map = {}
    df = pd.read_csv(word_map_file)
    for idx, row in df.iterrows():
        word_difficulty_map[row['word']] = row['error_rate']

    target_users = None
    if selection:
        target_users = filter_users_by_abilities(data_file, selection=selection)


    kt_evaluator = KTEvaluator()
    

    user_step_abilities = [] # 4层列表 user_cnt, interaction_cnt, interaction_words, 2
    with open(data_file, 'r') as fp:
        for line in fp.readlines():
            user_log = json.loads(line.strip())
            if target_users and user_log['user_id'] not in target_users:
                continue
            step_info = []
            for split in ['train', 'dev', 'test']:
                for interaction in user_log[split]:
                    word_info = []

                    for item in interaction['exercise']:
                        if item['text'] not in word_difficulty_map:
                            continue
                        word_info.append([word_difficulty_map[item['text']], item['label']])

                    step_info.append(word_info)


            user_step_abilities.append(step_info)


    ability_trend = [] # step_num, computed_users 
    for start in range(0, max_steps, group_size):
        end = start + group_size
        
        user_cur_step_abilities = []
        for user_log in user_step_abilities:
            if end > len(user_log):
                continue
            
            ## calc abilities in within group size
            difficulty = 0
            error_rate = 0
            cnt = 0
            
            for group_id in range(start, end):
                for step_info in user_log[group_id]:
                    difficulty += step_info[0]
                    error_rate += step_info[1]
                    cnt += 1
            
            error_rate /= cnt
            cur_step_ability = difficulty * (1 - error_rate)
        
            user_cur_step_abilities.append(cur_step_ability)
        
        if len(user_cur_step_abilities) < min_rate * len(user_step_abilities):
            break # 统计量太小

        ability_trend.append(user_cur_step_abilities)
    
    # average users
    for i in range(len(ability_trend)):
        ability_trend[i] = sum(ability_trend[i]) / len(ability_trend[i]) / 50
    

    # pprint(ability_trend)


    # plot figure
    yticks = [0.05*i for i in range(4, 10)]
    plt.plot(np.arange(len(ability_trend)), ability_trend, c='green', marker='*')
    # plt.yticks(yticks, labels=['0.{}'.format(i*5) for i in range(4, 10)])
    # plt.xticks([i for i in range(len(ability_trend))], labels=[50*i for i in range(1, len(ability_trend)+1)])
    plt.xlabel('Num. Exercises')
    plt.ylabel('Knowledge Mastery')
    # plt.show()
    



def filter_users_by_abilities(data_file, selection):

    user_abilities = []
    with open(data_file, 'r') as fp:
        for line in fp.readlines():
            user_log = json.loads(line.strip())
            user_abilities.append([user_log['user_id'], user_log['user_ability']])
    
    user_abilities.sort(key=lambda x:x[1])

    target_users = [item[0] for item in user_abilities[int(selection[0]*len(user_abilities)): int(selection[1]*len(user_abilities))]]

    return set(target_users)



if __name__ == '__main__':
    # difficulty_trend = calc_difficulty_trend(data_file='/Users/cuipeng/Documents/Datasets/duolingo_2018_shared_task/data_en_es/en_es_format.jsonl', word_map_file='/Users/cuipeng/Documents/Datasets/duolingo_2018_shared_task/data_en_es/words.csv')
    # calc_student_ability_distribution(data_file='/Users/cuipeng/Documents/Datasets/duolingo_2018_shared_task/data_en_es/en_es_format.jsonl', split=0.01)

    # draw_picture()
 
    # calc_word_error_rate('/Users/cuipeng/Documents/Datasets/duolingo_2018_shared_task/data_en_es/en_es_format.jsonl', 'worst_student_error_rate.csv', selection=[0, 0.15])
    # calc_word_error_rate('/Users/cuipeng/Documents/Datasets/duolingo_2018_shared_task/data_en_es/en_es_format.jsonl', 'best_student_error_rate.csv', selection=[0.85, 1])
    # calc_word_error_rate('/Users/cuipeng/Documents/Datasets/duolingo_2018_shared_task/data_en_es/en_es_format.jsonl', 'medium_student_error_rate.csv', selection=[0.15, 0.85])

    
    calc_ability_trend(data_file='/Users/cuipeng/Documents/Datasets/duolingo_2018_shared_task/data_en_es/en_es_format.jsonl', word_map_file='worst_student_error_rate.csv', max_steps=1000, selection=[0, 0.15])
    calc_ability_trend(data_file='/Users/cuipeng/Documents/Datasets/duolingo_2018_shared_task/data_en_es/en_es_format.jsonl', word_map_file='best_student_error_rate.csv', max_steps=1000, selection=[0.85, 1])
    plt.show()
    # difficulty_calibration(
    #     word_file='worst_student_error_rate.csv',  
    #     generated_results={
    #         'bart-base': 'qg_eval_results/non_adaptive/bart_base_w_d_24_epoch.jsonl', 
    #         't5-base': 'qg_eval_results/non_adaptive/t5_base_w_39_epoch.jsonl', 
    #         'gpt-2': 'qg_eval_results/non_adaptive/bart_base_w_29_epoch.jsonl'
    #     }, 
    #     style='broken_line', 
    #     fitting_degree=5
    # )


    
