import json, torch, os, math

import nltk
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
from rouge import Rouge
import pandas as pd
import numpy as np
from nltk import word_tokenize
from utils import ascii_decode
from tqdm import tqdm
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import corpus_bleu
from transformers import BartTokenizer, GPT2LMHeadModel, GPT2TokenizerFast
from models import ConstrainedDecodingWithLookahead


class PersonalizedQGEvaluator:
    def __init__(self, word_map, difficulty_map):
        self.word_map = word_map
        self.difficulty_map = difficulty_map
        self.collections = {}
        self.rouge = Rouge()
        self.tokenizer = word_tokenize

    def clear(self):
        self.collections = {}

    def add(self, student_id, references, generations, ability, input_difficulties, generated_difficulties, target_words, knowledge_states,
            ground_truth_difficulties, estimated_difficulties, generated_word_difficulties, reference_word_difficulties):
        assert len(references) == len(generations) == len(input_difficulties) == len(target_words) == len(knowledge_states)  # == len(split_ids)
        self.collections[student_id] = []
        for idx in range(len(references)):
            self.collections[student_id].append({
                'reference': references[idx],
                'generated': generations[idx],
                'ability': ability[idx],
                'input_difficulty': input_difficulties[idx],
                'generated_difficulty': generated_difficulties[idx],
                'ground_truth_difficulty': ground_truth_difficulties[idx],
                'estimated_difficulty': estimated_difficulties[idx],
                'reference_word_difficulties': reference_word_difficulties[idx],
                # 'split_ids': split_ids,
                'target_words': target_words[idx],
                'knowledge_state': knowledge_states[idx],
                'generated_word_difficulties': generated_word_difficulties[idx]
            })

    def compute_metrics(self):
        results = {}

        rouge = self.compute_rouge()
        results.update(rouge)

        skill_coverage = self.compute_skill_coverage()
        results.update(skill_coverage)

        meteor = self.compute_meteor()
        results.update(meteor)

        difficulty_consistency = self.compute_difficulty_consistency()
        results.update(difficulty_consistency)

        bleu = self.compute_bleu()
        results.update(bleu)

        return results

    def compute_skill_coverage(self):
        cover_cnt = 0
        total_cnt = 0
        for student in self.collections:
            for idx in range(len(self.collections[student])):
                target_words = self.tokenizer(self.collections[student][idx]['target_words'])
                generated_words = self.tokenizer(self.collections[student][idx]['generated'])
                self.collections[student][idx]['skill-coverage'] = len((set(target_words) & set(generated_words))) / (len(target_words) + 1e-5)
                total_cnt += len(target_words)
                for word in target_words:
                    if word in generated_words:
                        cover_cnt += 1

        return {'skill_coverage': cover_cnt / total_cnt}

    def compute_difficulty_consistency(self):
        mae = 0
        # kt_upper_bound_mae = 0
        # avg_upper_bound_mae = 0
        count = 0

        for student in self.collections:
            count += len(self.collections[student])
            for idx in range(len(self.collections[student])):
                # compute generated difficulties
                # generated_words = self.tokenizer(self.collections[student][idx]['generated'])
                # reference_words = self.tokenizer(self.collections[student][idx]['reference'])
                #
                # generated_word_ids = [self.word_map.get(word, 0) for word in generated_words]
                # reference_word_ids = [self.word_map.get(word, 0) for word in reference_words]
                #
                # generated_word_difficulties = [self.collections[student][idx]['knowledge_state'][word_id] for word_id in generated_word_ids]
                # reference_word_difficulties = [self.collections[student][idx]['knowledge_state'][word_id] for word_id in reference_word_ids]
                #
                # reference_word_difficulties_avg = [self.difficulty_map.get(word, 0) for word in reference_words]  # TODO: the boss'answer is positive
                # reference_difficulty_avg = sum(reference_word_difficulties_avg)
                #
                # generated_difficulty = sum(generated_word_difficulties)
                # reference_difficulty = sum(reference_word_difficulties)

                # self.collections[student][idx]['generated_difficulty'] = self.collections[student][idx]['generated']
                # self.collections[student][idx]['reference_difficulty'] = reference_difficulty
                # self.collections[student][idx]['generated_word_difficulties'] = generated_word_difficulties
                # self.collections[student][idx]['reference_word_difficulties'] = reference_word_difficulties
                self.collections[student][idx]['difficulty_consistency'] = abs(self.collections[student][idx]['input_difficulty'] - self.collections[student][idx]['generated_difficulty'])

                mae += self.collections[student][idx]['difficulty_consistency']
                # kt_upper_bound_mae += abs(reference_difficulty - self.collections[student][idx]['adaptive_difficulty'])
                # avg_upper_bound_mae += abs(reference_difficulty_avg - self.collections[student][idx]['adaptive_difficulty'])

        mae /= count
        # kt_upper_bound_mae /= count
        # avg_upper_bound_mae /= count

        return {'difficulty_consistency': mae}

    def compute_bleu(self):

        reference_list = []
        generation_list = []

        for student in self.collections:
            for idx in range(len(self.collections[student])):
                reference_list.append([self.tokenizer(self.collections[student][idx]['reference'])])
                generation_list.append(self.tokenizer(self.collections[student][idx]['generated']))

        bleu_1 = corpus_bleu(reference_list, generation_list, weights=[1., 0., 0., 0.])
        bleu_2 = corpus_bleu(reference_list, generation_list, weights=[0., 1., 0., 0.])
        bleu_3 = corpus_bleu(reference_list, generation_list, weights=[0., 0., 1., 0.])
        bleu_4 = corpus_bleu(reference_list, generation_list, weights=[0., 0., 0., 1.])
        bleu_all = corpus_bleu(reference_list, generation_list, weights=[0.25, 0.25, 0.25, 0.25])

        return {'bleu_1': bleu_1, 'bleu_2': bleu_2, 'bleu_3': bleu_3, 'bleu_4': bleu_4, 'bleu_all': bleu_all}

    def compute_rouge(self):

        rouge_1 = rouge_2 = rouge_l = 0
        num_examples = 0

        for student in self.collections:

            for idx in range(len(self.collections[student])):
                num_examples += 1
                try:
                    score = self.rouge.get_scores([self.collections[student][idx]['generated']], [self.collections[student][idx]['reference']])
                    self.collections[student][idx]['rouge-1'] = score[0]['rouge-1']['f']
                    self.collections[student][idx]['rouge-2'] = score[0]['rouge-2']['f']
                    self.collections[student][idx]['rouge-l'] = score[0]['rouge-l']['f']
                    rouge_1 += score[0]['rouge-1']['f']
                    rouge_2 += score[0]['rouge-2']['f']
                    rouge_l += score[0]['rouge-l']['f']
                except ValueError:
                    self.collections[student][idx]['rouge-1'] = 0
                    self.collections[student][idx]['rouge-2'] = 0
                    self.collections[student][idx]['rouge-l'] = 0

        return {'rouge-1': rouge_1 / num_examples, 'rouge-2': rouge_2 / num_examples, 'rouge-l': rouge_l / num_examples}

    def compute_meteor(self):
        num_examples = 0
        meteor = 0
        for student in self.collections:
            num_examples += len(self.collections[student])
            for idx in range(len(self.collections[student])):
                sentence_meteor = meteor_score(
                    [self.tokenizer(self.collections[student][idx]['reference'])],
                    self.tokenizer(self.collections[student][idx]['generated'])
                )
                self.collections[student][idx]['meteor'] = sentence_meteor
                meteor += sentence_meteor

        return {'meteor': meteor/num_examples}

    def output(self):
        for student_id in self.collections:
            for interaction in self.collections[student_id]:
                interaction.pop('knowledge_state')
                # interaction['student_id'] = student_id
            print(json.dumps({'student_id': student_id, 'test_gen': self.collections[student_id]}))

    def compute_upper_bound(self):
        average_inconsistency = 0
        kt_inconsistency = 0
        for student_id in self.collections:
            for idx in range(len(self.collections[student_id])):
                generated_words = self.collections[student_id][idx]['reference']
            pass

    def check_diversity(self):
        pass


class QGEvaluator:
    def __init__(self, vocab_difficulty):

        self.vocab_difficulty = vocab_difficulty
        
        self.prompt_words = []
        self.generated = []
        self.generated_difficulty_scores = []
        self.reference = []
        self.reference_difficulty_scores = []

        self.rouge = Rouge()
        self.tokenizer = word_tokenize

        self.rouge_scores = []
        # self.grammar_checker = LanguageTool('en-US')
        # self.skill_coverage = []
        # self.bleu_scores = []
        # self.grammaticality = []
        # self.cider_score = []
        # self.meteor_score = []

    def read(self, filename):
        with open(filename, 'r') as fp:
            for line in fp.readlines():
                data = json.loads(line.strip())
                self.prompt_words.append(self.tokenizer(data['prompt_words']))
                self.generated.append(data['generated'])
                self.generated_difficulty_scores.append(data['generated_difficulty_score'])
                self.reference.append(data['reference'])
                self.reference_difficulty_scores.append(data['reference_difficulty_score'])

    def compute_metrics(self):
        # try:
        #     assert len(self.generated) == len(self.reference) == len(self.generated_difficulty_scores) == len(self.reference_difficulty_scores) == len(self.prompt_words)
        # except AssertionError:
        #     print(len(self.generated), len(self.reference), len(self.generated_difficulty_scores), len(self.reference_difficulty_scores), len(self.prompt_words))
        #     exit(1)

        result = {}
        
        rouge_score = self.compute_rouge()
        result.update(rouge_score)
         
        difficulty_consistency = self.compute_difficulty_consistency()
        result.update(difficulty_consistency)

        coverage = self.compute_skill_coverage()
        result.update(coverage)

        bleu_score = self.compute_bleu()
        result.update(bleu_score)

        meteor = self.compute_meteor()
        result.update(meteor)

        copy_rate = self.compute_copy()
        result.update(copy_rate)

        # answerable = self.compute_grammaticality()
        # result.update(answerable)

        return result

    def compute_rouge(self):
        for idx in range(len(self.generated)):
            try:
                score = self.rouge.get_scores([self.generated[idx]], [self.reference[idx]])
                self.rouge_scores.append(score[0])
            except ValueError:
                self.rouge_scores.append({'rouge-1': {'r': 0, 'p': 0, 'f': 0}, 'rouge-2': {'r': 0, 'p': 0, 'f': 0}, 'rouge-l': {'r': 0, 'p': 0, 'f': 0}})

        rouge_1 = rouge_2 = rouge_l = 0
        for score in self.rouge_scores:
            rouge_1 += score['rouge-1']['f']
            rouge_2 += score['rouge-2']['f']
            rouge_l += score['rouge-l']['f']
       
        return {'rouge-1': rouge_1/len(self.generated), 'rouge-2':rouge_2/len(self.generated), 'rouge-l': rouge_l/len(self.generated)}

    def compute_skill_coverage(self):
        cover_cnt = 0
        total_cnt = 0
        for idx in range(len(self.prompt_words)):
            prompt_words = self.tokenizer(self.prompt_words[idx])
            generated_words = self.tokenizer(self.generated[idx])
            total_cnt += len(prompt_words)
            for word in prompt_words:
                if word in generated_words:
                    cover_cnt += 1
        
        return {'skill_coverage': cover_cnt/total_cnt}

    def compute_difficulty_consistency(self):
        for generation in self.generated:
            self.generated_difficulty_scores.append(self.compute_difficulty_score(generation))

        difficulty_consistency = np.mean(np.abs(np.array(self.generated_difficulty_scores) - np.array(self.reference_difficulty_scores))) # MAE
        return {'difficulty_consistency': difficulty_consistency}

    def compute_difficulty_score(self, question, vocab_difficulty=None):
        difficulty_score = 0
        words = self.tokenizer(question)
        # logging.info('-- sentence {}'.format(sentence))
        # logging.info('-- words {}'.format(words))
        for word in words:
            if vocab_difficulty:
                if word not in vocab_difficulty:
                    continue
                word_difficulty = vocab_difficulty[word]
            else:
                if word not in self.vocab_difficulty:
                    continue  # TODO: how to handle?
                word_difficulty = self.vocab_difficulty[word]
            difficulty_score += word_difficulty
        return difficulty_score

    def compute_bleu(self):
        reference_list = [[self.tokenizer(reference)] for reference in self.reference]
        generated_list = [self.tokenizer(generated) for generated in self.generated]

        bleu_1 = corpus_bleu(reference_list, generated_list, weights=[1., 0., 0., 0.])
        bleu_2 = corpus_bleu(reference_list, generated_list, weights=[0., 1., 0., 0.])
        bleu_3 = corpus_bleu(reference_list, generated_list, weights=[0., 0., 1., 0.])
        bleu_4 = corpus_bleu(reference_list, generated_list, weights=[0., 0., 0., 1.])
        bleu_all = corpus_bleu(reference_list, generated_list, weights=[0.25, 0.25, 0.25, 0.25])

        return {'bleu_1': bleu_1, 'bleu_2': bleu_2, 'bleu_3': bleu_3, 'bleu_4': bleu_4, 'bleu_all': bleu_all}

    def compute_copy(self):
        copy_cnt = 0
        for i in range(len(self.generated)):
            if self.generated[i] == self.reference[i]:
                copy_cnt += 1
        return {'copy': copy_cnt/len(self.generated)}

    def compute_meteor(self):
        meteor = 0
        for i in range(len(self.generated)):
            sentence_meteor = meteor_score([self.tokenizer(self.reference[i])], self.tokenizer(self.generated[i]))
            meteor += sentence_meteor

        meteor /= len(self.generated)
        return {'meteor': meteor}

    def compute_cider(self):
        pass

    def compute_grammaticality(self):
        total_gen_word_cnt = 0
        total_ref_word_cnt = 0

        total_ref_error_cnt = 0
        total_gen_error_cnt = 0

        entities = [('europe', 'Europe'), ('friday', 'Friday'), ('australia', 'Australia'), ('spain', 'Spain'), ('italy', 'Italy'), ('english', 'English'), ('mexico', 'Mexico'), ('luis', 'Luis'),
                    ('thursday', 'Thursday'), ('saturday', 'Saturday'), ('november', 'November'), ('tuesday', 'Tuesday'), ('september', 'September'), ('england', 'England'), ('china', 'China'),
                    ('france', 'France'), ('american', 'American'), ('brazil', 'Brazil'), ('pedro', 'Pedro'), ('french', 'French'), ('paris', 'Paris'), ('barcelona', 'Barcelona'),
                    ('sunday', 'Sunday'), ('wednesday', 'Wednesday'), ('tv', 'TV')]

        for i in range(len(self.generated)):
            generated = self.generated[i].capitalize()
            reference = self.reference[i].capitalize()

            for pair in entities:
                if pair[0] in generated:
                    generated = generated.replace(pair[0], pair[1])
                if pair[0] in reference:
                    reference = reference.replace(pair[0], pair[1])

            gen_matches = self.grammar_checker.check(generated)
            ref_matches = self.grammar_checker.check(reference)

            total_gen_word_cnt += len(self.tokenizer(generated))
            total_ref_word_cnt += len(self.tokenizer(reference))
            for match in gen_matches:
                if match.category in ['PUNCTUATION']:
                    continue
                if 'LOWERCASE' in match.ruleId or 'UPPERCASE' in match.ruleId:
                    continue
                total_gen_error_cnt += 1

            for match in ref_matches:
                if match.category in ['PUNCTUATION']:
                    continue
                if 'LOWERCASE' in match.ruleId or 'UPPERCASE' in match.ruleId:
                    continue
                total_ref_error_cnt += 1

        result = {
            'generated_errors_per_1k': (total_gen_error_cnt / total_gen_word_cnt) * 1000,
            'reference_errors_per_1k': (total_ref_error_cnt / total_ref_word_cnt) * 1000
        }

        # print(total_gen_error_cnt, total_gen_word_cnt, total_gen_error_cnt / total_gen_word_cnt, total_gen_error_cnt / 895)
        # print(total_ref_error_cnt, total_ref_word_cnt, total_ref_error_cnt / total_ref_word_cnt, total_ref_error_cnt / 895)

        return result

    def output_result(self, filepath):
        with open(filepath, 'w') as fp:
            for idx in range(len(self.prompt_words)):
                fp.write(json.dumps({
                    'prompt_words': self.prompt_words[idx],
                    'reference': self.reference[idx],
                    'reference_difficulty_score': self.reference_difficulty_scores[idx],
                    'generated': self.generated[idx],
                    'generated_difficulty_score': self.generated_difficulty_scores[idx]
                })+'\n')

    def clear(self):
        self.prompt_words.clear()
        self.generated.clear()
        self.generated_difficulty_scores.clear()
        self.reference.clear()
        self.reference_difficulty_scores.clear()
        self.rouge_scores.clear()


class KTEvaluator:
    def __init__(self, num_words, vocabulary_difficulty, threshold=0.5, label_pad_id=-100):
        self.data = []
        self.threshold = threshold
        self.num_words = num_words
        self.vocabulary_difficulty = vocabulary_difficulty

    def clear(self):
        self.data.clear()

    def compute_metrics(self):
        # ROC and F1 score
        # pos_probs: [example_num, ]
        # labels: [example_num, ]  TODO: split seen and unseen evaluation, compute consistency,
        
        total_examples = len(self.data)

        results = {
            'train': {
                'roc_next_w': 0., 'roc_cur_w': 0., 'roc_next_q': 0., 'roc_seen_next_w': 0., 'roc_unseen_next_w': 0.,
                'f1_next_w': 0., 'f1_cur_w': 0., 'f1_next_q': 0., 'roc_seen_next_q': 0., 'roc_unseen_next_q': 0.,
            },
            'dev':  {
                'roc_next_w': 0., 'roc_cur_w': 0., 'roc_next_q': 0., 'roc_seen_next_w': 0., 'roc_unseen_next_w': 0.,
                'f1_next_w': 0., 'f1_cur_w': 0., 'f1_next_q': 0., 'roc_seen_next_q': 0., 'roc_unseen_next_q': 0.,
            },
            'test':  {
                'roc_next_w': 0., 'roc_cur_w': 0., 'roc_next_q': 0., 'roc_seen_next_w': 0., 'roc_unseen_next_w': 0.,
                'f1_next_w': 0., 'f1_cur_w': 0., 'f1_next_q': 0., 'roc_seen_next_q': 0., 'roc_unseen_next_q': 0.,
            }
        }
        
        data_collections = {
            'train': {
                'pos_probs_from_cur_w': np.array([]), 'pos_probs_from_last_w': np.array([]), 'pos_probs_from_last_q': np.array([]), 'question_labels': np.array([]),
                'labels': np.array([]), 'pred_labels_from_last_w': np.array([]), 'pred_labels_from_cur_w': np.array([]), 'question_pred_labels': np.array([]),
                'word_seen_labels': np.array([]), 'question_seen_labels': np.array([])
            },
            'dev': {
                'pos_probs_from_cur_w': np.array([]), 'pos_probs_from_last_w': np.array([]), 'pos_probs_from_last_q': np.array([]), 'question_labels': np.array([]),
                'labels': np.array([]), 'pred_labels_from_last_w': np.array([]), 'pred_labels_from_cur_w': np.array([]), 'question_pred_labels': np.array([]),
                'word_seen_labels': np.array([]), 'question_seen_labels': np.array([])
            },
            'test': {
                'pos_probs_from_cur_w': np.array([]), 'pos_probs_from_last_w': np.array([]), 'pos_probs_from_last_q': np.array([]), 'question_labels': np.array([]),
                'labels': np.array([]), 'pred_labels_from_last_w': np.array([]), 'pred_labels_from_cur_w': np.array([]), 'question_pred_labels': np.array([]),
                'word_seen_labels': np.array([]), 'question_seen_labels': np.array([])
            },
        }

        pbar = tqdm(total=total_examples)
        for data in self.data:
            for sid, split in ([(1, 'train'), (2, 'dev'), (3, 'test')]):
                valid_positions = np.where(data['split_ids'] == sid, True, False)  # filter other splits
                if not valid_positions.any():
                    # logging.info('-- Single: user {} has no data for {} evaluation'.format(ascii_decode(data['user_id']), split))
                    continue  # no such split data

                valid_labels = data['labels'][valid_positions]
                data_collections[split]['labels'] = np.append(data_collections[split]['labels'], valid_labels)

                # collect pos_probs for AUC
                # pred cur word
                valid_pred_pos_probs_from_cur_w = data['pred_pos_probs_from_cur_step'][valid_positions]
                data_collections[split]['pos_probs_from_cur_w'] = np.append(data_collections[split]['pos_probs_from_cur_w'], valid_pred_pos_probs_from_cur_w)
                pred_labels_from_cur_w = np.where(valid_pred_pos_probs_from_cur_w > 0.5, 1, 0)
                data_collections[split]['pred_labels_from_cur_w'] = np.append(data_collections[split]['pred_labels_from_cur_w'], pred_labels_from_cur_w)

                # pred next word
                valid_pred_pos_probs_from_last_w = data['pred_pos_probs_from_last_step'][valid_positions]
                data_collections[split]['pos_probs_from_last_w'] = np.append(data_collections[split]['pos_probs_from_last_w'], valid_pred_pos_probs_from_last_w)
                pred_labels_from_last_w = np.where(valid_pred_pos_probs_from_last_w > 0.5, 1, 0)
                data_collections[split]['pred_labels_from_last_w'] = np.append(data_collections[split]['pred_labels_from_last_w'], pred_labels_from_last_w)

                # pred_next question
                valid_pred_pos_probs_from_last_q = data['pred_pos_probs_from_last_question_last_step'][valid_positions]
                valid_interaction_ids = data['interaction_ids'][valid_positions]

                # collect collapsed labels  (Move to add)
                # collapsed_question_pred_pos_prob = []
                # collapsed_question_labels = []
                #
                # cur_question_pred = []
                # cur_question_label = []
                # pre_interaction_id = None
                # for i in range(len(valid_interaction_ids)):
                #     if valid_interaction_ids[i] == -1:
                #         continue
                #     if pre_interaction_id is not None and valid_interaction_ids[i] != pre_interaction_id:
                #         assert len(cur_question_pred) == len(cur_question_label)
                #         collapsed_question_pred_pos_prob.append(max(cur_question_pred))  # sum(cur_question_pred) / len(cur_question_pred)
                #         collapsed_question_labels.append(max(cur_question_label))
                #         cur_question_label.clear()
                #         cur_question_pred.clear()
                #
                #     pre_interaction_id = valid_interaction_ids[i]
                #     cur_question_pred.append(valid_pred_pos_probs_from_last_q[i])
                #     cur_question_label.append(valid_labels[i])
                #
                # if len(cur_question_pred) > 0:
                #     collapsed_question_pred_pos_prob.append(sum(cur_question_pred) / len(cur_question_pred))  # TODO: average or max?
                #     collapsed_question_labels.append(max(cur_question_label))

                # print('collapsed_pred_labels', len(collapsed_question_labels))
                # print('valid labels', len(valid_labels))
                # print('collapsed_question_labels', collapsed_question_labels)
                # print('collapsed_question_pred_pos_prob', collapsed_question_pred_pos_prob)
                # exit(1)

                valid_question_positions = np.where(data['question_split_ids'] == sid, True, False)
                collapsed_pred_labels = np.where(data['question_pred_pos_probs'][valid_question_positions] > 0.5, 1, 0)

                data_collections[split]['pos_probs_from_last_q'] = np.append(data_collections[split]['pos_probs_from_last_q'], data['question_pred_pos_probs'][valid_question_positions])
                data_collections[split]['question_labels'] = np.append(data_collections[split]['question_labels'], data['question_labels'][valid_question_positions])
                data_collections[split]['question_pred_labels'] = np.append(data_collections[split]['question_pred_labels'], collapsed_pred_labels)

                data_collections[split]['word_seen_labels'] = np.append(data_collections[split]['word_seen_labels'], data['word_seen_labels'][valid_positions])
                data_collections[split]['question_seen_labels'] = np.append(data_collections[split]['question_seen_labels'], data['question_seen_labels'][valid_question_positions])
                # collect pos_probs for acc/precision/recall
                # pred_labels_from_last_q = np.where(valid_pred_pos_probs_from_last_q > 0.5, 1, 0)
                # data_collections[split]['pred_labels_from_last_q'].extend(pred_labels_from_last_q)
                # valid_pos_probs_n = data['pos_probs_n'][valid_positions]
                # valid_pos_probs_c = data['pos_probs_c'][valid_positions]
                # valid_labels = data['labels'][valid_positions]
                # pred_labels_n = np.where(valid_pos_probs_n > 0.5, 1, 0)
                # pred_labels_c = np.where(valid_pos_probs_c > 0.5, 1, 0)
                # metrics[split]['labels'].extend(valid_labels)
                # metrics[split]['pred_labels_n'].extend(pred_labels_n)
                # metrics[split]['pred_labels_c'].extend(pred_labels_c)
                # metrics[split]['pos_probs_n'].extend(valid_pos_probs_n)
                # metrics[split]['pos_probs_c'].extend(valid_pos_probs_c)

            pbar.update(1)
        pbar.close()

        for split in ['train', 'dev', 'test']:

            # metrics[split]['pred_labels_n'] = np.array(metrics[split]['pred_labels_n'])
            # metrics[split]['pred_labels_c'] = np.array(metrics[split]['pred_labels_c'])
            # metrics[split]['labels'] = np.array(metrics[split]['labels'])
            # metrics[split]['pos_probs_n'] = np.array(metrics[split]['pos_probs_n'])
            # metrics[split]['pos_probs_c'] = np.array(metrics[split]['pos_probs_c'])

            if len(data_collections[split]['pos_probs_from_cur_w']) == 0:
                # logging.warning('-- Total: no data for {} evaluation'.format(split))
                continue

            # print(metrics[split]['labels'].shape, metrics[split]['pos_probs'].shape)
            # print(set(metrics[split]['labels'].tolist()))
            #  'f1_score': 0, 'accuracy': 0, 'recall': 0, 'precision': 0, 'avg_mse': 0, 'kt_mse': 0
            #   'labels': [], 'pred_labels_from_last_w': [], 'pred_labels_from_cur_w': [], 'pred_labels_from_last_q': [], 'pred_labels_from_cur_q': []
            results[split]['f1_next_w'] = f1_score(y_true=data_collections[split]['labels'], y_pred=data_collections[split]['pred_labels_from_last_w'])
            results[split]['f1_cur_w'] = f1_score(y_true=data_collections[split]['labels'], y_pred=data_collections[split]['pred_labels_from_cur_w'])
            results[split]['roc_next_w'] = roc_auc_score(y_true=data_collections[split]['labels'], y_score=data_collections[split]['pos_probs_from_last_w'])
            results[split]['roc_cur_w'] = roc_auc_score(y_true=data_collections[split]['labels'], y_score=data_collections[split]['pos_probs_from_cur_w'])

            results[split]['f1_next_q'] = f1_score(y_true=data_collections[split]['question_labels'], y_pred=data_collections[split]['question_pred_labels'])
            results[split]['roc_next_q'] = roc_auc_score(y_true=data_collections[split]['question_labels'], y_score=data_collections[split]['pos_probs_from_last_q'])

            w_seen_labels = np.where(data_collections[split]['word_seen_labels'] == 1, True, False)
            w_unseen_labels = np.where(data_collections[split]['word_seen_labels'] == 1, False, True)
            q_seen_labels = np.where(data_collections[split]['question_seen_labels'] == 1, True, False)
            q_unseen_labels = np.where(data_collections[split]['question_seen_labels'] == 1, False, True)

            results[split]['roc_unseen_next_w'] = roc_auc_score(y_true=data_collections[split]['labels'][w_unseen_labels], y_score=data_collections[split]['pos_probs_from_last_w'][w_unseen_labels])
            results[split]['roc_seen_next_w'] = roc_auc_score(y_true=data_collections[split]['labels'][w_seen_labels], y_score=data_collections[split]['pos_probs_from_last_w'][w_seen_labels])

            results[split]['roc_unseen_next_q'] = roc_auc_score(y_true=data_collections[split]['question_labels'][q_unseen_labels], y_score=data_collections[split]['pos_probs_from_last_q'][q_unseen_labels])
            results[split]['roc_seen_next_q'] = roc_auc_score(y_true=data_collections[split]['question_labels'][q_seen_labels], y_score=data_collections[split]['pos_probs_from_last_q'][q_seen_labels])
            # results[split]['roc_n'] = roc_auc_score(y_true=metrics[split]['labels'], y_score=metrics[split]['pos_probs_n'])
            # results[split]['roc_c'] = roc_auc_score(y_true=metrics[split]['labels'], y_score=metrics[split]['pos_probs_c'])
            # # next predictions
            # results[split]['f1_score'] = f1_score(y_true=metrics[split]['labels'], y_pred=metrics[split]['pred_labels_n'])
            # results[split]['precision'] = precision_score(y_true=metrics[split]['labels'], y_pred=metrics[split]['pred_labels_n'])
            # results[split]['recall'] = recall_score(y_true=metrics[split]['labels'], y_pred=metrics[split]['pred_labels_n'])
            # results[split]['accuracy'] = accuracy_score(y_true=metrics[split]['labels'], y_pred=metrics[split]['pred_labels_n'])

        # exit(1)
        return results

    def add(self, user_id, user_ability, word_ids, pred_pos_probs_from_cur_step, pred_pos_probs_from_last_step,
            pred_pos_probs_from_last_question_last_step, labels, split_ids, interaction_ids, mastery_probs, valid_length, valid_interactions, pad='right'):

        data = {
            'user_id': user_id,
            'user_ability': user_ability,
            'word_ids': word_ids[:valid_length] if pad == 'right' else word_ids[-valid_length:],
            'pred_pos_probs_from_cur_step': pred_pos_probs_from_cur_step[:valid_length] if pad == 'right' else pred_pos_probs_from_cur_step[-valid_length:],
            'pred_pos_probs_from_last_step': pred_pos_probs_from_last_step[:valid_length] if pad == 'right' else pred_pos_probs_from_last_step[-valid_length:],
            'pred_pos_probs_from_last_question_last_step': pred_pos_probs_from_last_question_last_step[:valid_length] if pad == 'right' else pred_pos_probs_from_last_question_last_step[-valid_length:],
            'labels': labels[:valid_length] if pad == 'right' else labels[-valid_length:],
            'split_ids': split_ids[:valid_length] if pad == 'right' else split_ids[-valid_length:],
            'interaction_ids': interaction_ids[:valid_length] if pad == 'right' else interaction_ids[-valid_length:],
            'memory_states': mastery_probs[:valid_length] if pad == 'right' else mastery_probs[-valid_length:],  # [interaction_num+1, num_words]
            'mastery_level': np.mean(mastery_probs, axis=-1)[:valid_length] if pad == 'right' else np.mean(mastery_probs, axis=-1)[-valid_length:]  # [interaction_num+1, ]
        }

        word_seen_labels = []
        for i in range(len(data['word_ids'])):
            if data['word_ids'][i] in data['word_ids'][:i]:
                word_seen_labels.append(1)
            else:
                word_seen_labels.append(0)
        data['word_seen_labels'] = np.array(word_seen_labels)
        assert len(word_seen_labels) == len(data['word_ids'])
        all_question_ids = []  # previous questions for constructing seen/unseen labels

        collapsed_seen_labels = []  # question-level seen_labels
        collapsed_question_pred_pos_prob = []  # question-level pos_probs
        collapsed_question_labels = []  # question-level true label
        collapsed_question_split_ids = []  # question-level split_ids

        cur_question_ids = []
        cur_question_pred = []
        cur_question_label = []
        cur_question_split = []
        pre_interaction_id = None
        for i in range(len(data['interaction_ids'])):
            if data['interaction_ids'][i] == -1:
                continue
            if pre_interaction_id is not None and data['interaction_ids'][i] != pre_interaction_id:
                assert len(cur_question_pred) == len(cur_question_label)
                collapsed_question_pred_pos_prob.append(max(cur_question_pred))  # sum(cur_question_pred) / len(cur_question_pred)
                collapsed_question_labels.append(max(cur_question_label))
                collapsed_question_split_ids.append(cur_question_split[0])  # TODO: now assume words of a question have the same split_id
                collapsed_seen_labels.append(1 if '$'.join(cur_question_ids) in all_question_ids else 0)

                all_question_ids.append('$'.join(cur_question_ids))

                cur_question_label.clear()
                cur_question_pred.clear()
                cur_question_ids.clear()
                cur_question_split.clear()

            pre_interaction_id = data['interaction_ids'][i]
            cur_question_pred.append(data['pred_pos_probs_from_last_question_last_step'][i])
            cur_question_label.append(data['labels'][i])
            cur_question_split.append(data['split_ids'][i])
            cur_question_ids.append(str(data['word_ids'][i]))

        if len(cur_question_pred) > 0:
            collapsed_question_pred_pos_prob.append(max(cur_question_pred))  # TODO: average or max?
            collapsed_question_labels.append(max(cur_question_label))
            collapsed_question_split_ids.append(cur_question_split[0])  # TODO: max times?
            collapsed_seen_labels.append(1 if '$'.join(cur_question_ids) in all_question_ids else 0)

        assert len(collapsed_question_labels) == len(collapsed_question_pred_pos_prob) == len(collapsed_question_split_ids) == len(collapsed_seen_labels)

        data['question_labels'] = np.array(collapsed_question_labels)
        data['question_pred_pos_probs'] = np.array(collapsed_question_pred_pos_prob)
        data['question_split_ids'] = np.array(collapsed_question_split_ids)
        data['question_seen_labels'] = np.array(collapsed_seen_labels)

        # print('word num', len(data['word_ids']), 'question num', len(data['question_labels']))
        # exit(1)

        self.data.append(data)

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
        # select target users (by abilities)
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



def annotate_question_difficulty(question_file, word_map, word_difficulty, save_path):
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    inverse_word_map = {word_map[word]: word for word in word_map}  # id2word
    word_id_difficulty = np.array([word_difficulty[inverse_word_map[word_id]] for word_id in range(len(word_difficulty))])  # vocab_size
    token_skill_id_map = ConstrainedDecodingWithLookahead.map_ids(word_map, tokenizer, oov_id=word_map['<pad>']).numpy()  # [sub_word_size]
    mask_oov = np.where(token_skill_id_map == word_map['<pad>'], 0, 1)

    # print('token_skill_id_map', token_skill_id_map.shape, token_skill_id_map.tolist())
    # print('word_id_difficulty', word_id_difficulty)

    sub_word_difficulties = word_id_difficulty[token_skill_id_map] * mask_oov
    # print('sub_word_difficulties', sub_word_difficulties.shape)

    fp_out = open(save_path, 'w')

    with open(question_file, 'r') as fp:
        for line in fp.readlines():
            data = json.loads(line.strip())
            input_ids = tokenizer(data['text'], max_length=30, return_tensors='pt')['input_ids'].numpy()[0]
            kt_difficulty = float(sum(sub_word_difficulties[input_ids]))
            data['kt_estimated_difficulty'] = kt_difficulty
            fp_out.write(json.dumps(data)+'\n')
            # print(kt_difficulty, type(kt_difficulty))
            # exit(1)
            # print(data['text'], input_ids)
            # print(sub_word_difficulties[input_ids])
            # exit(1)

    fp_out.close()


def check_validness(result_file, exercise_file):
    from language_tool_python import LanguageTool
    grammar_checker = LanguageTool('en-US')
    cache = {}

    seen_exercises = set()
    with open(exercise_file, 'r') as fp:
        for line in fp.readlines():
            data = json.loads(line)
            seen_exercises.add(data['text'])

    result = {
        'seen': {'error_cnt': 0, 'error_exercise_cnt': 0, 'total_word_cnt': 0, 'total_exercise_cnt': 0, 'uniq_exercises': set([]), 'error_exercises': set([])},
        'unseen': {'error_cnt': 0, 'error_exercise_cnt': 0, 'total_word_cnt': 0, 'total_exercise_cnt': 0, 'uniq_exercises': set([]), 'error_exercises': set([])}
    }

    exc_cnt = 0
    line_cnt = 0
    with open(result_file, 'r') as fp:
        for line in tqdm(fp.readlines()):
            line_cnt += 1
            data = json.loads(line.strip())
            if 'test_gen' not in data:
                continue
            for item in data['test_gen']:
                if item['reference'] in seen_exercises:
                    key = 'seen'
                else:
                    key = 'unseen'

                if item['generated'] in cache:
                    result[key]['error_cnt'] += cache[item['generated']][0]  # total_gen_error_cnt
                    result[key]['total_word_cnt'] += cache[item['generated']][1]  # total_gen_word_cnt
                    result[key]['uniq_exercises'].add(item['generated'])
                    if cache[item['generated']][0] > 0:
                        result[key]['error_exercise_cnt'] += 1
                        result[key]['error_exercises'].add(item['generated'])
                    result[key]['total_exercise_cnt'] += 1
                    continue

                result[key]['uniq_exercises'].add(item['generated'])

                try:
                    gen_matches = grammar_checker.check(item['generated'])
                except Exception:
                    exc_cnt += 1
                    continue
                error_cnt = 0
                word_cnt = len(nltk.word_tokenize(item['generated']))
                for match in gen_matches:
                    if match.category in ['PUNCTUATION']:
                        continue
                    if 'LOWERCASE' in match.ruleId or 'UPPERCASE' in match.ruleId:
                        continue
                    if match.category in ['CASING', 'TYPOS'] and match.replacements[0].lower() == item['generated'][match.offset: match.offset+match.errorLength]:
                        continue  # ignore casing errors
                    error_cnt += 1

                if error_cnt > 0:
                    # error_gen_cnt += 1
                    result[key]['error_exercise_cnt'] += 1
                    result[key]['error_exercises'].add(item['generated'])
                # total_gen_cnt += 1
                result[key]['total_exercise_cnt'] += 1
                # total_gen_error_cnt += error_cnt
                # total_gen_word_cnt += word_cnt

                result[key]['error_cnt'] += error_cnt
                result[key]['total_word_cnt'] += word_cnt

                cache[item['generated']] = (error_cnt, word_cnt)

    result['seen']['uniq_exercises'] = len(result['seen']['uniq_exercises'])
    result['seen']['error_exercises'] = len(result['seen']['error_exercises'])

    result['unseen']['uniq_exercises'] = len(result['unseen']['uniq_exercises'])
    result['unseen']['error_exercises'] = len(result['unseen']['error_exercises'])

    print('{}/{} skipped'.format(exc_cnt, line_cnt))
    print('seen', result['seen'])
    print('unseen', result['unseen'])



def check_diversity(result_file, exercise_file):
    exercise_pool = set()
    df = pd.read_csv(exercise_file)
    for idx, row in df.iterrows():
        exercise_pool.add(row['exercise'].replace('#', ' '))
    questions = []
    novel_questions = set()
    with open(result_file, 'r') as fp:
        for line in fp.readlines():
            data = json.loads(line.strip())
            if 'test_gen' not in data:
                continue
            for item in data['test_gen']:
                questions.append(item['generated'])
                if item['generated'] not in exercise_pool:
                    novel_questions.add(item['generated'])
    unique_questions = set(questions)
    print('total questions:', len(questions), 'unique_questions:', len(unique_questions), 'novel_questions', len(novel_questions))


def perplexity(result_file, device):
    model_id = "gpt2-base"
    model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
    tokenizer = GPT2TokenizerFast.from_pretrained(model_id)

    total_loss = 0
    total_tokens = 0
    with open(result_file, 'r') as fp:
        for line in fp.readlines():
            data = json.loads(line.strip())
            generated = [item['generated'] for item in data['test_gen']]
            encodings = tokenizer(generated,  return_tensors="pt")
            labels = encodings.input_ids.clone()
            labels[encodings.attention_masks == 0] = -100
            outputs = model(encodings, labels=labels)
            total_tokens += torch.sum(encodings.attention_masks)
            total_loss += outputs.loss * torch.sum(encodings.attention_masks)

    ppl = math.exp(total_loss / total_tokens)

    return ppl



def split_eg_result(result_file, seen_exercise_file):
    # split result according to seen/unseen questions

    def calculate_result(output_collections):
        result = {
            'rouge-1': 0, 'rouge-2': 0, 'rouge-l': 0, 'meteor': 0, 'skill_coverage': 0, 'difficulty_consistency': 0, 'bleu-1': 0, 'bleu-2': 0, 'bleu-3': 0, 'bleu-4': 0, 'bleu-all': 0,
        }

        # rouge / meteor
        rouge = Rouge()
        total_target = 1e-6
        covered_target = 0
        reference_list = []
        generation_list = []
        for i, c in enumerate(output_collections):
            try:
                tokenized_reference = word_tokenize(c['reference'])
                tokenized_generated = word_tokenize(c['generated'])

                reference_list.append([tokenized_reference])
                generation_list.append(tokenized_generated)

                score = rouge.get_scores(c['generated'], c['reference'])
                result['rouge-1'] += score[0]['rouge-1']['f']
                result['rouge-2'] += score[0]['rouge-2']['f']
                result['rouge-l'] += score[0]['rouge-l']['f']
                result['meteor'] += meteor_score(
                    [tokenized_reference],
                    tokenized_generated
                )

                # result['difficulty_consistency'] += abs(c['input_difficulty'] - c['generated_difficulty'])
                # target_words = word_tokenize(c['target_words'])
                # total_target += len(target_words)
                # for word in target_words:
                #     if word in tokenized_generated:
                #         covered_target += 1

            except ValueError:
                continue
        # print(len(reference_list))
        # print(len(generation_list))
        result['rouge-l'] /= len(output_collections)
        result['rouge-1'] /= len(output_collections)
        result['rouge-2'] /= len(output_collections)
        result['meteor'] /= len(output_collections)
        result['difficulty_consistency'] /= len(output_collections)
        result['skill_coverage'] = covered_target / total_target

        # Bleu
        result['bleu-1'] = corpus_bleu(reference_list, generation_list, weights=[1., 0., 0., 0.])
        result['bleu-2'] = corpus_bleu(reference_list, generation_list, weights=[0., 1., 0., 0.])
        result['bleu-3'] = corpus_bleu(reference_list, generation_list, weights=[0., 0., 1., 0.])
        result['bleu-4'] = corpus_bleu(reference_list, generation_list, weights=[0., 0., 0., 1.])
        result['bleu-all'] = corpus_bleu(reference_list, generation_list, weights=[0.25, 0.25, 0.25, 0.25])

        return result

    seen_exercises = set()
    with open(seen_exercise_file, 'r') as fp:
        for line in fp.readlines():
            data = json.loads(line)
            seen_exercises.add(data['text'])

    seen_collections = []
    unseen_collections = []
    with open(result_file, 'r') as fp:
        for line in fp.readlines():
            data = json.loads(line.strip())
            if 'test_gen' not in data:
                print('overall_result', data)
                continue
            for item in data['test_gen']:
                if item['reference'] in seen_exercises:
                    seen_collections.append(item)
                else:
                    unseen_collections.append(item)

    seen_result = calculate_result(seen_collections)
    unseen_result = calculate_result(unseen_collections)

    print('seen_result', len(seen_collections),  seen_result)
    print('unseen_result', len(unseen_collections),  unseen_result)


def check_validness_new(result_file, exercise_file):
    from language_tool_python import LanguageTool
    grammar_checker = LanguageTool('en-US')

    seen_exercises = set()
    with open(exercise_file, 'r') as fp:
        for line in fp.readlines():
            data = json.loads(line)
            seen_exercises.add(data['text'])

    line_cnt = 0
    exc_cnt = 0
    novel_exercises = set([])
    with open(result_file, 'r') as fp:
        for line in tqdm(fp.readlines()):
            line_cnt += 1
            data = json.loads(line.strip())
            if 'test_gen' not in data:
                continue
            for item in data['test_gen']:
                if item['generated'] in seen_exercises:
                    continue
                else:
                    novel_exercises.add(item['generated'])

    result = {'novel_exercise_cnt': len(novel_exercises), 'error_exercise_cnt': 0, 'total_word_cnt': 0, 'error_word_cnt': 0}
    for exercise in novel_exercises:
        try:
            gen_matches = grammar_checker.check(exercise)
        except Exception:
            exc_cnt += 1
            continue
        error_cnt = 0
        word_cnt = len(nltk.word_tokenize(exercise))
        for match in gen_matches:
            if match.category in ['PUNCTUATION']:
                continue
            if 'LOWERCASE' in match.ruleId or 'UPPERCASE' in match.ruleId:
                continue
            if match.category in ['CASING', 'TYPOS'] and match.replacements[0].lower() == exercise[match.offset: match.offset+match.errorLength]:
                continue  # ignore casing errors
            error_cnt += 1

        if error_cnt > 0:
            print('error exercise', exercise)
            result['error_exercise_cnt'] += 1

        result['error_word_cnt'] += error_cnt
        result['total_word_cnt'] += word_cnt

    print(result)
    return result


if __name__ == '__main__':
    pass


