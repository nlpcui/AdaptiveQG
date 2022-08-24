import sys, json, spacy, logging
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
from rouge import Rouge
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
import numpy as np
from nltk import word_tokenize

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
    def __init__(self, logits, labels):
        self.logits = logits
        self.labels = labels
    
    def compute_metrics(logits, y_labels):
        # cacuclate ROC and F1 score
        # logits: [example_num, label_num]
        # labels: [example_num, ]

        valid_positions = self.labels.ge(0)
        
        # binary score
        pred_labels = torch.argmax(self.logits, dim=-1)
        pred_labels_selected = torch.masked_select(pred_labels, valid_positions).numpy()
        labels_selected = torch.masked_select(self.labels, valid_positions).numpy()
        
        f1 = f1_score(y_true=labels_selected, y_pred=pred_labels_selected)
        precision = precision_score(y_true=labels_selected, y_pred=pred_labels_selected)
        recall = recall_score(y_true=labels_selected, y_pred=pred_labels_selected)
        accuracy = accuracy_score(y_true=labels_selected, y_pred=pred_labels_selected)
        
        pos_probs = nn.functional.softmax(self.logits, dim=-1)[:,1]
        pos_probs_selected = torch.masked_select(pos_probs, valid_positions).numpy()

        auc = roc_auc_score(y_true=labels_selected, y_score=pos_probs_selected)

        return {'auc': round(auc, 4), 'precision': round(precision, 4), 'recall': round(recall, 4), 'f1': round(f1, 4), 'acc': round(accuracy, 4)}

    
