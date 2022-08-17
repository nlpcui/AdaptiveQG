import sys, json
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
from rouge import Rouge

from nltk.translate.bleu_score import sentence_bleu

class QGEvaluator:
    def __init__(self, generated, reference):
        self.generated = generated
        self.reference = reference

        self.rouge = Rouge()
    
    def score(self):
        result = {}
        
        rouge_score = self.calc_rouge()
        result.update(rouge_score)

        return result

    def calc_rouge(self):
        scores = self.rouge.get_scores(generated, reference)

        rouge_1 = sum([score['rouge-1']['f'] for score in scores])/len(self.generated)
        rouge_2 = sum([score['rouge-2']['f'] for score in scores])/len(self.generated)
        rouge_l = sum([score['rouge-l']['f'] for score in scores])/len(self.generated)

        return {'rouge-1': rouge_1, 'rouge-2': rouge_2, 'rouge-l':rouge_l}

    def calc_bleu(self):
        pass

    def calc_em(self):
        pass

    def calc_coverage(self):
        pass


class KTEvaluator:
    def __init__(self, predicted, true):
        self.predicted = predicted
        self.true = true
    
    def calc_accuracy(self):
        pass

    def calc_f1_score(self):
        pass

    def calc_roc_score(self):
        pass


# reference = ['I am Peng']
# generated = ["I am Dang", 'He is Bing']
# score = sentence_bleu(reference, generated)
# print(score)