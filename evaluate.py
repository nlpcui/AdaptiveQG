import sys, json, spacy
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
from rouge import Rouge

from nltk.translate.bleu_score import sentence_bleu


class QGEvaluator:
    def __init__(self, prompt_words, generated, reference):
        self.prompt_words = prompt_words
        self.generated = generated
        self.reference = reference

        self.rouge = Rouge()
        self.tokenizer = spacy.load('en_core_web_sm').tokenizer

    def compute_metrics(self):
        result = {}
        
        rouge_score = self.compute_rouge()
        result.update(rouge_score)

        return result

    def compute_rouge(self):
        scores = self.rouge.get_scores(self.generated, self.reference)

        rouge_1 = sum([score['rouge-1']['f'] for score in scores])/len(self.generated)
        rouge_2 = sum([score['rouge-2']['f'] for score in scores])/len(self.generated)
        rouge_l = sum([score['rouge-l']['f'] for score in scores])/len(self.generated)

        return {'rouge-1': rouge_1, 'rouge-2': rouge_2, 'rouge-l':rouge_l}


    def compute_coverage(self):
        cover_cnt = 0
        total_cnt = 0
        for idx in range(len(self.prompt_words)):
            prompt_words = self.tokenizer(self.prompt_words[idx])
            generated_words = self.tokenizer(self.generated[idx])
            total += len(prompt_words)
            for word in prompt_words:
                if word in generated_words:
                    cover_cnt += 1
        
        return {'coverage': cover_cnt/total_cnt}



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

    
