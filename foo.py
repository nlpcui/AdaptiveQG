import argparse, math
import torch
import random
import configparser
import sys
import logging
import time
from transformers import get_linear_schedule_with_warmup, BartTokenizer, BartForConditionalGeneration
from nltk import word_tokenize
import numpy as np
import matplotlib.pyplot as plt
from rouge import Rouge
from utils import sizeof
from sklearn import tree
from pprint import pprint
import pandas as pd
from tqdm import trange
import tqdm
import io
from tqdm import tqdm
from utils import *


def build_memory_update_mask(data):
    memory_update_mask = [
        [0 for i in range(len(data['word_ids']))] for j in range(len(data['word_ids']))]
    for column_id in range(len(data['sep_indices'])):
        if data['sep_indices'][column_id] == 0:
            continue

        memory_update_mask[column_id][column_id] = 1

        for row_id in range(column_id):
            if data['sep_indices'][row_id] == 1:
                memory_update_mask[row_id][column_id] = 1

    return memory_update_mask


class TqdmToLogger(io.StringIO):
    """
        Output stream for TQDM which will output to logger module instead of
        the StdOut.
    """
    logger = None
    level = None
    buf = ''

    def __init__(self, logger, level=None):
        super(TqdmToLogger, self).__init__()
        self.logger = logger
        self.level = level or logging.INFO

    def write(self, buf):
        self.buf = buf.strip('\t ')

    def flush(self):
        self.logger.log(self.level, self.buf)


if __name__ == '__main__':

    # tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    # model = AutoModelForSeq2SeqLM.from_pretrained('facebook/bart-large')

    # print(len(tokenizer))

    # tokenizer.add_special_tokens({'additional_special_tokens': ['<dif_{}>'.format(i) for i in range(4)]})
    # for attr in dir(tokenizer):
    #     print(attr)
    # print(list(zip(tokenizer.all_special_ids, tokenizer.all_special_tokens)))

    # decoded = tokenizer.decode([0, 100, 9, 26, 38, 779, 3, 2, 1, 1, 1], skip_special_tokens=True)
    # print(decoded)
    # a = tokenizer.additional_special_tokens[2] + ' ' + "he'll handle it"
    # input_ids = tokenizer(a)['input_ids']
    # print(input_ids)

    # tokenizer.decode([0, ])

    # model.resize_token_embeddings(len(tokenizer))
    # for name, parameters in model.named_parameters():
    #     print(name, parameters.size())

    # a = [1,2,3,4]
    # b = [5,1,7,8]

    # print(np.corrcoef(a, b)[0][1])

    # x = np.array(np.array([random.random() for i in range(900)]))
    # y = []
    # for i in range(900):
    #     t = -1 if random.random() > 0.5 else 1
    #     y_ = max(min(1, x[i]+t*random.random()*random.random()), 0)
    #     y.append(y_)

    # plt.plot((0,1), (0,1), ls='--')
    # plt.scatter(x, y)
    # plt.show()

    # rouger = Rouge()

    # a = ['I love you']
    # b = ['I love him']

    # score = rouger.get_scores(a, b)

    # print(score)

    # a = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    # index = torch.tensor([[3, 4, 7, 9], [1, 7]])

    # print(a[index])

    # a = np.array([
    #     [[0.2, 0.8], [0.9, 0.6], [0.9, 0.6]],
    #     [[0.4, 0.4], [0.5, 0.7], [0.1, 0.6]],
    #     [[0., 0.], [0., 0.], [0., 0.]]
    # ])

    # x = np.zeros([3, 2])
    # print(x)
    # positions = np.where((a==x).all(axis=0), True, False)

    # print(positions)

    # k = {
    #     'a': np.array([1,2,3]),
    #     'b': np.array([4,5,6])
    # }
    # np.savez('temp.npz', **k)
    # x = np.load('temp.npz')

    # print(type(x['a']))
    # print(type(x['b']))

    # from sklearn.datasets import make_regression
    # from sklearn.ensemble import GradientBoostingRegressor
    # from sklearn.model_selection import train_test_split
    # from IPython.display import Image
    # from pydotplus import graph_from_dot_data
    # from sklearn.tree import export_graphviz
    # import pydotplus
    # import pandas as pd

    # X, y = make_regression(random_state=0)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    # from sklearn.ensemble import GradientBoostingRegressor
    # gbdt = GradientBoostingRegressor(loss='absolute_error')
    # gbdt.fit(X_train, y_train)  # X_train: (n_samples, n_features), y_train: (n_samples, )
    # y_test_prediction = gbdt.predict(X_test)
    # feature_importances_ = np.argsort(-gbdt.feature_importances_)

    # from sklearn import tree
    # fp = open('visualization.txt', 'w')
    # for i in range(0,1):
    #     sub_tree = gbdt.estimators_[i, 0]
    #     tree_visualization = tree.export_text(sub_tree)
    #     fp.write(tree_visualization)
    # fp.close()

    # dot_data = export_graphviz(sub_tree, out_file=None, filled=True, rounded=True, special_characters=True, precision=2)
    # graph = pydotplus.graph_from_dot_data(dot_data)
    # Image(graph.create_png())

    # df = pd.DataFrame({'a':[1,2,3], 'b':[4,5,6]})
    # print(df)
    # for row_id, row in df.iterrows():
    #     row['a'] = row['a'] + 1

    # print(df.to_numpy())

    # pd.DataFrame({'id': x[:,0], 'price': prediction}).to_csv()

    # a = [1,2,3,4,5]

    # x = a[-3:]
    # print(x)

    # a = torch.tensor([[True,False], [False, True]])
    # b = (~a).int()

    # print(b)

    # a = np.array(.3)
    # b = np.array(.7)
    # print(a, b)
    # print(np.concatenate([a,b], axis=0))

    # a = np.array([1, 2, 3])
    # b = np.concatenate([a], axis=0)

    # print(b)

    # a = np.random.rand(350, 2053)
    # size = sys.getsizeof(a)/1024/1024
    # print(size)

    # a = np.array([[1,2,3], [4,5,6]])
    # b = []

    # b.extend(a)

    # print(b)

    # a = torch.tensor(np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]]))
    # b = a[1].detach().numpy()
    # b[0] = 100

    # print(a)
    # print(b)

    # a = torch.tensor([[0.7, 0.2], [0.3, 0.4]])

    # x = torch.nn.functional.pad(a, pad=(0,0,0,2))

    # print(x)

    # best_r = [-1]

    # for i in range(10):
    #     r = {i: [i]}
    #     if r[i][0] > best_r[0]:
    #         best_r = r[i]

    # print(best_r)

    # a = np.array([[False, True, True], [True, False, False], [False, True, True]])

    # a[-1,:] = False
    # print(a)

    # a = np.array([[1,2], [3, 4]])

    # print(a[0][0])

    # test_df = pd.DataFrame({'group':['student_1', 'student_1', 'student_2', 'student_2'],
    #                     'interaction_id': [1, 2, 3, 4]})

    # print(test_df)

    # a = np.array([3])
    # b = 3
    # print(a[0]==b)

    # a = np.array([[1,2,3], [4,5,6]])

    # indices = np.where(a==3, )

    # fp = open('test.log', 'w')

    # logging.basicConfig(level=logging.INFO, filename='test.log', filemode='w')
    # print('here')

    # # with logging_redirect_tqdm():
    # #     for i in trange(9):
    # #         pass

    # print(__name__)

    # logger = logging.getLogger()
    # logger.setLevel(logging.DEBUG)
    # handler = logging.FileHandler(filename='test.log', mode='w')
    # # formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', )
    # # handler.setFormatter(formatter)
    # logger.addHandler(handler)

    # logger.debug('test')

    # tqdm_out = TqdmToLogger(logger, level=logging.INFO)

    # pbar = tqdm(total=100, file=tqdm_out)
    # for i in range(100):
    #     time.sleep(1)
    #     pbar.update(1)

    # logits = torch.tensor([[.8, .3], [.6, .9], [.1, .2], [.4, .9]])
    # label = torch.tensor([1, 0, 0, -100])
    # ignore = label.eq(-100)
    # label[ignore] = 0
    # n_valid = ignore.eq(0).sum()
    # one_hot = torch.empty_like(logits)
    # one_hot = torch.fill(one_hot, 0)
    # one_hot = one_hot.scatter(1, label.unsqueeze(1), 1)
    # print('ignore', ignore)
    # print('n_valid', n_valid)
    # print('one hot', one_hot)

    # logs = torch.nn.functional.log_softmax(logits, dim=-1)
    # print(logs)

    # src = torch.rand(8, 1024, 2043, 2)
    # idx = torch.randint(0, 2043, (8, 1024))

    # c = torch.gather(src, dim=2, index=idx)
    # print(c.shape)

    # tensor_0 = torch.arange(3, 12).view(3, 3)
    # index = torch.tensor([[2, 1, 0]])
    # tensor_1 = tensor_0.gather(0, index)

    # a = torch.tensor([[1,2,3], [7, 8, 9]])
    # b = torch.tensor([[4,5,6], [10, 11, 3]])

    # c = torch.stack([a, b], dim=-1)

    # print(c)
    # x = [1, 2, 3, 4, 5]
    # y = [0.01, 0.02, 0.03, 0.04, 0.05]
    # corf = np.corrcoef(x, y)

    # print(corf)

    # from transformers import BartForConditionalGeneration
    # from transformers import AutoModel

    # print(AutoModel.from_pretrained("ainize/bart-base-cnn"))  # 官方给的代码
    # print("-"*100)

    # model_name = 'facebook/bart-base' # "ainize/bart-base-cnn"
    # model = BartForConditionalGeneration.from_pretrained(model_name)
    # print(dir(model))
    # exit(1)
    # print(model.parameters()['shared.weight'])
    # print('-'*100)
    # exit(1)
    # from transformers import RobertaForCausalLM, RobertaConfig
    # import torch
    # from transformers import RobertaForCausalLM, RobertaConfig
    # import torch

    # config = RobertaConfig.from_pretrained("roberta-base")
    # config.is_decoder = True
    # model = RobertaForCausalLM.from_pretrained('roberta-base', config=config)


    # exit(1)
    # for name, param in BartForConditionalGeneration.from_pretrained(model_name).named_parameters():
    #     print(name, param.shape)
    # parameters = model.parameters()
    # parameters['lm_head']

    # import torch
    # from transformers import BertTokenizer, BertForSequenceClassification
    
    # model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

    # for name, param in model.named_parameters():
    #     print(name, param.shape)
    
    # a = np.array([[1,1], [1,1]])
    # pad_width = ((1, 0), (1, 0))
    # a = np.pad(a, pad_width, constant_values=((3, 0), (4, 0)))
    # print(a)

    # a = [1,1,2,3]
    # print(a.count(1))
    # a = np.array([1,2,3,4,5,3])
    # b = np.where(a==3, True, False).sum()
    # print(b==2)
    # a = torch.tensor([[1,2,3],[5,6,7]])
    # b = torch.tensor([[2,9]])
    # c = torch.cat([a, b], dim=-1)

    # print(c)

    # logits = torch.randn(3, 4)
    # labels = torch.where(torch.randn(3, 4)>0, 1, 0).float()
    # print(labels)
    # loss_function_1 = torch.nn.BCEWithLogitsLoss(reduction='none')
    # loss_function_2 = torch.nn.functional.binary_cross_entropy_with_logits

    # loss_1 = loss_function_1(input=logits, target=labels) * labels
    # print(loss_1)
    # loss_1 = loss_1.sum()/labels.sum()
    # print(type(loss_1))
    # print('-'*100)
    # loss_2 = loss_function_2(input=logits, target=labels, weight=labels, reduction='none') 
    # print(loss_2)
    # print(torch.mean(loss_2))
    # print('-'*100)
    # loss_3 = loss_function_2(input=logits, target=labels, weight=labels)
    # print(loss_3)

    # a = torch.tensor([
    #     [1,2,3,4,5], 
    #     [6,7,8,9,10]
    # ])

    # b = torch.roll(a, shifts=-2, dims=1)
    # print(b)

    # a = np.array([1,2,3])
    # b = np.where(a>1, 1, 0)

    # print(b)

    # logits = torch.tensor([0.78, 2.14, -2.3, 1.5])
    # labels = torch.tensor([1., 0., 1., 0.])
    # pos_weight = torch.tensor([1.5, 1., 1.5, 1.])
    # weight = torch.tensor([1, 1.5, 1, 0])

    # criteria = torch.nn.BCEWithLogitsLoss(reduction='none')
    # loss = criteria(logits, labels)
    # loss_1 = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, weight=weight, reduction='none')
    # loss_2 = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, weight=weight, reduction='none').sum()
    # loss_3 = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, weight=weight, reduction='none').sum() / 3.5
    # loss_4 = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, weight=weight, reduction='mean')
    # print(loss_1, loss_2, loss_3, loss_4)

    # pos_weight = 1.5
    # labels = torch.tensor([1, 0, 1, 0, -100])
    # weight = torch.where(labels==1, pos_weight, labels)
    # print(weight)
    # weight = torch.where(labels==0, 1, weight)
    # print(weight)
    # weight = torch.where(labels==-100, 0, weight)
    # print(weight)

    # for i in range(1, 20):
    #     print(round(i*0.05,2), -math.log(i*0.05))

    # a = torch.tensor([0.1, 0.1, 0.2, 0.002])
    # b = torch.where(a==0.1, 1, 0)
    # print(b)

    # tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    # test_sent = "he likes me"
    # s = tokenizer(test_sent)['input_ids']
    # print(test_sent, s)
    # print(len(tokenizer))
    # for id_ in s:
    #     x = tokenizer.decode(id_)
    #     print(id_, '${}$'.format(x))
    # print(tokenizer.bos_token_id)
    # for subword in tokenizer.vocab:
    #     print(subword)

    model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    test_sentence = 'UN Chief Says There Is No <mask> in Syria'
    input_ids = tokenizer(test_sentence, return_tensors='pt')['input_ids']
    

    decoded = torch.tensor([[tokenizer.eos_token_id]])

    decoder_input_ids = torch.tensor([[tokenizer.eos_token_id]])
    past_key_values = None
    time0 = time.time()
    for i in range(10): 
        output = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids, past_key_values=past_key_values, use_cache=True)
        last_logits = output.logits[:,[-1],:]
        next_word_id = torch.argmax(last_logits, dim=-1)
        # decoder_input_ids = torch.cat((decoder_input_ids, next_word_id), dim=-1)
        decoder_input_ids = next_word_id
        decoded = torch.cat((decoded, next_word_id), dim=-1)
        past_key_values = output.past_key_values
        # print(i)
        # print(next_word_id, tokenizer.batch_decode(next_word_id))
        # print(decoder_input_ids)
    
    time1 = time.time()
    print(time1-time0)
    print(tokenizer.batch_decode(decoded))
    # print(output.logits.shape) # batch_size, seq_len, vocab_size
    # print(type(output))

    '''
    Input: Model, Beam_size, Vocab_difficulty V, target_difficulty, Input, number of groups K, difficulty function D 
    Output: QualifiedHyp = []

    HypGroups = [[InitHyp]]      # InitHyp = "<bos>"

    for t=1, t++, t<maxLen do
        extensions = []
        for group in HypGroups:
            for hpypothesis in group:
                extensions = extensions U model.generate(input, hpypothesis))  # |V| new extensions
        
        for ExtHyp in extensions:
            ExtHyp.difficulty = D(ExtHpy)           # difficulty
            ExtHpy.score = model.score(ExtHpy)      # likelihood

        HypGroups <- GroupHpyByDifficulty(extensions, K)
        GroupSize[1,...,K] <- AllocateGroupSizeByScore(HypoGroups, beam_size)

        for i=0, i++, i<K:
            HypGroups[i] <- GroupSize[i]-argmax(score)(HypoGroup[i])
        
        QualifiedHyp = QualifiedHyp U IsFinished(HypGroups[-1]) # has <eos> tokens

    return argmax(hpy.score+a*|hyp.difficulty-v|) QualifiedHyp

    Function GroupzHpyByDifficulty(Hpytheses, GroupNumber)

    '''