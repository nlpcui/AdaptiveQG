import sys, torch, os, json, configparser, argparse, datetime
from tqdm import tqdm
from process_data import DuolingoDatasetBuilder, DuolingoDataset, DuolingoTokenizer
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from models import SAKT
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup
from pprint import pprint
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device == 'cuda':
    print('using gpu {}'.format(torch.cuda.get_device_name(0)))
else:
    print('using cpu')

assistment_dir = '/cluster/work/sachan/pencui/ASSISTment'
duolingo_raw_en_es_dir = "/cluster/work/sachan/pencui/duolingo_2018_shared_task/data_en_es" # '/Users/cuipeng/Documents/Datasets/duolingo_2018_shared_task/data_en_es'
duolingo_processed_en_es_dir = "/cluster/project/sachan/pencui/duolingo/data_en_es/"
duolingo_en_es_models_dir = '/cluster/work/sachan/pencui/duolingo_2018_shared_task/data_en_es/models'

file_paths = {
    "Assistment_2004_2005": 'ASSISTment_2004_2005/ds92_tx_All_Data_172_2016_0504_081852.txt',
    "Assistment_2005_2006": 'ASSISTment_2005_2006/ds120_tx_All_Data_265_2017_0414_065125.txt',
    "Assistment_2006_2007": 'ASSISTment_2006_2007/ds339_tx_All_Data_1059_2015_0729_215742.txt',
    "Duolingo_en_es_train": "en_es.slam.20190204.train",
    "Duolingo_en_es_test": "en_es.slam.20190204.test",
    "Duolingo_en_es_dev": "en_es.slam.20190204.dev",
    "Duolingo_en_es_test_key": "en_es.slam.20190204.test.key",
    "Duolingo_en_es_dev_key": "en_es.slam.20190204.dev.key",
    "Duolingo_en_es_train_processed": "train_processed_1024",
    "vocab": "vocab.txt"
}


def cal_metrics(logits, y_labels):
    # cacuclate ROC and F1 score
    # logits: [example_num, label_num]
    # labels: [example_num, ]

    # metrics
    valid_positions = y_labels.ge(0)

    y_pos_probs = nn.functional.softmax(logits, dim=-1)[:,1]
    y_pred = torch.argmax(logits, dim=-1)

    y_labels_selected = torch.masked_select(y_labels, valid_positions).numpy()
    y_pos_probs_selected = torch.masked_select(y_pos_probs, valid_positions).numpy()
    y_pred_labels_selected = torch.masked_select(y_pred, valid_positions).numpy()

    auc = roc_auc_score(y_true=y_labels_selected, y_score=y_pos_probs_selected)
    f1 = f1_score(y_true=y_labels_selected, y_pred=y_pred_labels_selected)
    precision = precision_score(y_true=y_labels_selected, y_pred=y_pred_labels_selected)
    recall = recall_score(y_true=y_labels_selected, y_pred=y_pred_labels_selected)
    accuracy = accuracy_score(y_true=y_labels_selected, y_pred=y_pred_labels_selected)

    return round(auc, 4), round(precision, 4), round(recall, 4), round(f1, 4), round(accuracy, 4)


def eval(args):
    pass


def train(args):
    # dataset_builder = DuolingoDatasetBuilder(
    #     filepath=os.path.join(args.data_dir, args.data_file), 
    #     max_len=args.max_len,
    #     vocab_save_path=args.vocab_save_path
    # )
    # dataset_builder.save_by_stu(os.path.join(duolingo_en_es_dir, file_paths['Duolingo_en_es_train_processed']))
    # exit(1)
    dataset = DuolingoDataset(os.path.join(duolingo_processed_en_es_dir, file_paths['Duolingo_en_es_train_processed']), shuffle=True)
    tokenizer = DuolingoTokenizer(args.vocab_save_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, drop_last=True)

    model = SAKT(
        num_emb=len(tokenizer), 
        dim_emb=args.dim_emb, 
        max_len=args.max_len, 
        dim_pe=args.dim_pe, 
        num_exercise_encoder_layers=args.num_exercise_encoder_layers, 
        dim_attn_exercise=args.dim_attn_exercise, 
        num_attn_heads_exercise=args.num_attn_heads_exercise,
        num_interaction_encoder_layers=args.num_interaction_encoder_layers,
        dim_attn_interaction=args.dim_attn_interaction,
        num_attn_heads_interaction=args.num_attn_heads_interaction, 
        dim_ff_exercise=args.dim_ff_exercise, 
        dropout_exercise=args.dropout_exercise, 
        dim_ff_interaction=args.dim_ff_interaction, 
        dropout_interaction=args.dropout_interaction, 
        num_label=args.num_label
    ).to(device)
    
    model.train()

    total_steps = len(dataset) // args.batch_size * args.epoch

    loss_function = nn.CrossEntropyLoss(ignore_index=args.ignore_label)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.warmup_rate*total_steps,
        num_training_steps=total_steps
    )

    for epoch in range(args.epoch):
        for batch_id, (x_exercise, x_interaction, x_exercise_attn_mask, x_interaction_attn_mask, y_labels) in enumerate(dataloader):
            # print(x_exercise)
            # print(tokenizer.batch_detokenize(x_exercise))
            # print(x_interaction)
            # print(tokenizer.batch_detokenize(x_interaction))
            # print('x_exercise_attn_mask')
            # for i, row in enumerate(x_exercise_attn_mask[0]):
            #     # print(row.int().numpy().tolist())
            #     # print(min(row))
            #     if min(row.int().numpy().tolist()) == 1:
            #         print(i)
            # print('x_interaction_attn_mask')
            # for i, row in enumerate(x_interaction_attn_mask[0]):
            #     # print(row.int().numpy().tolist())
            #     # print(min(row.int().numpy().tolist()))
            #     if min(row.int().numpy().tolist()) == 1:
            #         print(i)
            # print(y_labels)
            # exit(1)
            optimizer.zero_grad()
            x_exercise_attn_mask = x_exercise_attn_mask.unsqueeze(1).repeat(1, args.num_attn_heads_exercise, 1, 1).view(args.batch_size*args.num_attn_heads_exercise, args.max_len, -1)
            x_interaction_attn_mask = x_interaction_attn_mask.unsqueeze(1).repeat(1, args.num_attn_heads_interaction, 1, 1).view(args.batch_size*args.num_attn_heads_interaction, args.max_len, -1)

            logits = model(
                x_exercise=x_exercise.to(device),
                x_interaction=x_interaction.to(device),
                x_exercise_attn_mask=x_exercise_attn_mask.to(device),
                x_interaction_attn_mask=x_interaction_attn_mask.to(device),
                y_labels=y_labels.to(device)
            )

            # optimize
            logits = logits.view(args.batch_size*args.max_len, -1)
            y_labels = y_labels.view(args.batch_size*args.max_len)
            loss = loss_function(logits, y_labels.to(device))
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            
            # metrics
            auc, precision, recall, f1_score, accuracy = cal_metrics(logits.cpu().detach(), y_labels.cpu().detach())

            print('In [{}/{}] epoch, [{}/{}] batch, loss:{}, auc:{}, precision:{}, recall:{}, f1_score:{}, accuracy:{}, --{}'.format(epoch, args.epoch, batch_id, len(dataset)//args.batch_size, loss, auc, precision, recall, f1_score, accuracy, datetime.datetime.now()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model parameters
    parser.add_argument('--num_emb', type=int)
    parser.add_argument('--dim_emb', type=int, default=256)
    parser.add_argument('--max_len', type=int, default=1024)
    parser.add_argument('--dim_pe', type=int, default=256)

    parser.add_argument('--dim_attn_exercise', type=int, default=256)
    parser.add_argument('--num_attn_heads_exercise', type=int, default=4)
    parser.add_argument('--dim_ff_exercise', type=int, default=256)
    parser.add_argument('--dropout_exercise', type=float, default=0.1)
    parser.add_argument('--num_exercise_encoder_layers', type=int, default=3)

    parser.add_argument('--dim_attn_interaction', type=int, default=256)
    parser.add_argument('--num_attn_heads_interaction', type=int, default=4)
    parser.add_argument('--dim_ff_interaction', type=int, default=256)
    parser.add_argument('--dropout_interaction', type=float, default=0.1)
    parser.add_argument('--num_interaction_encoder_layers', type=int, default=3)
    parser.add_argument('--num_label', type=int, default=2)

    # training setup
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--run', type=str, default='train') # train, val, test
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--warmup_rate', type=float, default=0.03)
    parser.add_argument('--ignore_label', type=int, default=-1)
    
    # paths
    parser.add_argument('--raw_data_dir', type=str, default=duolingo_raw_en_es_dir)
    parser.add_argument('--processed_data_dir', type=str, default=duolingo_processed_en_es_dir)
    parser.add_argument('--data_file', type=str, default=file_paths['Duolingo_en_es_train'])
    parser.add_argument('--vocab_save_path', type=str, default=os.path.join(duolingo_processed_en_es_dir, file_paths['vocab']))
    parser.add_argument('--model_save_dir', type=str, default=duolingo_en_es_models_dir)
    
    
    args = parser.parse_args()

    if args.run == 'train':
        train(args)
    elif args.run == 'test':
        test(args)
    elif args.run == 'val':
        val(args)

    # a = torch.tensor([1, 0, 1])
    # b = torch.tensor([0, 0, -1])
    # correct_cnt = torch.sum(torch.eq(b-a, 0))
    # valid_cnt = a.size(0) - torch.eq(b, -1).sum()
    # print(correct_cnt)
    # print(valid_cnt)

    # x = [i%2 for i in range(2000000)]
    # print(sys.getsizeof(x))