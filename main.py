import sys, torch, os, json, configparser, argparse, datetime, logging
from tqdm import tqdm
from process_data import DuolingoDatasetBuilder, DuolingoDataset, DuolingoKTDataset
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
        filename=args.duolingo_en_es_train_log, 
        level=logging.INFO, 
        filemode='w'
    )

    # if args.run == 'train':
    #     train(args)
    # elif args.run == 'test':
    #     test(args)
    # elif args.run == 'val':
    #     val(args)

    DuolingoKTDataset.get_format_datat(
        train_raw=args.duolingo_en_es_train_raw, 
        dev_raw=args.duolingo_en_es_dev_raw, 
        dev_key_raw=args.duolingo_en_es_dev_key_raw, 
        test_raw=args.duolingo_en_es_test_raw, 
        test_key_raw=args.duolingo_en_es_test_key_raw, 
        format_output=args.duolingo_en_es_format, 
        vocab_file=args.duolingo_en_es_vocab, 
        word_file=args.duolingo_en_es_words, 
        exercise_file=args.duolingo_en_es_exercises
    )

    # duolingo_kt_dataset = DuolingoKTDataset(
    #     data_file=args.duolingo_en_es_format, 
    #     vocab_file=args.duolingo_en_es_vocab, 
    #     word_file=args.duolingo_en_es_words, 
    #     exercise_file=args.duolingo_en_es_exercises,
    #     max_len=args.max_len
    # )

    # for split in [None, 0, 1, 2]:
    #     print('='*100)
    #     stat = duolingo_kt_dataset.get_statistics(split=split)
    #     for key in stat:
    #         print(key, stat[key])


    