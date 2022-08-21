import sys, os, json, configparser, argparse, datetime, logging, math
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,5'
import torch
from tqdm import tqdm
from process_data import *
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from models import KnowledgeTracer, QuestionGenerator
import torch.nn as nn
import torch.nn.functional as F
from transformers import get_linear_schedule_with_warmup, BartTokenizer
from pprint import pprint
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from evaluate import QGEvaluator, KTEvaluator
from copy import deepcopy

#sys.path.remove('/cluster/apps/nss/gcc-6.3.0/python_gpu/3.8.5') # euler




def train_kt(args):
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



def train_qg(args, gpu_cnt, global_rank, local_rank, device, model_name='bart_qg'):
    question_generator = QuestionGenerator(args.qg_model_name).to(device)
    if gpu_cnt > 1:
        question_generator = DDP(question_generator, device_ids=[local_rank]).module

    bart_tokenizer = BartTokenizer.from_pretrained(args.qg_model_name)
    
    data_collator = QGDataCollator(
        model=question_generator.generator,
        tokenizer=bart_tokenizer,
        x_max_length=int(args.qg_x_max_length),
        y_max_length=int(args.qg_y_max_length),
        truncation=True,
        padding='max_length',
        label_pad_token_id=int(args.qg_label_pad_token_id),
        return_tensors='pt'
    )

    logging.info('-- global_rank: {}, local_rank: {}, loading train datasets'.format(global_rank, local_rank))
    train_dataset = DuolingoGenDataset(
        split='train', 
        data_file=args.duolingo_en_es_format, 
        sample_rate=args.qg_keyword_sample_rate,
        word_file=args.duolingo_en_es_words
    )

    if global_rank == local_rank == 0: 
        for index in random.sample(range(len(train_dataset)), 3):
            logging.info('-- {} train data in total, sampled {}th example: {}.'.format(len(train_dataset), index, train_dataset[index]))

    if gpu_cnt > 1:
        train_sampler = DistributedSampler(train_dataset)
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=int(args.qg_train_batch_size), 
            collate_fn=data_collator,
            sampler=train_sampler
    	)
    else:
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=int(args.qg_train_batch_size),
            collate_fn=data_collator,
            shuffle=True
        )
	
    # eval
    logging.info('-- global_rank: {}, local_rank: {}, loading eval dataset'.format(global_rank, local_rank))
    eval_dataset = DuolingoGenDataset(
        split='dev', 
        data_file=args.duolingo_en_es_format,
        sample_rate=args.qg_keyword_sample_rate,
        word_file=args.duolingo_en_es_words
    )

    if global_rank == local_rank == 0:
        for index in random.sample(range(len(eval_dataset)), 3):
            logging.info('-- {} eval data in total, sampled {}th example: {}'.format(len(eval_dataset), index, eval_dataset[index]))

    if gpu_cnt > 1:
        eval_sampler = DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=int(args.qg_eval_batch_size),
            collate_fn=data_collator,
            sampler=eval_sampler
        )
    else:
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=int(args.qg_eval_batch_size),
            shuffle=True,
            collate_fn=data_collator
        ) 

    # Train!
    logging.info('global rank {}, local_rank {}, start training'.format(global_rank, local_rank))
    batch_steps = math.ceil(len(train_dataset)/int(args.qg_train_batch_size)/max(gpu_cnt, 1))
    total_steps = int(args.qg_num_train_epoch) * batch_steps
    optimizer = AdamW(question_generator.parameters(), lr=float(args.qg_learning_rate))

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=float(args.qg_warmup_rate)*total_steps,
        num_training_steps=total_steps
    )

    save_info = {
        'epoch': 0,
        'loss': 0,
        'best_validation_performance': 0,
        'model_state_dict': None,
        'optimizer_state_dict': None
    }

    for epoch_id in range(int(args.qg_num_train_epoch)):
        question_generator.train()
        epoch_loss = 0

        if gpu_cnt > 1:
            train_dataloader.sampler.set_epoch(epoch_id)

        for batch_id, (uids, y_difficulties, x_keyword_ids, x_attention_mask, y_exercise_labels, decoder_input_ids) in enumerate(train_dataloader):
            optimizer.zero_grad()

            if batch_id > 100:
                break
            # debug
            if epoch_id == 0 and batch_id == 0:
                logging.debug('example collated batch:\n user_ids: {},\nx_keyword_ids: {},\nx_attention_mask: {},\ny_exercise_labels: {},\n decoder_input_ids: {},\ny_difficulties: {}'.format(
                    uids, x_keyword_ids, x_attention_mask, y_exercise_labels, decoder_input_ids, y_difficulties
                ))

            # do train
            outputs = question_generator(
                x_keyword_ids=x_keyword_ids.to(device),
                x_attention_mask=x_attention_mask.to(device),
                x_knowledge_state=None, 
                y_difficulties=y_difficulties, 
                y_exercise_ids=y_exercise_labels.to(device),
                decoder_input_ids=decoder_input_ids.to(device)
            )
            # print(type(outputs))
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            logging.info('global rank: {}, local_rank: {}, {}/{} epoch, {}/{} batch, loss is {}'.format(
                global_rank, local_rank, epoch_id, args.qg_num_train_epoch, batch_id, batch_steps, loss 
            ))

            epoch_loss += loss

        # do eval
        eval_batch_steps = math.ceil(len(eval_dataset) / args.qg_eval_batch_size / gpu_cnt)
        question_generator.eval()
        qg_evaluator = QGEvaluator(generated=[], reference=[], prompt_words=[])
        with torch.no_grad():
            if gpu_cnt > 1:
                eval_dataloader.sampler.set_epoch(epoch_id)
            
            prompt_words = [] 
            generated = []
            reference = []
            for batch_id, (uids, y_difficulties, x_keyword_ids, x_attention_mask, y_exercise_labels, decoder_input_ids) in enumerate(eval_dataloader):
                if batch_id > 10:
                    break
                logging.info('-- global rank {}, local_rank {}, evaluating {}/{} batch, {} batch data_collected'.format(global_rank, local_rank, batch_id, eval_batch_steps, len(prompt_words)))
                output_ids = question_generator.generator.generate(
                    inputs=x_keyword_ids.to(device),
                    attention_mask=x_attention_mask.to(device),
                    num_beams=int(args.qg_num_beams),
                    max_length=int(args.qg_y_max_length),
                    # return_dict_in_generate=True # transformers.generation_utils.BeamSearchEncoderDecoderOutput
                ) # [batch_size, seq_len]
                
                # postprocess
                ## pad output_ids to collect across devices
                pad_length = int(args.qg_y_max_length) - output_ids.size(-1)
                if pad_length > 0:
                    output_ids = F.pad(output_ids, pad=(0, pad_length), value=bart_tokenizer.pad_token_id) 
                ## recover pad labels
                y_exercise_labels[y_exercise_labels==int(args.qg_label_pad_token_id)] = bart_tokenizer.pad_token_id 
                # end postprocess
                
                if gpu_cnt > 1:
                    batch_prompt_words = [torch.zeros_like(x_keyword_ids).to(device) for i in range(gpu_cnt)]
                    batch_reference = [torch.zeros_like(y_exercise_labels).to(device) for i in range(gpu_cnt)]
                    batch_generated = [torch.zeros_like(output_ids).to(device) for i in range(gpu_cnt)]
                    
                    dist.all_gather(batch_prompt_words, x_keyword_ids.to(device)) # [batch_size, input_len] * gpu_cnt
                    dist.all_gather(batch_reference, y_exercise_labels.to(device))
                    dist.all_gather(batch_generated, output_ids)
                    
                    # logging.info('-- rank: {}, gathered prompt {}'.format(local_rank, batch_prompt_words))
                    # logging.info('-- rank: {}, gathered generated {}'.format(local_rank, batch_generated))
                    # logging.info('-- rank: {}, gathered reference {}'.format(local_rank, batch_reference))
                    
                    prompt_words.extend(batch_prompt_words)
                    reference.extend(batch_reference)
                    generated.extend(batch_generated)
                else:
                    prompt_words.append(x_keywords_ids.to(device))
                    generated.append(output_ids)
                    reference.append(y_exercise_labels.to(device))
            
            logging.info(' -- global rank {}, local rank {}, complete evalution'.format(global_rank, local_rank))
        
        ## compute metrics
        if global_rank == local_rank == 0:
            prompt_words = torch.cat(prompt_words, dim=0).cpu()
            generated = torch.cat(generated, dim=0).cpu()
            reference = torch.cat(reference, dim=0).cpu()
            logging.info('-- collect {} data for evaluation'.format(prompt_words.size(0)))
            logging.info('-- shape prompt: {}, generated: {}, reference: {}'.format(prompt_words.shape, generated.shape, reference.shape))
            qg_evaluator.prompt_words = bart_tokenizer.batch_decode(prompt_words, skip_special_token=True) 
            qg_evaluator.generated  = bart_tokenizer.batch_decode(generated, skip_special_token=True)
            qg_evaluator.reference = bart_tokenizer.batch_decode(reference, skip_special_token=True)
            logging.info('-- rank {} computing metrics ...'.format(local_rank))
            score = qg_evaluator.compute_metrics()
            logging.info('{}/{} epoch, performance on eval set: {}'.format({
                epoch_id, int(args.qg_num_train_epoch), score
            }))

            validation_performance = (score['rouge-1'] + score['rouge-2'] + score['rouge-l'])/3

            logging.info('-- {}th epoch, loss is {}, global_rank: {}, local_rank:{}, validation performance is {}'.format(epoch_id, epoch_loss, global_rank, local_rank, validation_performance))
            if validation_performance > save_info['best_validation_performance']:
                save_info['epoch'] = epoch_id
                save_info['loss'] = epoch_loss
                save_info['best_validation_performance'] = validation_performance
                save_info['model_state_dict'] = deepcopy(question_generator.generator().state_dict())
                save_info['optimizer_state_dict'] = deepcopy(optimizer.state_dict())
    
            logging.info('-- global_rank:{}, local_rank:{}, finish training, best model: {}-th epoch, loss: {}, validation_performance: {}'.format(global_rank, local_rank, save_info['epoch'], save_info['loss'], save_info['best_validation_performance']))
    # save best performing model    
    if global_rank == local_rank == 0:
        save_path = os.path.join(args.model_save_dir, '{}_{}ep.pth'.format(model_name, save_info['epoch'])) 
        logging.info('saving model to {}'.format(save_path))
        torch.save(save_info, path=save_path)



def evaluate():
    pass



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='euler_conf.ini')
    
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
        filemode='a'
    )
	
    logging.info('cuda visible devices {}'.format(os.environ["CUDA_VISIBLE_DEVICES"]))
	


    # if args.run == 'train':
    #     train(args)
    # elif args.run == 'test':
    #     test(args)
    # elif args.run == 'val':
    #     val(args)

    # format raw data
    # DuolingoKTDataset.get_format_datat(
    #     train_raw=args.duolingo_en_es_train_raw, 
    #     dev_raw=args.duolingo_en_es_dev_raw, 
    #     dev_key_raw=args.duolingo_en_es_dev_key_raw, 
    #     test_raw=args.duolingo_en_es_test_raw, 
    #     test_key_raw=args.duolingo_en_es_test_key_raw, 
    #     format_output=args.duolingo_en_es_format, 
    #     vocab_file=args.duolingo_en_es_vocab, 
    #     word_file=args.duolingo_en_es_words, 
    #     exercise_file=args.duolingo_en_es_exercises
    # )
    

    gpu_cnt = torch.cuda.device_count() # gpu_cnt_per_machine
    global_rank = 0
    local_rank = 0

    if gpu_cnt == 0:
        logging.info('-- using cpu')
        device = torch.device('cpu')
    elif gpu_cnt == 1:
        logging.info('-- using single gpu: {}'.format(torch.cuda.get_device_name(0)))
        device = torch.device('cuda:0')
    else:
        local_rank = int(os.environ['LOCAL_RANK'])
        device = torch.device('cuda:{}'.format(local_rank))
        torch.cuda.set_device(local_rank)
        dist.init_process_group('nccl', init_method='env://', world_size=gpu_cnt, rank=local_rank)
        global_rank = dist.get_rank() # process_id
        
        args.qg_train_batch_size = int(args.qg_train_batch_size) // gpu_cnt
        args.qg_eval_batch_size = int(args.qg_eval_batch_size) // gpu_cnt
        if local_rank == global_rank == 0:
            logging.info('-- global_rank {}, local_rank {}, available gpus: {}'.format(global_rank, local_rank, [torch.cuda.get_device_name(i) for i in range(gpu_cnt)]))
            logging.info('-- global_rank {}, local_rank {}, total_train_batch_size is {}, per_device_train_batch_size is {}'.format(global_rank, local_rank, args.qg_train_batch_size*gpu_cnt, args.qg_train_batch_size)) 
    
    train_qg(args, gpu_cnt=gpu_cnt, global_rank=global_rank, local_rank=local_rank, device=device)
