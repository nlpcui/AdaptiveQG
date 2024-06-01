import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'  # 0, 7
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from models import ExerciseGeneratorD, ExerciseGeneratorC, DKT, ConstrainedDecodingWithLookahead
import torch.nn as nn
import torch.nn.functional as F
from transformers import get_linear_schedule_with_warmup
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from evaluate import QGEvaluator, KTEvaluator, PersonalizedQGEvaluator
from copy import deepcopy
from data import *
import numpy as np


class JointKTQGTrainer:
    def __init__(self, train_args, device, gpu_cnt, local_rank, use_ability, use_difficulty, d_template, max_difficulty_label, difficulty_bucket, d_type, d_source):
        self.args = train_args
        self.device = device
        self.gpu_cnt = gpu_cnt
        self.local_rank = local_rank
        self.word_sampler = WordSampler()

        assert d_type in ['continuous', 'discrete']
        assert d_source in ['kt', 'gd']
        self.use_ability = use_ability
        self.use_difficulty = use_difficulty
        self.d_template = d_template
        self.max_difficulty_label = max_difficulty_label
        self.difficulty_bucket = difficulty_bucket
        self.d_type = d_type
        self.d_source = d_source

        # prepare data
        self.user_map = get_user_map(self.args.duolingo_en_es_user_file)
        self.word_map = get_word_map(self.args.duolingo_en_es_word_file)
        self.w_l_tuple_map = get_w_l_tuple_map(self.args.duolingo_en_es_w_l_tuple_file)
        self.pos_tag_map = get_pos_tag_map(self.args.duolingo_en_es_pos_tag_file)
        self.word_difficulty_map = get_vocab_difficulty(self.args.duolingo_en_es_word_file)

        if local_rank == 0:
            logging.info('Loading data {} users, {} words, {} tuples, {} pos_tags.'.format(len(self.user_map), len(self.word_map), len(self.w_l_tuple_map), len(self.pos_tag_map)))
            logging.info('initializing models ...')

        # initialize model
        self.knowledge_tracer = DKT(
            input_size=self.args.dkt_input_size,
            hidden_size=self.args.dkt_input_size,
            num_layers=self.args.dkt_num_layers,
            num_tuples=len(self.w_l_tuple_map),
            num_users=len(self.user_map),
            encoder=self.args.dkt_encoder,
            num_attn_heads=4,
            dropout=0.1,
            device=device,
            max_length=self.args.kt_max_seq_len,
            num_words=len(self.word_map),
        ).to(device)

        self.qg_tokenizer = BartTokenizer.from_pretrained(self.args.qg_model_name)
        if self.use_difficulty and self.d_type == 'discrete':
            difficulty_control_tokens = {'additional_special_tokens': [self.d_template.format(i) for i in range(self.max_difficulty_label)]}
            self.qg_tokenizer.add_special_tokens(difficulty_control_tokens)

        if self.d_type == 'continuous':
            self.question_generator = ExerciseGeneratorC(model_name=self.args.qg_model_name, num_words=len(self.word_map), use_d=self.use_difficulty, use_a=self.use_ability).to(self.device)
        else:
            self.question_generator = ExerciseGeneratorD(
                model_name=self.args.qg_model_name,
                use_difficulty=self.use_difficulty,
                max_difficulty_label=self.max_difficulty_label,
                d_template=self.d_template
            ).to(self.device)
        # self.question_generator = BartForConditionalGeneration.from_pretrained(self.args.qg_model_name).to(self.device)

        self.prepare_decoder_input_ids_from_labels = self.question_generator.generator.prepare_decoder_input_ids_from_labels

        # decoding algorithm
        # self.searching_algo = ConstrainedDecodingWithLookahead(
        #     beam_size=10,
        #     word_map=self.word_map,
        #     tokenizer=self.qg_tokenizer,
        #     model=self.question_generator.generator,
        #     lookahead=int(self.args.qg_search_lookahead),
        #     factor_c=float(self.args.qg_search_factor_c),
        #     factor_d=float(self.args.qg_search_factor_d),
        #     factor_s=float(self.args.qg_search_factor_s),
        # )

        self.kt_evaluator = KTEvaluator(len(self.word_map), vocabulary_difficulty=self.word_difficulty_map)
        # self.qg_evaluator = QGEvaluator(vocab_difficulty=self.word_difficulty_map)
        self.pqg_evaluator = PersonalizedQGEvaluator(word_map=self.word_map, difficulty_map=self.word_difficulty_map)

    def test_qg(self, max_examples=-1, use_difficulty=False, use_skills=False, use_state=False, use_history=False):
        if not use_state and not use_difficulty and not use_skills and not use_history:
            logging.error('test model type')
            exit(1)

        train_test_dev_dataset = self.build_dataset(
            target_split=[1, 2, 3],
            return_kt=True,
            return_aqg=True,
            qg_use_skills=use_skills,
            qg_use_difficulty=use_difficulty,
            qg_use_state=use_state,
            qg_use_history=use_history,
            max_examples=max_examples,
        )

        logging.info('dataset created, {} examples'.format(len(train_test_dev_dataset)))
        for index in random.sample(range(len(train_test_dev_dataset)), 1):
            for key in train_test_dev_dataset[index]:
                logging.info('{}-th example: {}, shape {}'.format(index, key, train_test_dev_dataset[index][key].shape))

        data_loader = DataLoader(train_test_dev_dataset, batch_size=1)

        token_skill_id_map = ConstrainedDecodingWithLookahead.map_ids(self.word_map, self.qg_tokenizer, oov_id=self.word_map['<pad>'])
        mask_oov = torch.where(token_skill_id_map == self.word_map['<pad>'], 0, 1).to(self.device)

        with torch.no_grad():
            for batch_id, batch_data in enumerate(data_loader):
                # logging.info('batch_id {}'.format(batch_id))
                batch_data = {key: batch_data[key].to(self.device) for key in batch_data}
                logits = self.knowledge_tracer(
                    x_word_ids=batch_data['x_kt_word_ids'],
                    y_labels=batch_data['y_kt_labels'],
                    x_user_ids=None
                )
                knowledge_states = torch.sigmoid(logits).squeeze(0)  # [batch_size(1), seq_len(1024), num_words]

                num_questions = batch_data['x_qg_input_ids'].size(1)

                start_point = None
                for i in range(num_questions):
                    # print(batch_data['x_qg_split_ids'].cpu().detach().numpy().tolist())
                    if batch_data['x_qg_split_ids'][0][i] == 3:
                        start_point = i
                        break

                test_num_questions = num_questions - start_point

                logging.info('{}/{} examples'.format(batch_id+1, len(train_test_dev_dataset)))
                skill_difficulties = knowledge_states[batch_data['x_qg_state_positions'][:, start_point:].squeeze(0)]  # [question_num, skill_num]
                sub_word_difficulties = skill_difficulties[torch.arange(test_num_questions).unsqueeze(1), token_skill_id_map] * mask_oov  # [question_num, 1, vocab_size]

                reference_ids = batch_data['y_qg_labels'][:, start_point:].clone().squeeze(0)
                reference_ids[reference_ids == -100] = self.qg_tokenizer.pad_token_id
                reference_word_difficulties = torch.gather(input=sub_word_difficulties, index=reference_ids, dim=-1)

                # print('reference word difficulties', reference_word_difficulties.detach().cpu().numpy().tolist())
                # print('labels', batch_data['y_qg_labels'].detach().cpu().numpy().tolist())
                # exit(1)
                estimated_difficulty = torch.sum(reference_word_difficulties, dim=-1).unsqueeze(1)

                ground_truth_difficulty = batch_data['x_adaptive_difficulties'][:, start_point:].transpose(0, 1)

                input_difficulties = None
                if self.d_source == 'kt':
                    input_difficulties = estimated_difficulty
                elif self.d_source == 'gd':
                    input_difficulties = ground_truth_difficulty
                # print('here', input_difficulties.detach().cpu().numpy().tolist())
                # print('input_difficulties', input_difficulties.shape)
                # exit(1)

                # print('here', batch_data['x_qg_input_ids'][:, start_point:].squeeze(0))
                # exit(1)
                if self.d_type == 'discrete':
                    output_ids = self.question_generator.generator.generate(
                            inputs=batch_data['x_qg_input_ids'][:, start_point:].squeeze(0),
                            attention_mask=batch_data['x_qg_attention_masks'][:, start_point:].squeeze(0),
                            num_beams=int(self.args.qg_num_beams),
                            max_length=int(self.args.qg_y_max_length)
                    )
                else:
                    inputs_embeds = self.question_generator.get_inputs_embeds(
                            x_input_ids=batch_data['x_qg_input_ids'][:, start_point:].squeeze(0),  # [num_questions, input_len]
                            # difficulty=batch_data['x_adaptive_difficulties'][:, start_point:].squeeze(0).unsqueeze(1) if use_difficulty else None,  # [num_questions, 1]
                            difficulty=input_difficulties if use_difficulty else None,
                            student_state=knowledge_states[batch_data['x_qg_state_positions'][:, start_point:]].squeeze(0).to(self.device) if use_state else None,
                    )
                    output_ids = self.question_generator.generator.generate(
                            inputs_embeds=inputs_embeds,
                            attention_mask=batch_data['x_qg_attention_masks'][:, start_point:].squeeze(0),
                            num_beams=int(self.args.qg_num_beams),
                            max_length=int(self.args.qg_y_max_length),
                    )

                # split_ids = batch_data['x_qg_split_ids'][:, start_point:].squeeze(0).detach().cpu().numpy().tolist()
                # print(1, split_ids.shape)
                adaptive_difficulties = batch_data['x_adaptive_difficulties'][:, start_point:].squeeze(0).detach().cpu().numpy().tolist()
                # print(2, adaptive_difficulties.shape)
                target_words = self.qg_tokenizer.batch_decode(batch_data['x_qg_keyword_ids'][:, start_point:].squeeze(0).detach().cpu(), skip_special_tokens=True)
                generations = self.qg_tokenizer.batch_decode(output_ids.detach().cpu(), skip_special_tokens=True)
                reference_ids = batch_data['y_qg_labels'][:, start_point:].squeeze(0)
                reference_ids[reference_ids == -100] = self.qg_tokenizer.pad_token_id
                references = self.qg_tokenizer.batch_decode(reference_ids.detach().cpu(), skip_special_tokens=True)
                # print(3, len(target_words), len(generations), len(references))
                abilities = 1 - torch.mean(skill_difficulties, dim=-1)  # []
                # print(knowledge_states.shape)
                # print(batch_data['x_qg_state_positions'][:, start_point:].shape)
                target_knowledge_states = knowledge_states[batch_data['x_qg_state_positions'][:, start_point:].squeeze(0)].detach().cpu().numpy().tolist()
                # print(4, target_knowledge_states.shape)

                # print('skill_difficulties', skill_difficulties.shape)
                # print('sub_word_difficulties', sub_word_difficulties.shape)
                # print('output_ids', output_ids.shape)
                input_difficulties = input_difficulties.squeeze(1).detach().cpu().numpy().tolist()
                generated_word_difficulties = torch.gather(input=sub_word_difficulties, index=output_ids, dim=-1)
                generated_difficulties = torch.sum(generated_word_difficulties, dim=-1).detach().cpu().numpy().tolist()
                # print('here', generated_difficulties)
                self.pqg_evaluator.add(
                    student_id=ascii_decode(batch_data['x_user_ascii'][0]),
                    references=references,
                    generations=generations,
                    ability=abilities.detach().cpu().numpy().tolist(),
                    input_difficulties=input_difficulties,
                    ground_truth_difficulties=ground_truth_difficulty.squeeze(1).detach().cpu().numpy().tolist(),
                    estimated_difficulties=estimated_difficulty.squeeze(1).detach().cpu().numpy().tolist(),
                    generated_difficulties=generated_difficulties,
                    reference_word_difficulties=reference_word_difficulties.detach().cpu().numpy().tolist(),
                    generated_word_difficulties=generated_word_difficulties.detach().cpu().numpy().tolist(),
                    # split_ids=split_ids,
                    target_words=target_words,
                    knowledge_states=target_knowledge_states,
                )

        results = self.pqg_evaluator.compute_metrics()
        self.pqg_evaluator.output()
        print(json.dumps(results))
        exit(1)

    def train_kt(self, use_dev=True, save_result=False):

        kt_train_dev_dataset = self.build_dataset(
            target_split=[1, 2] if use_dev else [1],
            return_kt=True,
            return_aqg=True,
            qg_use_skills=True,
            qg_use_difficulty=False,
            qg_use_state=False,
            qg_use_history=False,
            max_examples=-1,
        )   # for train
        # exit(1)
        # kt_train_dev_test_dataset = self.build_dataset(
        #     target_split=[1, 2, 3],
        #     return_kt=True,
        #     return_aqg=True,
        #     qg_use_skills=True,
        #     qg_use_difficulty=False,
        #     qg_use_state=False,
        #     qg_use_history=False,
        #     max_examples=10
        # )   # for test

        kt_train_dev_dataloader = DataLoader(kt_train_dev_dataset, batch_size=1)  # 因为要做question-level prediction因此batch比较麻烦，暂时取1
        kt_train_dev_test_dataloader = DataLoader(kt_train_dev_dataset, batch_size=1, shuffle=False) # DataLoader(kt_train_dev_test_dataset, batch_size=1, shuffle=False)

        batch_steps = len(kt_train_dev_dataset)  # // args.kt_train_batch_size // max(1, gpu_cnt)
        total_steps = batch_steps * self.args.kt_train_epoch
        warmup_steps = int(self.args.kt_warmup_rate * total_steps)

        loss_function = torch.nn.BCEWithLogitsLoss(reduction='none')

        optimizer = AdamW(self.knowledge_tracer.parameters(), lr=self.args.kt_learning_rate)
        lr_scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

        logging.info('rank {} start training, learning_rate: {}, total_batch_size: {}, batch_size_per_device: {}, batch_steps: {}, total_steps: {}, warmup_rate: {} warmup_steps: {}'.format(
            self.local_rank, self.args.kt_learning_rate, self.args.kt_train_batch_size * max(1, self.gpu_cnt), self.args.kt_train_batch_size, batch_steps, total_steps, self.args.kt_warmup_rate, warmup_steps
        ))

        save_info = {
            'epoch': 0,
            'loss': 0,
            'best_performance': None,
            'model_state_dict': None,
            'optimizer_state_dict': None
        }

        for epoch_id in range(self.args.kt_train_epoch):
            if self.gpu_cnt > 1:
                kt_train_dev_dataloader.sampler.set_epoch(epoch_id)

            self.kt_evaluator.clear()

            epoch_loss = 0
            self.knowledge_tracer.train()

            for batch_id, batch_data in enumerate(kt_train_dev_dataloader):
                if torch.unique(batch_data['x_kt_split_ids']).size(0) != 4:
                    logging.error('incomplete data'.format(batch_id))
                    continue  # incomplete data
                optimizer.zero_grad()
                batch_data = {key: batch_data[key].to(self.device) for key in batch_data}

                logits = self.knowledge_tracer(
                    x_user_ids=None,
                    x_word_ids=batch_data['x_kt_word_ids'],
                    y_labels=batch_data['y_kt_labels']
                )  # [batch_size, seq_len, num_words]

                kt_predictions = self.collect_kt_predictions(logits, batch_data)
                train_positions = torch.where(batch_data['x_kt_split_ids'] == 1, 1, 0)
                dev_positions = torch.where(batch_data['x_kt_split_ids'] == 2, 1, 0)

                optimized_positions = train_positions
                if use_dev:
                    optimized_positions = train_positions | dev_positions

                y_kt_labels_optimized = torch.where(optimized_positions == 1, batch_data['y_kt_labels'], -100)   # "original labels" for trained positions, "-100" for others
                example_weights = torch.where(y_kt_labels_optimized == 1, self.args.dkt_pos_weight, y_kt_labels_optimized)  # set positive weights
                example_weights = torch.where(y_kt_labels_optimized == 0, self.args.dkt_neg_weight, example_weights)  # set negative weights
                example_weights = torch.where(y_kt_labels_optimized == -100, 0, example_weights)  # set pad weights (0)

                # print('y_kt_labels_optimized', torch.isnan(y_kt_labels_optimized).any())
                # print('logits_from_cur_step', torch.isnan(kt_predictions['logits_from_cur_step']).any())
                # print('y_kt_labels_optimized', y_kt_labels_optimized.detach().cpu().numpy().tolist())
                # exit(1)
                # loss for predicting current word
                loss_current_word = loss_function(input=kt_predictions['logits_from_cur_step'], target=y_kt_labels_optimized.to(self.device).float())  # BCE
                loss_current_word = (loss_current_word * example_weights).sum() / example_weights.sum()  # filter pad tokens and get correct mean

                # loss for predicting next word
                loss_next_word = loss_function(kt_predictions['logits_from_last_step'], y_kt_labels_optimized.to(self.device).float())
                loss_next_word = (loss_next_word * example_weights).sum() / example_weights.sum()

                # loss for predicting next question
                loss_next_question = loss_function(kt_predictions['logits_from_last_question_last_step'], y_kt_labels_optimized.to(self.device).float())
                loss_next_question = (loss_next_question * example_weights).sum() / example_weights.sum()

                kt_total_loss = self.args.dkt_loss_next_word_weight * loss_next_word + \
                                self.args.dkt_loss_current_word_weight * loss_current_word + \
                                self.args.dkt_loss_next_question_weight * loss_next_question + \
                                self.args.dkt_loss_l1_weight * kt_predictions['l1_reg'] + \
                                self.args.dkt_loss_l2_weight * kt_predictions['l2_reg']

                epoch_loss += kt_total_loss
                kt_total_loss.backward()
                optimizer.step()
                lr_scheduler.step()

                logging.info('{}/{} epoch, {}/{} batch, total_loss {}, next_word_loss {}, cur_word_loss {}, next_q_loss {}, l1_reg: {}, l2_reg: {}, updated_skills_per_step {}'.format(
                    epoch_id + 1, self.args.kt_train_epoch, batch_id+1, batch_steps, kt_total_loss, loss_next_word, loss_current_word, loss_next_question,
                    kt_predictions['l1_reg'], kt_predictions['l2_reg'], kt_predictions['num_updated_skills']
                ))

            self.knowledge_tracer.eval()  # eval
            for batch_id, batch_data in enumerate(kt_train_dev_test_dataloader):
                if torch.unique(batch_data['x_kt_split_ids']).size(0) != 4:
                    logging.error('incomplete data {}'.format(batch_id))
                optimizer.zero_grad()
                batch_data = {key: batch_data[key].to(self.device) for key in batch_data}

                logits = self.knowledge_tracer(
                    x_user_ids=None,
                    x_word_ids=batch_data['x_kt_word_ids'],
                    y_labels=batch_data['y_kt_labels']
                )  # [batch_size, seq_len, num_words]

                kt_predictions = self.collect_kt_predictions(logits, batch_data)

                # (teacher forcing) evaluation  #TODO: consider modification
                self.kt_evaluator.add(
                    user_id=batch_data['x_user_ascii'].detach().cpu().numpy()[0],
                    user_ability=0,
                    word_ids=batch_data['x_kt_word_ids'].detach().cpu().numpy()[0],
                    pred_pos_probs_from_cur_step=torch.sigmoid(kt_predictions['logits_from_cur_step']).detach().cpu().numpy()[0],
                    pred_pos_probs_from_last_step=torch.sigmoid(kt_predictions['logits_from_last_step']).detach().cpu().numpy()[0],
                    pred_pos_probs_from_last_question_last_step=torch.sigmoid(kt_predictions['logits_from_last_question_last_step']).detach().cpu().numpy()[0],
                    labels=batch_data['y_kt_labels'].detach().cpu().numpy()[0],
                    split_ids=batch_data['x_kt_split_ids'].detach().cpu().numpy()[0],
                    interaction_ids=batch_data['x_kt_interaction_ids'].detach().cpu().numpy()[0],
                    mastery_probs=(1-torch.sigmoid(logits)).detach().cpu().numpy()[0],
                    valid_length=batch_data['x_kt_valid_length'].detach().cpu().numpy()[0],
                    valid_interactions=batch_data['x_kt_valid_interactions'].detach().cpu().numpy()[0],
                )

            results = self.kt_evaluator.compute_metrics()
            # self.inference_kt(kt_train_dev_test_dataset, is_dev=False)
            # results = self.kt_evaluator.compute_metrics()
            logging.info('-- {}/{} epoch, total loss is {}, kt_performance:\n train: {},\n dev: {},\n test: {}.'.format(
                epoch_id+1, self.args.kt_train_epoch, epoch_loss, results['train'], results['dev'], results['test']))

            if not save_info['best_performance'] or results['test']['roc_next_w'] > save_info['best_performance']['test']['roc_next_w']:
                save_info['epoch'] = epoch_id
                save_info['loss'] = epoch_loss
                save_info['best_performance'] = results
                save_info['model_state_dict'] = deepcopy(self.knowledge_tracer.state_dict())
                save_info['optimizer_state_dict'] = deepcopy(optimizer.state_dict())

            logging.info('-- local rank {} finished training'.format(self.local_rank))

        logging.info('Rank 0, best model: {}-th epoch, loss: {}, best_performance: {}, saving best epoch result to {}'.format(
                    save_info['epoch'], save_info['loss'], save_info['best_performance'], self.args.kt_best_epoch_result))

        if save_result:
            self.kt_evaluator.save_result(self.args.kt_best_epoch_result)
        logging.info('Rank 0, best epoch results saved!')

        model_name = 'kt_{}_nw{}_nq{}_l1{}_l2{}'.format(
            self.args.dkt_encoder,
            self.args.dkt_loss_next_word_weight,
            self.args.dkt_loss_next_question_weight,
            self.args.dkt_loss_l1_weight,
            self.args.dkt_loss_l2_weight,
        )
        model_save_path = os.path.join(self.args.kt_model_save_dir, '{}_{}ep.pth'.format(model_name, save_info['epoch']))
        logging.info('Rank 0, saving best model to {} ...'.format(model_save_path))
        torch.save(save_info, model_save_path)
        logging.info('Rank 0, best knowledge tracing model saved!')

    def train_qg(self, use_difficulty, use_skill, use_state, use_history, min_history, temperature, inner_batch=64, max_examples=-1, joint_train=False, joint_start=3, kt_use_dev=False, inc=False):
        qg_train_dataset = self.build_dataset(
            target_split=[1, 2],
            return_kt=True,
            return_aqg=True,
            qg_use_skills=use_skill,
            qg_use_difficulty=use_difficulty,
            qg_use_state=use_state,
            qg_use_history=use_history,
            max_examples=max_examples,
        )   # for train

        # qg_dev_dataset = self.build_dataset(
        #     target_split=[1, 2],
        #     return_kt=False,
        #     return_aqg=True,
        #     qg_use_skills=use_skill,
        #     qg_use_difficulty=use_difficulty,
        #     qg_use_state=use_state,
        #     qg_use_history=use_history,
        #     max_examples=100,
        # )

        qg_train_dev_dataloader = DataLoader(qg_train_dataset, batch_size=1)
        # qg_dev_dataloader = DataLoader(qg_dev_dataset, batch_size=1, shuffle=True)
        total_steps = len(qg_train_dataset) * self.args.qg_num_train_epoch   # int(self.args.qg_num_train_epoch) * len(qg_train_dataset)
        warmup_steps = int(total_steps * float(self.args.qg_warmup_rate))
        kt_loss_function = torch.nn.BCEWithLogitsLoss(reduction='none', )
        l1_loss_function = nn.L1Loss()
        optimizer = AdamW([
            {'params': self.question_generator.parameters()},
            {'params': self.knowledge_tracer.parameters()}
        ], lr=float(self.args.qg_learning_rate))

        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        logging.info('local_rank {}, start training, total steps {}, warm up steps {}'.format(self.local_rank, total_steps, warmup_steps))

        total_batch = 0

        token_skill_id_map = ConstrainedDecodingWithLookahead.map_ids(self.word_map, self.qg_tokenizer, oov_id=self.word_map['<pad>'])
        mask_oov = torch.where(token_skill_id_map == self.word_map['<pad>'], 0, 1).to(self.device)

        # for batch_id, batch_data in enumerate(qg_train_dev_dataloader):
        #     print(ascii_decode(batch_data['x_user_ascii'][0]))
        # exit(1)
        best_kt_result = None
        best_eval_loss = float('inf')
        best_epoch_id = -1
        for epoch_id in range(int(self.args.qg_num_train_epoch)):
            self.question_generator.train()
            train_loss = 0
            if self.gpu_cnt > 1:
                qg_train_dev_dataloader.sampler.set_epoch(epoch_id)

            for batch_id, batch_data in enumerate(qg_train_dev_dataloader):
                # for key in batch_data:
                #     print(key)
                if epoch_id == 0 and batch_id == 0:
                    for key in batch_data:
                        logging.debug('batch_data: {}:{}'.format(key, batch_data[key]))
                # data keys: 'x_target_words',  x_difficulties, 'x_input_ids', 'x_attention_mask', 'y_decoder_input_ids', 'y_labels'
                # batch_data = {key: batch_data[key].to(self.device) for key in batch_data}
                # x_input_ids, x_attention_mask, x_difficulties, y_decoder_input_ids, y_labels = batch_data
                # batch_data = {key: batch_data[key].to(self.device) for key in batch_data}  ## Avoid OOM

                if batch_data['x_qg_input_ids'].size(1) <= min_history:
                    continue  # Drop too short sequence

                dev_start_id = (batch_data['x_qg_split_ids'][0] == 2).nonzero(as_tuple=True)[0][0]
                if dev_start_id <= min_history:
                    continue  # Drop too short sequence

                if joint_train and epoch_id+1 >= joint_start:
                    # joint learning
                    self.knowledge_tracer.train()
                    logits = self.knowledge_tracer(
                        x_word_ids=batch_data['x_kt_word_ids'].to(self.device),
                        y_labels=batch_data['y_kt_labels'].to(self.device),
                        x_user_ids=None
                    )
                    knowledge_states = torch.sigmoid(logits).squeeze(0)  # [batch_size(1), seq_len(1024), num_words]
                    kt_predictions = self.collect_kt_predictions(logits, batch_data)
                    train_positions = torch.where(batch_data['x_kt_split_ids'] == 1, 1, 0).to(self.device)
                    dev_positions = torch.where(batch_data['x_kt_split_ids'] == 2, 1, 0).to(self.device)
                    optimized_positions = train_positions
                    if kt_use_dev:
                        optimized_positions = train_positions | dev_positions
                    y_kt_labels_optimized = torch.where(optimized_positions == 1, batch_data['y_kt_labels'].to(self.device), -100)  # "original labels" for trained positions, "-100" for others
                    example_weights = torch.where(y_kt_labels_optimized == 1, self.args.dkt_pos_weight, y_kt_labels_optimized)  # set positive weights
                    example_weights = torch.where(y_kt_labels_optimized == 0, self.args.dkt_neg_weight, example_weights)  # set negative weights
                    example_weights = torch.where(y_kt_labels_optimized == -100, 0, example_weights)  # set pad weights (0)
                    loss_current_word = kt_loss_function(input=kt_predictions['logits_from_cur_step'], target=y_kt_labels_optimized.float())  # BCE
                    loss_current_word = (loss_current_word * example_weights).sum() / example_weights.sum()  # filter pad tokens and get correct mean
                    # loss for predicting next word
                    loss_next_word = kt_loss_function(kt_predictions['logits_from_last_step'], y_kt_labels_optimized.float())
                    loss_next_word = (loss_next_word * example_weights).sum() / example_weights.sum()
                    # loss for predicting next question
                    loss_next_question = kt_loss_function(kt_predictions['logits_from_last_question_last_step'], y_kt_labels_optimized.float())
                    loss_next_question = (loss_next_question * example_weights).sum() / example_weights.sum()
                    kt_batch_loss = self.args.dkt_loss_next_word_weight * loss_next_word + \
                                    self.args.dkt_loss_current_word_weight * loss_current_word + \
                                    self.args.dkt_loss_next_question_weight * loss_next_question + \
                                    self.args.dkt_loss_l1_weight * kt_predictions['l1_reg'] + \
                                    self.args.dkt_loss_l2_weight * kt_predictions['l2_reg']

                else:
                    # print('here separate learning')
                    # separate learning
                    with torch.no_grad():
                        logits = self.knowledge_tracer(
                            x_word_ids=batch_data['x_kt_word_ids'].to(self.device),
                            y_labels=batch_data['y_kt_labels'].to(self.device),
                            x_user_ids=None
                        )
                        knowledge_states = torch.sigmoid(logits).squeeze(0)  # [batch_size(1), seq_len(1024), num_words]
                    kt_batch_loss = None

                student_states = knowledge_states[batch_data['x_qg_state_positions'].squeeze(0)]  # [question_num, num_words]
                # print('x_qg_state_positions', batch_data['x_qg_state_positions'].shape)
                # print('student_states', student_states.shape)
                # further split batch

                # print('split_ids', batch_data['x_qg_split_ids'][0].detach().cpu().numpy().tolist())

                # exit(1)
                # print('x_qg_input_ids', batch_data['x_qg_input_ids'].shape)
                # print('reference_ids', reference_ids.shape)
                # print('sub_word_difficulties', sub_word_difficulties.shape)
                # exit(1)
                input_difficulties = None
                sub_word_difficulties = student_states[torch.arange(batch_data['x_qg_input_ids'].size(1)).unsqueeze(1), token_skill_id_map] * mask_oov  # [question_num, vocab_size]
                if self.d_source == 'kt':
                    reference_ids = batch_data['y_qg_labels'].clone().squeeze(0)  # [question_num, seq_len]
                    reference_ids[reference_ids == -100] = self.qg_tokenizer.pad_token_id  # [question_num, seq_len]
                    input_difficulties = torch.sum(torch.gather(input=sub_word_difficulties.cpu(), index=reference_ids, dim=-1), dim=-1).unsqueeze(1)  # [question_num, 1]
                    # print('here use kt difficulty', input_difficulties.shape)
                else:
                    input_difficulties = batch_data['x_adaptive_difficulties'].transpose(0, 1)
                # print('here', input_difficulties.detach().cpu().numpy().tolist())
                # exit(1)
                # print('{}{}{}'.format('*'*20, batch_id, '*'*20))
                # print('dev_start_id: {}, min_history: {}, valid examples: {}, minibatch: {}'.format(dev_start_id, min_history, dev_start_id - min_history, n))
                # # print('steps', n)

                n = torch.max(torch.div(dev_start_id - min_history, inner_batch, rounding_mode='trunc'), torch.tensor(1))
                total_batch += n
                batch_qg_loss = []
                batch_tokens = []
                optimizer.zero_grad()

                output_ids = []
                # print('dev_start_id', dev_start_id, 'n', n)
                # for i in range(n):
                #
                #     start_id = min_history + i*inner_batch
                #     if i != n-1:
                #         end_id = min_history + (i + 1) * inner_batch
                #     else:
                #         end_id = dev_start_id
                #     # print('start_id', start_id, 'end_id', end_id)
                #     # print('start, end', start_id, end_id)
                #     # print('input_ids', batch_data['x_qg_input_ids'].squeeze(0)[start_id:end_id].shape)
                #     # print('x_qg_attention_masks', batch_data['x_qg_attention_masks'].squeeze(0)[:dev_start_id].shape)
                #     # print('y_qg_decoder_input_ids', batch_data['y_qg_decoder_input_ids'].squeeze(0)[:dev_start_id].shape)
                #     # print('y_qg_labels', batch_data['y_qg_labels'].squeeze(0)[:dev_start_id].shape)
                #     # print('here', batch_data['x_qg_input_ids'].squeeze(0)[start_id: end_id])
                #     # print('there', self.qg_tokenizer.batch_decode(batch_data['x_qg_input_ids'].squeeze(0)[start_id: end_id]))
                #     # exit(1)
                #     # print('x_qg_input_ids', batch_data['x_qg_input_ids'].squeeze(0)[start_id: end_id].shape)
                #     # print('x_qg_attention_masks', batch_data['x_qg_attention_masks'].squeeze(0)[start_id: end_id].shape)
                #     # print('input_difficulties', input_difficulties[start_id:end_id].shape)
                #     # print('x_qg_state_positions', knowledge_states[batch_data['x_qg_state_positions'][:, start_id:end_id]].squeeze(0).shape)
                #     # print('y_qg_decoder_input_ids', batch_data['y_qg_decoder_input_ids'].squeeze(0)[start_id: end_id].shape)
                #     # print('y_qg_labels', batch_data['y_qg_labels'].squeeze(0)[start_id: end_id].shape)
                #     # print('x_qg_input_ids', batch_data['x_qg_input_ids'].squeeze(0)[start_id: end_id][0])
                #     # print('input_difficulties', input_difficulties[start_id:end_id][0])
                #     # print('decoder_input_ids', batch_data['y_qg_decoder_input_ids'].squeeze(0)[0])
                #     # exit(1)
                #     outputs = self.question_generator(
                #         x_input_ids=batch_data['x_qg_input_ids'].squeeze(0)[start_id: end_id].to(self.device),
                #         x_attention_mask=batch_data['x_qg_attention_masks'].squeeze(0)[start_id: end_id].to(self.device),
                #         difficulty=input_difficulties[start_id:end_id].to(self.device),
                #         student_state=knowledge_states[batch_data['x_qg_state_positions'][:, start_id:end_id]].squeeze(0).to(self.device),  # [question_num, skill_num],
                #         decoder_input_ids=batch_data['y_qg_decoder_input_ids'].squeeze(0)[start_id: end_id].to(self.device),
                #         labels=batch_data['y_qg_labels'].squeeze(0)[start_id: end_id].to(self.device)
                #     )
                #
                #     min_batch_output_ids = torch.softmax(outputs.logits / temperature, dim=-1)  # [q_num(mini_batch), seq_len, vocab_size]
                #     output_ids.append(min_batch_output_ids)
                #     # print('x_qg_input_ids', batch_data['x_qg_input_ids'].squeeze(0)[start_id: end_id].detach().cpu().numpy().tolist())
                #     # print('x_qg_input', self.qg_tokenizer.batch_decode(batch_data['x_qg_input_ids'].squeeze(0)[start_id: end_id]))
                #     # print('x_qg_target', self.qg_tokenizer.batch_decode(batch_data['x_qg_keyword_ids'].squeeze(0)[start_id: end_id]))
                #     # exit(1)
                #     tokens = torch.where(batch_data['y_qg_labels'].squeeze(0)[start_id: end_id] == -100, 0, 1).sum()
                #     batch_tokens.append(tokens)
                #     batch_qg_loss.append(outputs.loss*tokens)

                # output_ids = torch.cat(output_ids)  # [question_num, seq_len, vocab_size]

                # exit(1)
                outputs = self.question_generator(
                    x_input_ids=batch_data['x_qg_input_ids'].squeeze(0)[: dev_start_id].to(self.device),
                    x_attention_mask=batch_data['x_qg_attention_masks'].squeeze(0)[: dev_start_id].to(self.device),
                    difficulty=input_difficulties[:dev_start_id].to(self.device),
                    student_state=knowledge_states[batch_data['x_qg_state_positions'][:, :dev_start_id]].squeeze(0).to(self.device),  # [question_num, skill_num],
                    decoder_input_ids=batch_data['y_qg_decoder_input_ids'].squeeze(0)[: dev_start_id].to(self.device),
                    labels=batch_data['y_qg_labels'].squeeze(0)[: dev_start_id].to(self.device)
                )
                output_ids = torch.softmax(outputs.logits / temperature, dim=-1)

                # exit(1)

                generated_difficulty_app = torch.bmm(output_ids, sub_word_difficulties[min_history:dev_start_id, :].unsqueeze(-1))
                generated_difficulty_app = torch.sum(generated_difficulty_app.squeeze(-1), dim=1)

                inconsistency_loss = l1_loss_function(input_difficulties[min_history:dev_start_id].to(self.device).squeeze(1), generated_difficulty_app)
                # print('dev_start_id: {}, min_history: {}, valid examples: {}, minibatch: {}, stack_num:{}'.format(dev_start_id, min_history, dev_start_id - min_history, n, len(batch_loss)))
                # batch_tokens = torch.sum(torch.stack(batch_tokens))
                # batch_qg_loss = torch.sum(torch.stack(batch_qg_loss)) / batch_tokens
                batch_qg_loss = outputs.loss
                train_loss += batch_qg_loss

                # logging.info('kt loss is {}, qg loss is {}'.format(kt_batch_loss, batch_qg_loss))
                if joint_train and epoch_id + 1 >= joint_start and kt_batch_loss is not None:
                    batch_loss = kt_batch_loss + batch_qg_loss
                    if inc:
                        batch_loss += inconsistency_loss
                else:
                    batch_loss = batch_qg_loss
                    if inc:
                        batch_loss += inconsistency_loss
                batch_loss.backward()
                optimizer.step()
                lr_scheduler.step()

                logging.info('local_rank: {}, {}/{} epoch, {}/{} batch, {} tokens, qg_loss is {}, kt_loss is {}, inconsistency loss is {}, total_loss is {}'.format(
                        self.local_rank, epoch_id + 1, self.args.qg_num_train_epoch, batch_id + 1, len(qg_train_dataset), batch_tokens, batch_qg_loss, kt_batch_loss, inconsistency_loss, batch_loss
                ))

            # evaluation on dev set
            self.question_generator.eval()
            self.knowledge_tracer.eval()
            self.pqg_evaluator.clear()
            self.kt_evaluator.clear()
            eval_loss = 0
            for batch_id, batch_data in enumerate(qg_train_dev_dataloader):
                # data keys: 'x_target_words',  x_difficulties, 'x_input_ids', 'x_attention_mask', 'y_decoder_input_ids', 'y_labels'
                # batch_data = {key: batch_data[key].to(self.device) for key in batch_data}
                # x_input_ids, x_attention_mask, x_difficulties, y_decoder_input_ids, y_labels = batch_data
                # batch_data = {key: batch_data[key].to(self.device) for key in batch_data}
                if batch_data['x_qg_input_ids'].size(1) <= min_history:
                    continue  # Drop too short sequence
                logits = self.knowledge_tracer(
                    x_word_ids=batch_data['x_kt_word_ids'].to(self.device),
                    y_labels=batch_data['y_kt_labels'].to(self.device),
                    x_user_ids=None
                )
                kt_predictions = self.collect_kt_predictions(logits, batch_data)
                self.kt_evaluator.add(
                        user_id=batch_data['x_user_ascii'].detach().cpu().numpy()[0],
                        user_ability=0,
                        word_ids=batch_data['x_kt_word_ids'].detach().cpu().numpy()[0],
                        pred_pos_probs_from_cur_step=torch.sigmoid(kt_predictions['logits_from_cur_step']).detach().cpu().numpy()[0],
                        pred_pos_probs_from_last_step=torch.sigmoid(kt_predictions['logits_from_last_step']).detach().cpu().numpy()[0],
                        pred_pos_probs_from_last_question_last_step=torch.sigmoid(kt_predictions['logits_from_last_question_last_step']).detach().cpu().numpy()[0],
                        labels=batch_data['y_kt_labels'].detach().cpu().numpy()[0],
                        split_ids=batch_data['x_kt_split_ids'].detach().cpu().numpy()[0],
                        interaction_ids=batch_data['x_kt_interaction_ids'].detach().cpu().numpy()[0],
                        mastery_probs=(1 - torch.sigmoid(logits)).detach().cpu().numpy()[0],
                        valid_length=batch_data['x_kt_valid_length'].detach().cpu().numpy()[0],
                        valid_interactions=batch_data['x_kt_valid_interactions'].detach().cpu().numpy()[0],
                )

                knowledge_states = torch.sigmoid(logits).squeeze(0)  # [batch_size(1), seq_len(1024), num_words]
                student_states = knowledge_states[batch_data['x_qg_state_positions'].squeeze(0)]  # [question_num, num_words]
                reference_ids = batch_data['y_qg_labels'].clone().squeeze(0)  # [question_num, seq_len]
                reference_ids[reference_ids == -100] = self.qg_tokenizer.pad_token_id  # [question_num, seq_len]
                sub_word_difficulties = student_states[torch.arange(batch_data['x_qg_input_ids'].size(1)).unsqueeze(1), token_skill_id_map] * mask_oov
                input_difficulties = torch.sum(torch.gather(input=sub_word_difficulties.cpu(), index=reference_ids, dim=-1), dim=-1).unsqueeze(1)  # [question_num, 1]

                dev_start_id = (batch_data['x_qg_split_ids'][0] == 2).nonzero(as_tuple=True)[0][0]  # dev start position
                num_examples = batch_data['x_qg_input_ids'].size(1) - dev_start_id
                # print('num_examples', num_examples)
                n = torch.max(torch.div(num_examples, inner_batch, rounding_mode='trunc'), torch.tensor(1))

                for i in range(n):
                    start_id = dev_start_id + i*inner_batch
                    if i != n-1:
                        end_id = dev_start_id + (i+1)*inner_batch
                    else:
                        end_id = batch_data['x_qg_input_ids'].size(1)
                    with torch.no_grad():
                        # print('='*50)
                        # print('start, end', start_id, end_id)
                        # print('input_ids', batch_data['x_qg_input_ids'].squeeze(0)[start_id:end_id].shape)
                        # print('x_qg_attention_masks', batch_data['x_qg_attention_masks'].squeeze(0)[start_id:end_id].shape)
                        # print('y_qg_decoder_input_ids', batch_data['y_qg_decoder_input_ids'].squeeze(0)[start_id:end_id].shape)
                        # print('y_qg_labels', batch_data['y_qg_labels'].squeeze(0)[start_id:end_id].shape)
                        outputs = self.question_generator(
                            x_input_ids=batch_data['x_qg_input_ids'].squeeze(0)[start_id: end_id].to(self.device),
                            x_attention_mask=batch_data['x_qg_attention_masks'].squeeze(0)[start_id: end_id].to(self.device),
                            difficulty=input_difficulties[start_id:end_id].to(self.device),
                            student_state=knowledge_states[batch_data['x_qg_state_positions'][:, start_id:end_id]].squeeze(0).to(self.device),  # [question_num, skill_num],
                            decoder_input_ids=batch_data['y_qg_decoder_input_ids'].squeeze(0)[start_id: end_id].to(self.device),
                            labels=batch_data['y_qg_labels'].squeeze(0)[start_id: end_id].to(self.device)
                        )
                        num_tokens = torch.where(batch_data['y_qg_labels'].squeeze(0)[start_id: end_id] == -100, 0, 1).sum()
                        loss = outputs.loss
                        eval_loss += loss * num_tokens

            kt_eval_results = self.kt_evaluator.compute_metrics()
            logging.info('rank {}, {}/{} epoch, eval loss is {}, kt_eval_results is {}'.format(self.local_rank, epoch_id+1, self.args.qg_num_train_epoch, eval_loss, kt_eval_results))

            if eval_loss < best_eval_loss:
                # save_info['epoch'] = epoch_id + 1
                # save_info['loss'] = train_loss
                # save_info['best_validation_performance'] = eval_loss
                # # save_info['best_model'] = copy.deepcopy(self.question_generator).cpu()
                # save_info['model_state_dict'] = deepcopy(self.question_generator.state_dict())
                # save_info['optimizer_state_dict'] = deepcopy(optimizer.state_dict())
                # best_eval_loss = eval_loss
                best_eval_loss = eval_loss  # eval_result
                best_epoch_id = epoch_id + 1
                file_name = 'baseline_{}_'.format(int(self.args.qg_prompt_words_sample_rate * 100))
                if use_history:
                    file_name += 'h_'
                if use_difficulty:
                    file_name += 'd_'
                    file_name += '{}_'.format(self.d_source)
                if use_skill:
                    file_name += 's_'
                if use_state:
                    file_name += 'a_'
                if joint_train:
                    file_name += 'joint_'
                if inc:
                    file_name += 'inc_'
                file_name += self.args.qg_model_name.replace('/', '-')
                file_name += '_best.pth'

                save_path = os.path.join(self.args.qg_model_save_dir, file_name)
                logging.info('saving best model to {}'.format(save_path))
                torch.save({
                    'epoch': epoch_id + 1,
                    'loss': train_loss,
                    'best_validation_performance': best_eval_loss,
                    'model_state_dict': self.question_generator.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, save_path)

            if not best_kt_result or best_kt_result['test']['roc_next_w'] < kt_eval_results['test']['roc_next_w']:
                best_kt_result = kt_eval_results
                if joint_train and epoch_id + 1 >= joint_start:
                    file_name = 'kt_joint_inc_best.pth'
                    kt_save_path = os.path.join(self.args.kt_model_save_dir, file_name)
                    torch.save({
                        'model_state_dict': self.knowledge_tracer.state_dict(),
                        'result': best_kt_result,
                        'epoch': epoch_id + 1
                    }, kt_save_path)

        # save best performing model
        if self.local_rank == 0:
            logging.info('-- local_rank:{}, training, best model: {}-th epoch, qg_eval_result: {}, kt_eval_result: {}'.format(
                self.local_rank, best_epoch_id, best_eval_loss, best_kt_result
            ))


    def load_knowledge_tracer(self, model_save_path):
        save_dict = torch.load(model_save_path)
        logging.info('loading knowledge tracing model from {}'.format(model_save_path))
        self.knowledge_tracer.load_state_dict(save_dict['model_state_dict'])

    def load_question_generator(self, model_save_path, rename_weights=False):
        save_dict = torch.load(model_save_path)
        logging.info('loading model best epoch {}'.format(save_dict['epoch']))
        if rename_weights:  # TODO: temp code
            for key in list(save_dict['model_state_dict'].keys()):
                save_dict['model_state_dict']['generator.'+key] = save_dict['model_state_dict'][key]
                del save_dict['model_state_dict'][key]
        logging.info('loading question generation model from {}'.format(model_save_path))
        self.question_generator.load_state_dict(save_dict['model_state_dict'])

    # def inference_kt(self, dataset, is_dev):
    #     # use previous outputs for inference
    #     self.knowledge_tracer.eval()
    #     kt_dataloader = DataLoader(dataset, shuffle=False, batch_size=1)
    #     for batch_id, batch_data in kt_dataloader:
    #         batch_data = {key: batch_data[key] for key in batch_data}
    #         non_pad_idx = torch.where(batch_data['x_kt_split_ids'] > 0, True, False)
    #
    #         if is_dev:
    #             # inference on dev set
    #             history_idx = torch.where(batch_data['x_kt_split_ids'] < 2, True, False)
    #             history_idx = history_idx & non_pad_idx
    #             target_idx = torch.where(batch_data['x_kt_split_ids'] == 2, True, False)
    #             target_idx = target_idx & non_pad_idx
    #         else:
    #             # inference on test set
    #             history_idx = torch.where(batch_data['x_kt_split_ids'] < 3)
    #             history_idx = history_idx & non_pad_idx
    #             target_idx = torch.where(batch_data['x_kt_split_ids'] == 3)
    #             target_idx = target_idx * non_pad_idx
    #
    #         x_word_ids_history = batch_data['x_kt_word_ids'][history_idx]
    #         x_word_ids_target = batch_data['x_kt_word_ids'][target_idx]
    #
    #         y_labels_history = batch_data['y_kt_labels'][history_idx]
    #         y_labels_target = batch_data['y_kt_labels'][target_idx]
    #
    #         # x_w_l_tuple_ids_history = x_w_l_tuple_ids[history_idx]
    #         # x_w_l_tuple_ids_target = x_w_l_tuple_ids[target_idx]
    #
    #         pos_probs = batch_data['y_kt_labels'].clone().float()
    #         logits = None
    #         inverse_mastery_probs = None
    #
    #         for idx in range(target_idx[0].size(1)):
    #             logits = self.knowledge_tracer(x_user_ids=None, x_word_ids=x_word_ids_history, y_labels=y_labels_history)
    #             inverse_mastery_probs = torch.sigmoid(logits)  # [batch_size(1), seq_len, num_words]
    #
    #             next_word_id = x_word_ids_target[0, idx].squeeze(0)  # [batch_size(1), ]
    #             next_word_pos_prob = inverse_mastery_probs[0, -1, next_word_id]
    #             pos_probs[0][idx] = next_word_pos_prob
    #             if next_word_pos_prob > args.kt_threshold:
    #                 next_label = 1
    #             else:
    #                 next_label = 0
    #
    #             x_word_ids_history = torch.cat([x_word_ids_history, next_word_id.view(1, 1)], dim=-1)
    #             y_labels_history = torch.cat([y_labels_history, torch.tensor(next_label).view(1, 1)], dim=-1)
    #
    #         # collect kt loss TODO: a method for KT loss
    #         kt_predictions = self.collect_kt_predictions(logits, batch_data)
    #
    #         # collect evaluation results
    #         self.kt_evaluator.add(
    #             user_id=batch_data['user_id'].detach().cpu().numpy(),
    #             user_ability=0,
    #             word_ids=batch_data['x_kt_word_ids'].detach().cpu().numpy(),
    #             pred_pos_probs_from_cur_step=torch.sigmoid(kt_predictions['logits_from_cur_step']).detach().cpu().numpy(),
    #             pred_pos_probs_from_last_step=torch.sigmoid(kt_predictions['logits_from_last_step']).detach().cpu().numpy(),
    #             pred_pos_probs_from_last_question_last_step=torch.sigmoid(kt_predictions['logits_from_last_question_last_step']).detach().cpu().numpy(),
    #             labels=batch_data['y_kt_labels'].detach().cpu().numpy(),
    #             split_ids=batch_data['x_kt_split_ids'].detach().cpu().numpy(),
    #             interaction_ids=batch_data['x_kt_interaction_ids'].detach().cpu().numpy(),
    #             mastery_probs=1-inverse_mastery_probs.detach().cpu().numpy(),
    #             valid_length=batch_data['x_kt_valid_length'].detach().cpu().numpy(),
    #             valid_interactions=batch_data['x_kt_valid_interactions'].detach().cpu().numpy(),
    #         )

    def simulate(self, use_skills, use_difficulty, use_state, use_history, max_examples, user_list, start_interaction, end_interaction, exercises, mode, save_path=None):

        assert mode in ['random', 'select', 'generate']
        train_dev_test_dataset = self.build_dataset(
            target_split=[1, 2, 3],
            return_kt=True,
            return_aqg=True,
            qg_use_skills=use_skills,
            qg_use_difficulty=use_difficulty,
            qg_use_state=use_state,
            qg_use_history=use_history,
            max_examples=max_examples,
            user_list=user_list
        )

        logging.info('{} data for simulation'.format(len(train_dev_test_dataset)))
        train_dev_test_dataloader = DataLoader(train_dev_test_dataset, batch_size=1, shuffle=False)

        simulation_results = []

        exercise_extended_word_ids = []
        exercise_lengths = []

        for exercise in exercises:
            words = exercise.split('#')
            word_ids = [self.word_map[word] for word in words]
            word_ids.extend([0 for i in range(15-len(word_ids))])
            exercise_lengths.append(len(words))
            exercise_extended_word_ids.append(word_ids)

        exercise_extended_word_ids = torch.tensor(exercise_extended_word_ids).to(self.device)  # [num_exercise, 15]
        exercise_lengths = torch.tensor(exercise_lengths).unsqueeze(1).to(self.device)  # # [num_exercise, 1]

        logging.info('exercise_extended_word_ids: {}, exercise_lengths: {}'.format(exercise_extended_word_ids.shape, exercise_lengths.shape))

        with torch.no_grad():
            for batch_id, batch_data in enumerate(train_dev_test_dataloader):
                logging.info('{}/{} examples'.format(batch_id+1, len(train_dev_test_dataloader)))
                # print('user_id', ascii_decode(batch_data['x_user_ascii'][0]), batch_data['x_kt_word_ids'].shape, batch_data['x_kt_valid_length'])
                # user_ascii = [str(x) for x in batch_data['x_user_ascii'].numpy().tolist()[0]]
                # file_name = '{}.npz'.format('-'.join(user_ascii))
                # dump_data = np.load(os.path.join('/cluster/project/sachan/pencui/ProjectsData/AdaptiveQG/kt/best_epoch_result', file_name))
                # for key in dump_data:
                #     print(key, dump_data[key].shape)
                # print('saved word_ids:', dump_data['word_ids'].tolist())
                # print('run word_ids:', batch_data['x_kt_word_ids'][0].numpy().tolist())
                # print('saved_states:', dump_data['mastery_level'].tolist())
                batch_data = {key: batch_data[key].to(self.device) for key in batch_data}
                start_position = (batch_data['x_kt_interaction_ids'][0] == start_interaction).nonzero(as_tuple=True)[0][-1]
                # end_position = (batch_data['x_kt_interaction_ids'][0] == end_interaction).nonzero(as_tuple=True)[0][-1]
                # print('start_positions', start_position, 'start_interaction', start_interaction)
                # exit(1)
                # original_knowledge_states = 1 - torch.sigmoid(self.knowledge_tracer(
                #     x_word_ids=batch_data['x_kt_word_ids'],
                #     y_labels=batch_data['y_kt_labels'],
                #     x_user_ids=None
                # ))
                # print('inferred states:', torch.mean(original_knowledge_states[0], dim=-1).detach().cpu().numpy().tolist())
                #
                # print('original start state:{}, end state:{}'.format(original_knowledge_states[0, start_position, :].mean(), original_knowledge_states[0, end_position, :].mean()))
                history_word_ids = batch_data['x_kt_word_ids'][:, :start_position+1]
                history_labels = batch_data['y_kt_labels'][:, :start_position+1]

                logits = self.knowledge_tracer(
                    x_word_ids=history_word_ids,
                    y_labels=history_labels,
                    x_user_ids=None
                )
                history_knowledge_states = torch.sigmoid(logits)

                # print('user_id: {},initial knowledge state {}'.format(ascii_decode(batch_data['x_user_ascii'][0]), mastery_levels[0]))
                # exit(1)
                exercise_list = []
                mastery_levels = [1 - history_knowledge_states[0, -1].mean().detach().cpu().numpy().tolist()]
                # print('mastery_levels', mastery_levels)
                for step in range(start_interaction, end_interaction):
                    logging.info('{}/{}'.format(step+1-start_interaction, end_interaction-start_interaction))
                    # select best exercise
                    best_next_exercise_ids = None
                    best_next_exercise_labels = None
                    if mode == 'random':
                        best_next_exercise = random.choice(exercises).replace('#', ' ')
                        best_next_exercise_ids = torch.tensor([[self.word_map[word] for word in best_next_exercise.split(' ')]]).to(self.device)
                        best_next_exercise_labels = torch.tensor([[0 if random.random() > history_knowledge_states[0, -1, word_id] else 1 for word_id in best_next_exercise_ids[0]]]).to(self.device)
                    elif mode == 'select':
                        torch.cuda.empty_cache()
                        # sort exercise by learning gain  TODO: reduce batch_size
                        history_word_ids_extended = history_word_ids.repeat(len(exercises), 1)
                        history_word_ids_extended = torch.cat([history_word_ids_extended, exercise_extended_word_ids], dim=-1)

                        r = torch.tensor(np.random.random(exercise_extended_word_ids.shape)).to(self.device)
                        probs = history_knowledge_states[0, -1, :].view(-1)[exercise_extended_word_ids]
                        extended_labels = torch.where(r > probs, 0, 1)
                        # print('extended_labels', extended_labels)
                        history_word_labels_extended = history_labels.repeat(len(exercises), 1)
                        history_word_labels_extended = torch.cat([history_word_labels_extended, extended_labels], dim=-1)

                        logits = self.knowledge_tracer(x_word_ids=history_word_ids_extended, y_labels=history_word_labels_extended, x_user_ids=None)
                        knowledge_states_extended = torch.mean(torch.sigmoid(logits), dim=-1)  # [exercise_num, seq_len, word_num]
                        new_knowledge_states = knowledge_states_extended[torch.arange(len(exercises)).unsqueeze(1), history_knowledge_states.size(1)+exercise_lengths].squeeze(0)
                        sorted_idx = torch.argsort(new_knowledge_states, dim=0, descending=False)
                        best_next_exercise = exercises[sorted_idx[0]]
                        # print('knowledge_states_extended', knowledge_states_extended.shape)
                        # print('new_knowledge_states', new_knowledge_states.shape)
                        # print('best_next_exercise', best_next_exercise)
                        # print('index', sorted_idx[0])
                        # print('sorted_index', sorted_idx)
                        best_next_exercise_ids = exercise_extended_word_ids[sorted_idx[0][0]][:exercise_lengths[sorted_idx[0]]].unsqueeze(0)
                        best_next_exercise_labels = extended_labels[sorted_idx[0][0]][:exercise_lengths[sorted_idx[0]]].unsqueeze(0)

                        # print('best_next_exercise_ids', best_next_exercise_ids)
                        # print('best_next_labels', best_next_labels)
                        # exit(1)
                        # print('here', new_knowledge_states[sorted_idx[0]].detach().cpu().numpy().tolist()[0], type(new_knowledge_states[sorted_idx[0]].detach().cpu().numpy().tolist()[0]))
                        # print(
                        #     'best_next exercise: {}, labels: {}, difficulties: {}, input_state: {}, output_state: {}'.format(
                        #         best_next_exercise,
                        #         extended_labels[sorted_idx[0]],
                        #         probs[sorted_idx[0]],
                        #         1 - torch.mean(history_knowledge_states[0, -1, :]).detach().cpu().numpy().tolist(),
                        #         1 - new_knowledge_states[sorted_idx[0]].detach().cpu().numpy().tolist()[0][0]
                        # ))
                        # exit(1)
                        # print('new_knowledge_states', new_knowledge_states.shape)
                        # exit(1)
                        # print('history_word_ids_extended', history_word_ids_extended.shape)
                        # print('probs', probs.shape)
                        # exit(1)
                        # for exercise in tqdm(exercises):  # TODO: replace this with expect calculation
                        #     new_word_ids = torch.tensor([[self.word_map[word] for word in exercise.split('#')]]).to(self.device)
                        #     new_word_labels = torch.tensor([[0 if random.random() > history_knowledge_states[0, -1, word_id] else 1 for word_id in new_word_ids[0]]]).to(self.device)
                        #
                        #     valid_lengths.append(len(new_word_ids))
                        #     new_word_ids = torch.tensor([[self.word_map[word] for word in exercise.split('#')]]).to(self.device)
                        #     new_word_labels = torch.tensor([[0 if random.random() > history_knowledge_states[0, -1, word_id] else 1 for word_id in new_word_ids[0]]]).to(self.device)
                        #     extended_word_ids = torch.cat([history_word_ids, new_word_ids], dim=-1)
                        #     extended_word_labels = torch.cat([history_labels, new_word_labels], dim=-1)
                        #     logits = self.knowledge_tracer(x_word_ids=extended_word_ids, y_labels=extended_word_labels, x_user_ids=None)
                        #     extended_knowledge_states = 1 - torch.sigmoid(logits)
                        #     exercise_gain.append((exercise, torch.mean(extended_knowledge_states[0, -1, :], dim=-1).detach().cpu().numpy().tolist()))
                        # exercise_gain.sort(key=lambda x: x[-1], reverse=True)
                        # best_next_exercise = exercise_gain[0][0]
                    elif mode == 'gen':
                        # sort words by learning gain
                        word_gain = []
                        for word_id in range(4, len(self.word_map)-1):
                            correct_prob = 1 - history_knowledge_states[0, -1, word_id]
                            word_extended = torch.cat([history_word_ids, torch.tensor([[word_id]]).to(self.device)], dim=-1)
                            correct_label_extended = torch.cat([history_labels, torch.tensor([[0]]).to(self.device)], dim=-1)
                            wrong_label_extended = torch.cat([history_labels, torch.tensor([[1]]).to(self.device)], dim=-1)

                            memory_states_correct = torch.sigmoid(self.knowledge_tracer(x_word_ids=word_extended, y_labels=correct_label_extended, x_user_ids=None))
                            memory_states_wrong = torch.sigmoid(self.knowledge_tracer(x_word_ids=word_extended, y_labels=wrong_label_extended, x_user_ids=None))

                            correct_gain = torch.sum(memory_states_correct[:, -2, :] - memory_states_correct[:, -1, :])
                            wrong_gain = torch.sum(memory_states_wrong[:, -2, :] - memory_states_wrong[:, -1, :])
                            expected_gain = correct_prob * correct_gain + (1 - correct_prob) * wrong_gain
                            word_gain.append((word_id, expected_gain))

                        word_gain.sort(key=lambda x: x[1], reverse=True)
                        best_word = word_gain[0][0]

                        best_next_exercise = None  # TODO: call generation model to generate based on best word

                    # print('best exercise "{}"'.format(best_next_exercise))
                    # new_word_wrong_probs = [history_knowledge_states[0, -1, word_id] for word_id in new_word_ids[0]]
                    history_word_ids = torch.cat([history_word_ids, best_next_exercise_ids], dim=-1)
                    history_labels = torch.cat([history_labels, best_next_exercise_labels], dim=-1)
                    logits = self.knowledge_tracer(x_word_ids=history_word_ids, y_labels=history_labels, x_user_ids=None)
                    history_knowledge_states = torch.sigmoid(logits)

                    # print('here', history_knowledge_states[0, -1, :].mean().detach().cpu().numpy().tolist())
                    # exit(1)
                    exercise_list.append(best_next_exercise.replace('#', ' '))
                    mastery_levels.append(1 - history_knowledge_states[0, -1, :].mean().detach().cpu().numpy().tolist())
                    # print('new exercise {}, new_probs: {}, new_labels: {}, levels: {}'.format(
                    #     exercise_list[-1], new_word_wrong_probs, new_word_labels, mastery_levels
                    # ))
                    # exit(1)

                simulation_results.append({
                    'student_id': ascii_decode(batch_data['x_user_ascii'][0]), 'mastery_levels': mastery_levels, 'exercise_list': exercise_list
                })
                # print(simulation_results[0])
                # exit(1)
        with open(save_path, 'w') as fp:
            for sr in simulation_results:
                fp.write(json.dumps(sr)+'\n')

        return simulation_results

    def simulate_difficulty(self, use_skills, use_difficulty, use_state, use_history, max_examples, student_list, target_difficulty, start_interaction, step, increasing_rate=0):

        word_list = list(self.word_map.keys())
        train_dev_test_dataset = self.build_dataset(
            target_split=[1, 2, 3],
            return_kt=True,
            return_aqg=True,
            qg_use_skills=use_skills,
            qg_use_difficulty=use_difficulty,
            qg_use_state=use_state,
            qg_use_history=use_history,
            max_examples=max_examples,
            user_list=student_list
        )

        train_dev_test_dataloader = DataLoader(train_dev_test_dataset, batch_size=1)

        token_skill_id_map = ConstrainedDecodingWithLookahead.map_ids(self.word_map, self.qg_tokenizer, oov_id=self.word_map['<pad>'])
        mask_oov = torch.where(token_skill_id_map == self.word_map['<pad>'], 0, 1).to(self.device)

        for batch_id, batch_data in enumerate(train_dev_test_dataloader):
            logging.info('{}/{} example'.format(batch_id, len(train_dev_test_dataloader)))
            start_position = (batch_data['x_kt_interaction_ids'][0] == start_interaction).nonzero(as_tuple=True)[0][-1]
            history_word_ids = batch_data['x_kt_word_ids'][:, :start_position]
            history_word_labels = batch_data['y_kt_labels'][:, :start_position]

            knowledge_states = torch.sigmoid(self.knowledge_tracer(x_word_ids=history_word_ids.to(self.device), y_labels=history_word_labels.to(self.device), x_user_ids=None))

            for idx in range(step):
                logging.info('{}/{} step'.format(idx, step))
                target_word = word_list[128]  # random.choice(word_list)
                target_difficulty = torch.tensor([target_difficulty * (1+increasing_rate) ** batch_id])
                print('target difficulty', target_difficulty)
                sub_word_difficulties = knowledge_states[0, -1][token_skill_id_map] * mask_oov  # [question_num, 1, vocab_size]
                # print('here', knowledge_states[0, -1].shape)
                # print('sub_word difficulties', sub_word_difficulties.shape)

                inputs = ''
                if use_state:
                    inputs += self.qg_tokenizer.pad_token
                if use_difficulty:
                    inputs += self.qg_tokenizer.pad_token
                if use_skills:
                    inputs += target_word
                print('inputs', inputs)

                encoded = self.qg_tokenizer(inputs,  padding='max_length', return_tensors='pt', max_length=self.args.qg_x_max_length)
                inputs_embeds = self.question_generator.get_inputs_embeds(
                    x_input_ids=encoded['input_ids'].to(self.device),  # [num_questions, input_len]
                    # difficulty=batch_data['x_adaptive_difficulties'][:, start_point:].squeeze(0).unsqueeze(1) if use_difficulty else None,  # [num_questions, 1]
                    difficulty=target_difficulty.to(self.device) if use_difficulty else None,
                    student_state=knowledge_states[:, -1].to(self.device) if use_state else None,
                )
                output_ids = self.question_generator.generator.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=encoded['attention_mask'].to(self.device),
                    num_beams=int(self.args.qg_num_beams),
                    max_length=int(self.args.qg_y_max_length),
                )
                output_difficulty = torch.sum(torch.gather(input=sub_word_difficulties, index=output_ids.squeeze(0), dim=0))
                print('output ids', output_ids, self.qg_tokenizer.batch_decode(output_ids))
                print('output_difficulty', output_difficulty)
                exit(1)
        pass

    def eval_kt(self, save_result):

        kt_train_dev_test_dataset = self.build_dataset(
            target_split=[1, 2, 3],
            return_kt=True,
            return_aqg=True,
            qg_use_skills=True,
            qg_use_difficulty=False,
            qg_use_state=False,
            qg_use_history=False,
            max_examples=-1
        )   # for test

        kt_train_dev_test_dataloader = DataLoader(kt_train_dev_test_dataset, batch_size=1, shuffle=False)

        self.knowledge_tracer.eval()  # eval
        self.kt_evaluator.clear()
        for batch_id, batch_data in enumerate(kt_train_dev_test_dataloader):
            batch_data = {key: batch_data[key].to(self.device) for key in batch_data}

            logits = self.knowledge_tracer(
                x_user_ids=None,
                x_word_ids=batch_data['x_kt_word_ids'],
                y_labels=batch_data['y_kt_labels']
            )  # [batch_size, seq_len, num_words]

            kt_predictions = self.collect_kt_predictions(logits, batch_data)

            # (teacher forcing) evaluation  #TODO: consider modification
            self.kt_evaluator.add(
                user_id=batch_data['x_user_ascii'].detach().cpu().numpy()[0],
                user_ability=0,
                word_ids=batch_data['x_kt_word_ids'].detach().cpu().numpy()[0],
                pred_pos_probs_from_cur_step=torch.sigmoid(kt_predictions['logits_from_cur_step']).detach().cpu().numpy()[0],
                pred_pos_probs_from_last_step=torch.sigmoid(kt_predictions['logits_from_last_step']).detach().cpu().numpy()[0],
                pred_pos_probs_from_last_question_last_step=torch.sigmoid(kt_predictions['logits_from_last_question_last_step']).detach().cpu().numpy()[0],
                labels=batch_data['y_kt_labels'].detach().cpu().numpy()[0],
                split_ids=batch_data['x_kt_split_ids'].detach().cpu().numpy()[0],
                interaction_ids=batch_data['x_kt_interaction_ids'].detach().cpu().numpy()[0],
                mastery_probs=(1 - torch.sigmoid(logits)).detach().cpu().numpy()[0],
                valid_length=batch_data['x_kt_valid_length'].detach().cpu().numpy()[0],
                valid_interactions=batch_data['x_kt_valid_interactions'].detach().cpu().numpy()[0],
            )

        results = self.kt_evaluator.compute_metrics()
        logging.info('kt evaluation result {}'.format(results))
        if save_result:
            logging.info('save result to {}'.format(self.args.kt_best_epoch_result))
            self.kt_evaluator.save_result(self.args.kt_best_epoch_result)

    def collect_kt_predictions(self, logits, batch_data):
        # logits: batch_size, seq_len, num_word
        x_word_ids_indices = batch_data['x_kt_word_ids'].unsqueeze(-1).to(self.device)  # [batch_size, seq_len, 1]

        # predictions from cur (t) knowledge_state
        logits_from_cur_step = torch.gather(logits, -1, x_word_ids_indices).squeeze(-1)  # [batch_size, seq_len]
        # print('logits_from_cur_step', logits_from_cur_step.detach().cpu().numpy().tolist())
        # print('logits_from_cur_step', logits_from_cur_step.shape)

        # predictions from last (t-1) knowledge_state
        logits_from_last_step = torch.gather(torch.roll(logits, 1, dims=1), -1, x_word_ids_indices).squeeze(-1)  # [batch_size, seq_len]
        # print('logits_from_last_step', logits_from_last_step.detach().cpu().numpy().tolist())
        # print('logits_from_last_step', logits_from_last_step.shape)

        # predictions from last (n-1) last knowledge_state
        # print('x_qg_state_positions', batch_data['x_qg_state_positions'].shape)
        # print('x_kt_interaction_ids', batch_data['x_kt_interaction_ids'].shape, torch.max(batch_data['x_kt_interaction_ids']), torch.min(batch_data['x_kt_interaction_ids']))

        x_kt_interaction_ids = batch_data['x_kt_interaction_ids'].clone().to(self.device)
        x_kt_interaction_ids[x_kt_interaction_ids == -1] = 0

        # equal to roll operation
        state_indices_last_question = torch.gather(input=batch_data['x_qg_state_positions'].to(self.device), index=x_kt_interaction_ids, dim=-1)  # logits positions
        logits_from_last_question_last_step = logits[torch.arange(logits.size(0)), state_indices_last_question]  # [1, 1024, 2098]

        # gather words
        logits_from_last_question = torch.gather(logits_from_last_question_last_step, -1, x_word_ids_indices).squeeze(-1)
        # print('logits_from_last_question', logits_from_last_question.detach().cpu().numpy().tolist())
        #
        # exit(1)

        # print('state_indices_last_question', state_indices_last_question.shape)
        # print(state_indices_last_question.detach().numpy().tolist())
        # print('logits', logits.shape)
        # exit(1)
        # print('state_indices_last_question', state_indices_last_question.shape)
        # logits_from_last_question_last_step = torch.gather(logits, index=state_indices_last_question, dim=1)

        # l1 and l2 norm of knowledge state
        reg_positions = torch.where(batch_data['x_kt_split_ids'] != 0, 1, 0).unsqueeze(-1).to(self.device)  # [batch_size, seq_len, 1]
        # print('reg_positions', reg_positions.detach().numpy().tolist())
        # exit(1)
        probs = torch.sigmoid(logits)  # [batch_size, seq_len, num_words]
        shifted_probs = torch.roll(probs, 1, dims=1)  # state of the last step

        # print('abs', torch.abs(probs - shifted_probs).shape)
        # print('reg_positions', reg_positions.shape)
        # exit(1)
        # changes per step per skill
        l1_reg = torch.sum(torch.abs(probs - shifted_probs) * reg_positions) / reg_positions.sum() / len(self.word_map)
        l2_reg = torch.sum(torch.square(probs - shifted_probs) * reg_positions) / reg_positions.sum() / len(self.word_map)

        # number of skills with more than 0.05 growth.
        updated_skills = torch.where(torch.abs(probs - shifted_probs) * reg_positions >= 0.05, 1, 0).sum() / reg_positions.sum() / len(self.word_map)

        return {
            'logits_from_cur_step': logits_from_cur_step,
            'logits_from_last_step': logits_from_last_step,
            'logits_from_last_question_last_step': logits_from_last_question,
            'l1_reg': l1_reg, 'l2_reg': l2_reg, 'num_updated_skills': updated_skills
        }

    def joint_train(self):
        kt_pqg_train_dev_dataset = self.build_dataset(
            target_split=[1, 2],
            return_kt=True,
            return_aqg=True,
            qg_use_skills=True,
            qg_use_difficulty=True,
            qg_use_state=True,
        )
        kt_pqg_train_dev_test_dataset = self.build_dataset(
            target_split=[1, 2, 3],
            return_kt=True,
            return_aqg=True,
            qg_use_skills=True,
            qg_use_state=True,
            qg_use_difficulty=True
        )

        kt_pqg_train_dev_dataloader = DataLoader(kt_pqg_train_dev_dataset, batch_size=1, shuffle=True)
        kt_pqg_train_dev_test_dataloader = DataLoader(kt_pqg_train_dev_test_dataset, batch_size=1, shuffle=True)

        for batch_id, batch_data in enumerate(kt_pqg_train_dev_dataloader):
            batch_data = {key: batch_data[key].to(self.device) for key in batch_data}
            kt_logits = self.knowledge_tracer(
                x_user_ids=None,
                x_word_ids=batch_data['x_kt_word_ids'],
                y_labels=batch_data['y_kt_labels']
            )

            kt_predictions = self.collect_kt_predictions(kt_logits, batch_data)
            last_states = torch.sigmoid(kt_logits[batch_data['x_qg_state_positions']])  # [batch_size, num_question, num_words]
            outputs = self.question_generator(
                x_input_ids=batch_data['x_qg_input_ids'],
                x_attention_mask=batch_data['x_qg_attention_masks'],
                difficulty=batch_data['x_adaptive_difficulties'],
                student_state=last_states,
                decoder_input_ids=batch_data['y_qg_decoder_input_ids'],
                labels=batch_data['y_qg_labels']
            )

            qg_gumbel_max = torch.softmax(outputs.logits / self.temperature)  # [batch_size, question_length, vocab_size]

    def build_dataset(self, target_split, return_kt, return_aqg, qg_use_skills, qg_use_difficulty, qg_use_state, qg_use_history=True, max_examples=-1, user_list=None, target_example=None):
        return JointKTQGDataset(
            data_dir=self.args.kt_format_data,
            word_map=self.word_map,
            pos_tag_map=self.pos_tag_map,
            target_split=target_split,  # train:1, dev:2, test:3
            word_sampler=self.word_sampler,
            qg_tokenizer_right=BartTokenizer.from_pretrained(self.args.qg_model_name, truncation_side='right'),
            qg_tokenizer_left=BartTokenizer.from_pretrained(self.args.qg_model_name, truncation_side='left'),
            return_kt=return_kt,
            return_aqg=return_aqg,
            pad_label_id=-100,
            prepare_decoder_input_ids_from_labels=self.prepare_decoder_input_ids_from_labels,
            kt_trunc_direction='left',
            kt_pad_direction='right',
            max_seq_length=1024,
            max_keyword_length=15,
            max_question_length=30,
            qg_use_skills=qg_use_skills,
            qg_use_difficulty=qg_use_difficulty,
            qg_use_state=qg_use_state,
            qg_use_history=qg_use_history,
            max_examples=max_examples,
            target_example=target_example,
            sample_rate=self.args.qg_prompt_words_sample_rate,
            difficulty_bucket=self.difficulty_bucket,
            difficulty_max_label=self.max_difficulty_label,
            d_template=self.d_template,
            d_type=self.d_type,
            d_source=self.d_source,
            user_list=user_list
        )


def query_kt_best_next_skill(args, output_dir, kt_model_file, device, start_point=100, direction='right', min_length=1024):
    dataset = DuolingoKTDataset(
        raw_data_file=None,  # args.duolingo_en_es_format,
        data_dir=args.kt_format_data_1024,
        word_file=args.duolingo_en_es_word_file,
        w_l_tuple_file=args.duolingo_en_es_w_l_tuple_file,
        user_file=args.duolingo_en_es_user_file,
        max_seq_len=args.kt_max_seq_len,
        label_pad_id=int(args.kt_pad_label_id),
        target_split=['train', 'dev', 'test'],
        max_lines=-1,
        discard_rate=args.kt_discard_rate,
        pos_tag_vocab_file=args.duolingo_en_es_pos_tag_file,
    )
    dataloader = DataLoader(dataset, shuffle=False, batch_size=1, collate_fn=dataset.construct_collate_fn(max_seq_len=args.kt_max_seq_len))
    model = DKT(
        input_size=args.dkt_input_size,
        hidden_size=args.dkt_input_size,
        num_layers=args.dkt_num_layers,
        num_tuples=dataset.num_w_l_tuples,
        num_users=dataset.num_users,
        encoder=args.dkt_encoder,
        num_attn_heads=4,
        dropout=0.1,
        device=device,
        max_length=args.kt_max_seq_len,
        num_words=dataset.num_words
    ).to(device)
    kt_model_save_path = os.path.join(args.kt_model_save_dir, kt_model_file)
    save_dict = torch.load(kt_model_save_path)
    # print(type(save_dict))
    # print(save_dict.keys())
    model.load_state_dict(save_dict['model_state_dict'])
    model.eval()

    logging.info('logging knowledge tracer model from {}'.format(kt_model_save_path))
    for batch_id, batch_data in enumerate(dataloader):
        # print(device)
        batch_data = (data.to(device) for data in batch_data)
        (x_user_ascii, x_user_ids, x_user_abilities, x_word_ids, x_word_attn_masks, x_w_l_tuple_ids,
         x_w_l_tuple_attn_masks, x_position_ids, x_task_ids, x_days, x_time, x_interaction_ids, y_labels,
         split_ids, x_valid_lengths, x_valid_interactions) = batch_data
        # (x_user_ascii, x_user_ids, x_user_abilities, x_word_ids, x_word_attn_masks, x_w_l_tuple_ids,
        #  x_w_l_tuple_attn_masks, x_position_ids, x_task_ids, x_days, x_time, x_interaction_ids, y_labels,
        #  split_ids, x_valid_lengths, x_valid_interactions) = (data.to(device) for data in batch_data)
        if x_w_l_tuple_ids.size(1) < min_length:
            continue

        if direction == 'right':
            history_interactions = x_w_l_tuple_ids[:, :x_w_l_tuple_ids.size(1)-start_point]
            history_word_ids = x_word_ids[:, :x_word_ids.size(1)-start_point]
            history_labels = y_labels[:, :y_labels.size(1)-start_point]
            future_interactions = x_w_l_tuple_ids[:, -start_point:]
            future_word_ids = x_word_ids[:, -start_point:]
            future_labels = y_labels[:, -start_point:]
        else:
            history_interactions = x_w_l_tuple_ids[:, :start_point]
            history_word_ids = x_word_ids[:, :start_point]
            history_labels = y_labels[:, :start_point]
            future_interactions = x_w_l_tuple_ids[:, start_point:]
            future_word_ids = x_word_ids[:, start_point:]
            future_labels = y_labels[:, start_point:]

        print(history_word_ids.shape, future_word_ids.shape)
        print(history_labels.shape, future_labels.shape)

        logits = model(x_word_ids=history_word_ids, y_labels=history_labels, x_user_ids=None)
        memory_states = torch.sigmoid(logits)
        original_memory_states = memory_states.clone()
        recommend_word_ids = []
        recommend_word_difficulties = []
        extended_memory_states = []

        for i in range(future_word_ids.size(1)):
            print(i, history_word_ids.shape)
            best_word_id = None
            best_word_difficulty = None
            best_knowledge_state = None
            best_correct_gain = None
            best_wrong_gain = None
            next_label = None
            best_gain = -99999

            for word_id in range(4, dataset.num_words):
                correct_prob = 1 - memory_states[0, -1, word_id]

                word_extended = torch.cat([history_word_ids, torch.tensor([[word_id]]).to(device)], dim=-1)
                correct_label_extended = torch.cat([history_labels, torch.tensor([[0]]).to(device)], dim=-1)
                wrong_label_extended = torch.cat([history_labels, torch.tensor([[1]]).to(device)], dim=-1)

                memory_states_correct = torch.sigmoid(model(x_word_ids=word_extended, y_labels=correct_label_extended, x_user_ids=None))
                memory_states_wrong = torch.sigmoid(model(x_word_ids=word_extended, y_labels=wrong_label_extended, x_user_ids=None))

                correct_gain = torch.sum(memory_states_correct[:, -2, :] - memory_states_correct[:, -1, :])
                wrong_gain = torch.sum(memory_states_wrong[:, -2, :] - memory_states_wrong[:, -1, :])
                expected_gain = correct_prob * correct_gain + (1 - correct_prob) * wrong_gain
                # print('word_id {}, word_difficulty {}, expected_gain {}'.format(word_id, 1-correct_prob, expected_gain))

                if word_id == future_word_ids[0, i]:
                    print('step={}, real next word is {}, difficulty is {}, expected gain={}, correct_gain={}, wrong_gain={}'.format(
                        i, dataset.inverse_word_map[word_id], 1 - correct_prob, expected_gain, correct_gain, wrong_gain
                    ))

                if expected_gain > best_gain:
                    best_gain = expected_gain
                    best_correct_gain = correct_gain
                    best_wrong_gain = wrong_gain
                    best_word_id = word_id
                    next_label = 0 if correct_prob > 0.5 else 1
                    best_word_difficulty = 1 - correct_prob
                    best_knowledge_state = memory_states_correct[0, -1, :].detach().cpu().numpy() if correct_prob > 0.5 else memory_states_wrong[0, -1, :].detach().cpu().numpy()

            recommend_word_ids.append(best_word_id)
            recommend_word_difficulties.append(best_word_difficulty.detach().cpu().numpy().tolist())
            extended_memory_states.append(best_knowledge_state)
            history_word_ids = torch.cat([history_word_ids, torch.tensor([[best_word_id]]).to(device)], dim=-1)
            history_labels = torch.cat([history_labels, torch.tensor([[next_label]]).to(device)], dim=-1)
            print('step={}, best_word={}, best difficulty={}, best_gain={}, best_correct_gain={}, best_wrong_gain={}'.format(
                i, dataset.inverse_word_map[best_word_id], best_word_difficulty, best_gain, best_correct_gain, best_wrong_gain
            ))

        assert len(recommend_word_ids) == start_point
        exit(1)
        save_path = os.path.join(output_dir, '{}.npz'.format('-'.join([str(num) for num in x_user_ascii[0]])))
        simulated_results = {
            'x_user_ascii': x_user_ascii.detach().cpu().numpy,
            'original_memory_states': original_memory_states,
            'recommend_word_ids': np.array(recommend_word_ids),
            'recommend_difficulties': np.array(recommend_word_difficulties),
            'simulated_memory_states': np.array(extended_memory_states)
        }
        np.savez(save_path, **simulated_results)


class NonIndividualizedQGTrainer:
    def __init__(self, args, gpu_cnt, local_rank, device, use_difficulty, use_skill, difficulty_type, d_source,
                 d_template='<d_{}>', max_difficulty_label=4, difficulty_bucket=0.5, use_dev_for_train=False):
        assert difficulty_type in ['continuous', 'discrete']
        self.args = args
        self.gpu_cnt = gpu_cnt
        self.device = device
        self.local_rank = local_rank
        self.use_difficulty = use_difficulty
        self.use_skill = use_skill
        self.use_dev_for_train = use_dev_for_train
        self.do_train = True
        self.do_eval = True
        self.do_predict = True

        self.max_difficulty_label = max_difficulty_label
        self.difficulty_bucket = difficulty_bucket
        self.d_template = d_template
        self.d_type = difficulty_type
        self.d_source = d_source

        self.args.qg_train_batch_size = int(self.args.qg_train_batch_size) // max(1, self.gpu_cnt)
        self.args.qg_eval_batch_size = int(self.args.qg_eval_batch_size) // max(1, self.gpu_cnt)

        self.vocab_difficulty = get_vocab_difficulty(args.duolingo_en_es_word_file)
        self.word_map = get_vocab_difficulty(args.duolingo_en_es_word_file)

        logging.info('-- model name is {}'.format(self.args.qg_model_name))

        self.tokenizer = BartTokenizer.from_pretrained(self.args.qg_model_name)  # begin 0 end 2 <pad> 1
        # self.model = QuestionGenerator(model_name=self.args.qg_model_name, num_words=len(word_map)).to(self.device)

        if self.d_type == 'continuous':
            self.model = ExerciseGeneratorC(model_name=self.args.qg_model_name, num_words=len(self.word_map), use_d=args.d, use_a=args.a).to(self.device)
        else:
            self.model = ExerciseGeneratorD(model_name=self.args.qg_model_name, use_difficulty=self.use_difficulty, max_difficulty_label=self.max_difficulty_label, d_template=self.d_template).to(self.device)
            if self.use_difficulty:
                difficulty_control_tokens = {'additional_special_tokens': [self.d_template.format(i) for i in range(self.max_difficulty_label)]}
                self.tokenizer.add_special_tokens(difficulty_control_tokens)
                self.model.resize_token_embeddings(len(self.tokenizer))

        self.evaluator = QGEvaluator(vocab_difficulty=self.vocab_difficulty)

        logging.info('-- local_rank: {}, loading train dataset'.format(self.local_rank))
        self.train_dataset = NonIndividualizedQGDataset(
            data_file=self.args.duolingo_en_es_non_adaptive_exercise_gen_train,
            tokenizer=self.tokenizer,
            sample_rate=self.args.qg_prompt_words_sample_rate,
            idx=self.args.qg_sample_round,
            use_difficulty=use_difficulty,
            use_skill=use_skill,
            prepare_decoder_input_ids=self.model.generator.prepare_decoder_input_ids_from_labels,
            difficulty_bucket=self.difficulty_bucket,
            difficulty_max_label=self.max_difficulty_label,
            d_template=self.d_template,
            d_type=self.d_type,
            d_source=self.d_source
        )
        # if enable_difficulty:  # input: prompt words + difficulty
        #     difficulty_control_tokens = {'additional_special_tokens': ['<dif_{}>'.format(i) for i in range(4)]}
        #     added_special_token_ids = tokenizer.add_special_tokens(difficulty_control_tokens)
        #     question_generator.resize_token_embeddings(len(tokenizer))
        #     if local_rank == 0:
        #         logging.info('-- added special tokens :{}'.format(
        #             [list(zip(tokenizer.additional_special_tokens, tokenizer.additional_special_tokens_ids))]))

        if self.local_rank == 0:
            for index in random.sample(range(len(self.train_dataset)), 1):
                logging.info('-- {} train data in total, sampled {}th example: {}.'.format(len(self.train_dataset), index, self.train_dataset[index]))

        logging.info('-- local_rank: {}, loading eval dataset'.format(self.local_rank))
        self.eval_dataset = NonIndividualizedQGDataset(
            data_file=self.args.duolingo_en_es_non_adaptive_exercise_gen_dev,
            tokenizer=self.tokenizer,
            sample_rate=self.args.qg_prompt_words_sample_rate,
            idx=self.args.qg_sample_round,
            use_difficulty=self.use_difficulty,
            use_skill=self.use_skill,
            prepare_decoder_input_ids=self.model.generator.prepare_decoder_input_ids_from_labels,
            difficulty_bucket=self.difficulty_bucket,
            difficulty_max_label=self.max_difficulty_label,
            d_template=self.d_template,
            d_type=self.d_type,
            d_source=self.d_source
        )

        if self.local_rank == 0:
            for index in random.sample(range(len(self.eval_dataset)), 1):
                logging.info('-- {} eval data in total, sampled {}th example: {}'.format(len(self.eval_dataset), index, self.eval_dataset[index]))

        logging.info('-- local_rank: {}, loading test dataset'.format(self.local_rank))
        self.test_dataset = NonIndividualizedQGDataset(
            data_file=self.args.duolingo_en_es_non_adaptive_exercise_gen_test,
            tokenizer=self.tokenizer,
            sample_rate=self.args.qg_prompt_words_sample_rate,
            idx=self.args.qg_sample_round,
            use_skill=self.use_skill,
            use_difficulty=self.use_difficulty,
            prepare_decoder_input_ids=self.model.generator.prepare_decoder_input_ids_from_labels,
            difficulty_bucket=self.difficulty_bucket,
            difficulty_max_label=self.max_difficulty_label,
            d_template=self.d_template,
            d_type=self.d_type,
            d_source=self.d_source
        )
        if self.local_rank == 0:
            for index in random.sample(range(len(self.eval_dataset)), 1):
                logging.info('-- {} test data in total, sampled {}th example: {}'.format(len(self.eval_dataset), index, self.eval_dataset[index]))

        if self.gpu_cnt > 1:
            self.model = DDP(self.model, device_ids=[self.local_rank]).module
            train_sampler = DistributedSampler(self.train_dataset)
            self.train_dataloader = DataLoader(self.train_dataset, batch_size=int(args.qg_train_batch_size), sampler=train_sampler)
            eval_sampler = DistributedSampler(self.eval_dataset)
            self.eval_dataloader = DataLoader(self.eval_dataset, batch_size=int(args.qg_eval_batch_size), sampler=eval_sampler)
            test_sampler = DistributedSampler(self.test_dataset)
            self.test_dataloader = DataLoader(self.test_dataset, batch_size=int(args.qg_eval_batch_size), sampler=test_sampler)
        else:
            self.train_dataloader = DataLoader(self.train_dataset, batch_size=int(args.qg_train_batch_size), shuffle=True)
            self.eval_dataloader = DataLoader(self.eval_dataset, batch_size=int(args.qg_eval_batch_size), shuffle=True)
            self.test_dataloader = DataLoader(self.test_dataset, batch_size=int(args.qg_eval_batch_size), shuffle=True)

    def train(self):
        # Train!
        batch_steps = math.ceil(len(self.train_dataset) / int(self.args.qg_train_batch_size) / max(self.gpu_cnt, 1))
        total_steps = int(self.args.qg_num_train_epoch) * batch_steps
        warmup_steps = int(total_steps * float(self.args.qg_warmup_rate))

        optimizer = AdamW(self.model.parameters(), lr=float(self.args.qg_learning_rate))

        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        save_info = {
            'epoch': 0,
            'loss': 0,
            'best_validation_performance': -1,
            'model_state_dict': None,
            'optimizer_state_dict': None
        }

        # if self.use_dev_for_train:
        #     train_set = torch.utils.data.ConcatDataset([self.train_dataset, self.eval_dataset])
        # else:
        #     train_set = self.train_dataset

        logging.info('local_rank {}, start training, total steps {}, warm up steps {}'.format(self.local_rank, total_steps, warmup_steps))
        for epoch_id in range(int(self.args.qg_num_train_epoch)):
            self.model.train()
            epoch_loss = 0

            if self.gpu_cnt > 1:
                self.train_dataloader.sampler.set_epoch(epoch_id)

            for batch_id, batch_data in enumerate(self.train_dataloader):
                # data keys: 'x_target_words',  x_difficulties, 'x_input_ids', 'x_attention_mask', 'y_decoder_input_ids', 'y_labels'
                batch_data = {key: batch_data[key].to(self.device) for key in batch_data}
                # x_input_ids, x_attention_mask, x_difficulties, y_decoder_input_ids, y_labels = batch_data
                optimizer.zero_grad()

                if epoch_id == 0 and batch_id == 0:
                    for key in batch_data:
                        logging.debug('batch_data: {}:{}'.format(key, batch_data[key]))
                exit(1)
                if self.d_type == 'continuous':
                    outputs = self.model(
                        x_input_ids=batch_data['x_input_ids'],
                        x_attention_mask=batch_data['x_attention_mask'],
                        difficulty=batch_data['x_difficulties'] if self.use_difficulty else None,
                        student_state=None,
                        decoder_input_ids=batch_data['y_decoder_input_ids'],
                        labels=batch_data['y_labels']
                    )
                else:
                    outputs = self.model(
                        input_ids=batch_data['x_input_ids'],
                        attention_mask=batch_data['x_attention_mask'],
                        decoder_input_ids=batch_data['y_decoder_input_ids'],
                        labels=batch_data['y_labels']
                    )
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                lr_scheduler.step()

                logging.info('local_rank: {}, {}/{} epoch, {}/{} batch, loss is {}'.format(
                    self.local_rank, epoch_id + 1, self.args.qg_num_train_epoch, batch_id + 1, batch_steps, loss
                ))
                epoch_loss += loss

            # do_eval
            difficulty_scores, prompt_words, generated, reference = self.inference(epoch_id, target_set='dev')

            # compute metrics
            if self.local_rank == 0:
                difficulty_scores = torch.cat(difficulty_scores, dim=0).detach().cpu().numpy().tolist()
                prompt_words = torch.cat(prompt_words, dim=0).cpu()
                generated = torch.cat(generated, dim=0).cpu()
                reference = torch.cat(reference, dim=0).cpu()

                logging.info('-- collect {} data for evaluation'.format(prompt_words.size(0)))

                self.evaluator.reference_difficulty_scores = difficulty_scores
                self.evaluator.prompt_words = self.tokenizer.batch_decode(prompt_words, skip_special_tokens=True)
                self.evaluator.generated = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
                self.evaluator.reference = self.tokenizer.batch_decode(reference, skip_special_tokens=True)

                logging.info('-- local_rank {} computing metrics ...'.format(self.local_rank))
                epoch_result = self.evaluator.compute_metrics()
                logging.info('-- {}/{} epoch, performance on eval set: {}'.format(epoch_id + 1, int(self.args.qg_num_train_epoch), epoch_result))

                validation_performance = (epoch_result['rouge-1'] + epoch_result['rouge-2'] + epoch_result['rouge-l']) / 3

                logging.info('-- {}th epoch, loss is {}, local_rank:{}, validation performance is {}'.format(epoch_id + 1, epoch_loss, self.local_rank, validation_performance))

                if validation_performance > save_info['best_validation_performance']:
                    save_info['epoch'] = epoch_id + 1
                    save_info['loss'] = epoch_loss
                    save_info['best_validation_performance'] = validation_performance
                    save_info['model_state_dict'] = deepcopy(self.model.state_dict())
                    save_info['optimizer_state_dict'] = deepcopy(optimizer.state_dict())
                    save_file = os.path.join(self.args.qg_eval_output, '{}_ep{}_output'.format(self.args.qg_model_name.replace('/', '_'), epoch_id + 1))
                    self.evaluator.output_result(save_file)

            self.evaluator.clear()
            logging.info(' -- local_rank {}, finish evaluation'.format(self.local_rank))

        # save best performing model
        if self.local_rank == 0:
            logging.info('-- local_rank:{}, finish training, best model: {}-th epoch, loss: {}, validation_performance: {}'.format(
                self.local_rank, save_info['epoch'], save_info['loss'], save_info['best_validation_performance']
            ))
            file_name = 'baseline_{}_'.format(int(self.args.qg_prompt_words_sample_rate*100))
            # enable difficulty
            if self.use_difficulty:
                file_name += 'd_{}_'.format(self.d_type[:3])
                file_name += '{}_'.format(self.d_source)  # difficulty source

            # enable words
            if self.use_skill:
                file_name += 's_'
            # model_name
            file_name += self.args.qg_model_name.replace('/', '-')
            # epoch
            file_name += '{}ep.pth'.format(save_info['epoch'])

            save_path = os.path.join(self.args.qg_model_save_dir, file_name)
            logging.info('saving model to {}'.format(save_path))
            torch.save(save_info, save_path)

    def inference(self, epoch_id, target_set):
        inference_dataloader = None
        if target_set == 'train':
            inference_dataloader = self.train_dataloader
        elif target_set == 'dev':
            inference_dataloader = self.eval_dataloader
        elif target_set == 'test':
            inference_dataloader = self.test_dataloader

        eval_batch_steps = int(len(target_set) / self.args.qg_eval_batch_size / max(1, self.gpu_cnt)) + 1
        self.model.eval()

        difficulty_scores = []
        prompt_words = []
        generated = []
        reference = []

        with torch.no_grad():
            if self.gpu_cnt > 1:
                inference_dataloader.sampler.set_epoch(epoch_id)

            for batch_id, batch_data in enumerate(inference_dataloader):
                # data keys: 'x_target_words',  x_difficulties, 'x_input_ids', 'x_attention_mask', 'y_decoder_input_ids', 'y_labels'
                batch_data = {key: batch_data[key].to(self.device) for key in batch_data}

                if epoch_id == 0 and batch_id == 0:
                    for key in batch_data:
                        logging.debug('batch_data: {}:{}'.format(key, batch_data[key]))

                # [batch_size, seq_len]
                if self.d_type == 'discrete':
                    output_ids = self.model.generator.generate(
                        inputs=batch_data['x_input_ids'],
                        attention_mask=batch_data['x_attention_mask'],
                        num_beams=int(self.args.qg_num_beams),
                        max_length=int(self.args.qg_y_max_length),
                        # return_dict_in_generate=True # transformers.generation_utils.BeamSearchEncoderDecoderOutput
                    )
                else:
                    inputs_embeds = self.model.get_inputs_embeds(
                        x_input_ids=batch_data['x_input_ids'],
                        difficulty=batch_data['x_difficulties'] if self.use_difficulty else None,
                        student_state=None
                    )
                    output_ids = self.model.generator.generate(
                        inputs_embeds=inputs_embeds,
                        attention_mask=batch_data['x_attention_mask'],
                        num_beams=int(self.args.qg_num_beams),
                        max_length=int(self.args.qg_y_max_length),
                    )

                # postprocess
                # pad output_ids to collect across devices
                pad_length = int(self.args.qg_y_max_length) - output_ids.size(-1)
                if pad_length > 0:
                    output_ids = F.pad(output_ids, pad=(0, pad_length), value=self.tokenizer.pad_token_id)
                # recover pad labels for tokenizer.decode
                batch_data['y_labels'][batch_data['y_labels'] == int(self.args.qg_label_pad_token_id)] = self.tokenizer.pad_token_id
                # end postprocess

                if self.gpu_cnt > 1:
                    # collect evaluation data across gpus
                    batch_difficulties = [torch.zeros_like(batch_data['x_difficulties']).to(self.device) for i in range(self.gpu_cnt)]
                    batch_input_words = [torch.zeros_like(batch_data['x_target_word_ids']).to(self.device) for i in range(self.gpu_cnt)]
                    batch_reference = [torch.zeros_like(batch_data['y_labels']).to(self.device) for i in range(self.gpu_cnt)]
                    batch_generated = [torch.zeros_like(output_ids).to(self.device) for i in range(self.gpu_cnt)]

                    batch_data['y_labels'][batch_data['y_labels'] == -100] = self.tokenizer.pad_token_id
                    dist.all_gather(batch_difficulties, batch_data['x_difficulties'])
                    dist.all_gather(batch_input_words, batch_data['x_target_word_ids'])  # [batch_size, input_len] * gpu_cnt
                    dist.all_gather(batch_reference, batch_data['y_labels'])
                    dist.all_gather(batch_generated, output_ids)

                    difficulty_scores.extend(batch_difficulties)
                    prompt_words.extend(batch_input_words)
                    reference.extend(batch_reference)
                    generated.extend(batch_generated)
                else:
                    # collect results on a single card
                    difficulty_scores.extend(batch_data['x_difficulties'])
                    prompt_words.append(batch_data['x_target_word_ids'])
                    generated.append(output_ids)
                    reference.append(batch_data['y_labels'])

                logging.info('-- local_rank {}, evaluating {}/{} batch, {} batch data_collected'.format(self.local_rank, batch_id + 1, eval_batch_steps, len(prompt_words)))

        return difficulty_scores, prompt_words, generated, reference

    def load_model(self, model_save_path):
        save_dict = torch.load(model_save_path)
        self.model.load_state_dict(save_dict)

