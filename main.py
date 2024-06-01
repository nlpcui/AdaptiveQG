import argparse
import configparser
import logging
import torch
import os
from train import JointKTQGTrainer, NonIndividualizedQGTrainer
from data import auto_convert
import numpy as np
import random
import torch.distributed as dist


D_TEMPLATE = '<d_{}>'


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='config/euler_conf.ini')
    parser.add_argument('--job', type=str, default='eval_qg')
    # control parameters
    parser.add_argument('--d', type=bool, default=True)   # enable difficulty
    parser.add_argument('--h', type=bool, default=False)  # enable history
    parser.add_argument('--s', type=bool, default=True)   # enable knowledge state
    parser.add_argument('--a', type=bool, default=True)   # enable skill (words)
    parser.add_argument('--d_type', type=str, default='continuous')  #
    parser.add_argument('--d_source', type=str, default='kt')  # kt, gd
    parser.add_argument('--decoding', type=str, default='normal')  # normal, constrained
    parser.add_argument('--joint', type=bool, default=False)
    parser.add_argument('--inc', type=bool, default=False)
    parser.add_argument('--temperature', type=float, default=2.0)

    args, remaining_argv = parser.parse_known_args()
    config = configparser.ConfigParser()
    config.read(args.conf)

    for section in config:
        if section == 'DEFAULT':
            continue
        for option in config.options(section):
            value = auto_convert(config.get(section, option))
            parser.add_argument('--{}'.format(option), default=value, type=type(value))

    args = parser.parse_args(remaining_argv)

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        # filename='train_kt.log',  # 'kt_log_reg/train_kt_0.5_0.log',  # '{}.log'.format(args.job),  # args.kt_train_log,
        level=logging.INFO,
        filemode='w'
    )

    setup_seed(args.random_seed)

    gpu_cnt = torch.cuda.device_count()  # gpu_cnt_per_machine
    global_rank = 0
    local_rank = 0

    if gpu_cnt == 0:
        logging.info('-- using cpu')
        device = torch.device('cpu')
    elif gpu_cnt == 1:
        logging.info('-- using single gpu: {}'.format(torch.cuda.get_device_name(0)))
        device = torch.device('cuda:0')
        logging.info('cuda visible devices {}'.format(os.environ["CUDA_VISIBLE_DEVICES"]))
    else:
        local_rank = int(os.environ['LOCAL_RANK'])
        device = torch.device('cuda:{}'.format(local_rank))
        torch.cuda.set_device(local_rank)
        dist.init_process_group('nccl', init_method='env://', world_size=gpu_cnt, rank=local_rank)
        # global_rank = dist.get_rank() # process_id

        if local_rank == 0:
            logging.info('cuda visible devices {}'.format(os.environ["CUDA_VISIBLE_DEVICES"]))
            logging.info('-- available gpus: {}'.format([torch.cuda.get_device_name(i) for i in range(gpu_cnt)]))
            # logging.info('-- total_train_batch_size is {}, per_device_train_batch_size is {}'.format(args.qg_train_batch_size*gpu_cnt, args.qg_train_batch_size))

    if args.job == 'train_qg':
        if args.h or args.a:
            qg_trainer = JointKTQGTrainer(
                train_args=args, device=device, gpu_cnt=gpu_cnt, local_rank=local_rank, max_difficulty_label=4,
                difficulty_bucket=0.5, d_template=D_TEMPLATE, use_difficulty=args.d, d_type=args.d_type,
                use_ability=args.a, d_source=args.d_source
            )
            qg_trainer.load_knowledge_tracer(model_save_path='kt_rnn_nw1_nq0.5_l10.1_l20.1_9ep.pth')
            qg_trainer.train_qg(
                use_difficulty=args.d,
                use_skill=args.s,
                use_state=args.a,
                use_history=args.h,
                max_examples=100,
                inner_batch=64,
                min_history=0,
                joint_train=args.joint,
                joint_start=3,
                inc=args.inc,
                temperature=args.temperature
            )
        else:
            qg_trainer = NonIndividualizedQGTrainer(
                args=args,
                gpu_cnt=gpu_cnt,
                device=device,
                local_rank=local_rank,
                use_difficulty=args.d,
                use_skill=args.s,
                use_dev_for_train=False,
                max_difficulty_label=4,
                difficulty_bucket=0.5,
                difficulty_type=args.d_type,
                d_source=args.d_source
            )
            qg_trainer.train()
    elif args.job == 'train_kt':
        kt_trainer = JointKTQGTrainer(train_args=args, device=device, gpu_cnt=gpu_cnt, local_rank=local_rank, max_difficulty_label=4, difficulty_bucket=0.5,
                                      d_template=D_TEMPLATE, use_difficulty=args.d, d_type=args.d_type, use_ability=args.a, d_source=args.d_source)
        kt_trainer.train_kt(use_dev=True, save_result=False)
    elif args.job == 'eval_kt':
        kt_trainer = JointKTQGTrainer(train_args=args, device=device, gpu_cnt=gpu_cnt, local_rank=local_rank, max_difficulty_label=4, difficulty_bucket=0.5,
                                      d_template=D_TEMPLATE, use_difficulty=args.d, d_type=args.d_type, use_ability=args.a, d_source=args.d_source)
        kt_trainer.load_knowledge_tracer(model_save_path='kt_rnn_nw1_nq0.5_l10.1_l20.1_5ep.pth')
        kt_trainer.eval_kt(save_result=True)

    elif args.job == 'eval_qg':
        joint_trainer = JointKTQGTrainer(train_args=args, device=device, gpu_cnt=gpu_cnt, local_rank=local_rank, max_difficulty_label=4, difficulty_bucket=0.5,
                                         d_template=D_TEMPLATE, use_difficulty=args.d, d_type=args.d_type, use_ability=args.a, d_source=args.d_source)
        joint_trainer.load_knowledge_tracer(model_save_path='kt_rnn_nw1_nq0.5_l10.1_l20.1_9ep.pth')
        joint_trainer.load_question_generator(
            model_save_path='baseline_25_d_s_a_joint_inc_facebook-bart-base_best.pth',
        )

        joint_trainer.test_qg(
            max_examples=-1,
            use_difficulty=args.d,
            use_skills=args.s,
            use_state=args.a,
            use_history=args.h
        )