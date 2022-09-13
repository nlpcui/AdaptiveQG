# bsub -n 8 -J bart_qg_train -R "rusage[mem=2048,ngpus_excl_p=1]" -W 24:00 -R "select[gpu_model0==NVIDIATITANRTX]" sh run.sh
rm /cluster/work/sachan/pencui/ProjectsData/AdaptiveQG/kt/train.log
python main.py
# python -m torch.distributed.launch --use_env --nproc_per_node=8 main.py
