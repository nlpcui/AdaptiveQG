# bsub -n 8 -J bart_qg_train -R "rusage[mem=2048,ngpus_excl_p=1]" -W 24:00 -R "select[gpu_model0==NVIDIATITANRTX]" sh run.sh
rm train.log
python -u -m torch.distributed.launch --use_env --nproc_per_node=4 main.py
