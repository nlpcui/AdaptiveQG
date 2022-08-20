# bsub -n 8 -J bart_qg_train -R "rusage[mem=2048,ngpus_excl_p=1]" -W 24:00 -R "select[gpu_model0==NVIDIATITANRTX]" sh run.sh
/cluster/project/sachan/pencui/anaconda3/bin/python -u main.py