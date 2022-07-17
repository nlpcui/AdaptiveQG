# bsub -n 8 -J job_name -R "rusage[mem=2048,ngpus_excl_p=1]" -W 24:00 -R "select[gpu_model0==NVIDIAA100_PCIE_40GB]" ./my_cuda_program
~/anaconda3/bin/python -u main.py > train_log.txt 