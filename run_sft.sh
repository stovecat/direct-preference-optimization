cuda=4,5,6,7
CUDA_VISIBLE_DEVICES=$cuda python -u train.py model=santacoder1b datasets=[cctg] loss=sft exp_name=cctg_sft_santacoder1b gradient_accumulation_steps=2 batch_size=64 eval_batch_size=32 trainer=FSDPTrainer sample_during_eval=false
