cuda=0,1,2,3,4,5,6,7
cache_dir=.cache/root
sft_ckpt_path=$cache_dir/cctg_sft_santacoder1b_2024-03-05_10-10-16_473711/step-59904
CUDA_VISIBLE_DEVICES=$cuda python -u train.py model=santacoder1b datasets=[cctg] loss=dpo loss.beta=0.1 lr=5e-6 model.archive=$sft_ckpt_path/policy.pt exp_name=cctg_dpo_santacoder1b gradient_accumulation_steps=2 batch_size=32 eval_batch_size=32 trainer=FSDPTrainer sample_during_eval=false
