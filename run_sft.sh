python -u train.py model=pythia69 datasets=[hh] loss=sft exp_name=anthropic_dpo_pythia69 gradient_accumulation_steps=2 batch_size=64 eval_batch_size=32 trainer=FSDPTrainer sample_during_eval=false
