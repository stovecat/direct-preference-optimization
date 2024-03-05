import argparse
import torch
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from utils import get_local_dir, get_local_run_dir, disable_dropout, init_distributed, get_open_port
from tqdm import tqdm
import json


from utils import *

precision = torch.bfloat16

cache_dir = '.cache/root'


def get_description(prompt):
    lines = prompt.split("\ndef ")[1].split(">>> ")[0].split('\n')[1:]
#     lines = prompt.split("\ndef ")[1].split('\n')[1:]
    
    description = ''
    for l in lines:
        l = l.replace('"""', '').replace("'''", "").strip()
        if l == '':
            continue
#         print("# "+l)
        description += "# "+l+'\n'
    
    description += "\n# 0th test case\n"
    description += "INPUT = "
    
    return description


if __name__ == '__main__':
    
    # Config
    ckpt_path = f"{cache_dir}/cctg_sft_santacoder1b_2024-03-04_12-58-52_242520/step-134400"
    model_name = "bigcode/santacoder"
    max_input_len = 512
    max_new_tokens = 512
    benchmark_name = "openai_humaneval"
    device = 2

    
    # Initialize model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, low_cpu_mem_usage=True, torch_dtype=precision, trust_remote_code=True)
    disable_dropout(model)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model.to(device)
    model.eval()

    
    # Load ckpt
    state_dict = torch.load(f"{ckpt_path}/policy.pt", map_location='cpu')
    
    step, metrics = state_dict['step_idx'], state_dict['metrics']
    print(f'loading pre-trained weights at step {step} from {ckpt_path} with metrics {json.dumps(metrics, indent=2)}')
    model.load_state_dict(state_dict['state'])
    
    
    # Load benchmark
    from datasets import load_dataset
    data = load_dataset(benchmark_name, split='test', cache_dir=cache_dir)
    
    
    # Inference
    results = []
    with torch.no_grad():
        for i in tqdm(range(len(data))):
            _prompt = data[i]["prompt"]
            _input = get_description(_prompt)

            # Truncate from rightside
            tokenizer.truncation_side='right'
            input_tensor = tokenizer(_input, truncation=True, 
                                         max_length=max_input_len, 
                                         return_tensors="pt").to(device)


            completion = model.generate(**input_tensor, max_new_tokens=max_new_tokens) #, top_p=0.95, temperature=0.8)

            _preds = [tokenizer.convert_ids_to_tokens(l) for l in completion.cpu().numpy()[:,-max_new_tokens:]]

            preds = [tokenizer.convert_tokens_to_string(p) for p in _preds]

            tmp_dict = {key: val for key,val in data[i].items()}
            tmp_dict['prediction'] = preds
            results.append(tmp_dict)

            
    save_path = f"{ckpt_path}/{benchmark_name}.pkl"
    dump_pkl(results, save_path)