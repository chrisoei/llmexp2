#!/usr/bin/env python3

import time
import torch
import transformers

# See https://stackoverflow.com/questions/58216000/get-total-amount-of-free-gpu-memory-and-available-using-pytorch
def cudastats():
    return {
        "total_memory": torch.cuda.get_device_properties(0).total_memory,
        "reserved_memory": torch.cuda.memory_reserved(0),
        "allocated_memory": torch.cuda.memory_allocated(0)
    }

def hf(m1):
    return {
        "tokenizer": transformers.AutoTokenizer.from_pretrained(m1),
        "model": transformers.AutoModelForCausalLM.from_pretrained(
            m1,
            device_map="auto",
            offload_folder="/tmp/offload")
    }

def infer(x, prompt1):
    inputs1 = x["tokenizer"](prompt1, return_tensors="pt")\
        .input_ids\
        .to("cuda")
    sample1 = x["model"].generate(
        inputs1, 
        do_sample=True,
        max_new_tokens=1024,
# https://discuss.huggingface.co/t/setting-pad-token-id-to-eos-token-id-50256-for-open-end-generation/22247/3
        pad_token_id=50256,
        temperature=0.9,
        repetition_penalty=1.2
    )
    output1 = x["tokenizer"].decode(sample1[0], skip_special_tokens=True)
    return output1

modellist = [
    "facebook/galactica-125m",
    "Salesforce/codegen-350M-mono",
    "bigscience/bloom-560m",
    "bigscience/bloomz-560m",
    "facebook/galactica-1.3b",
    "Salesforce/codegen-2B-mono",
    "Salesforce/codegen-6B-mono",
    "facebook/galactica-6.7b",
    "Salesforce/codegen-16B-mono",
    "facebook/galactica-30b"
]

for m1 in modellist:
    print(m1)
    x1 = hf(m1)
    print(cudastats())
    t0 = time.time()
    print(infer(x1, "def helloworld():"))
    t1 = time.time()
    print("Time spent inferring: ", t1 - t0)

