#!/usr/bin/env python3

import os
import json
import time
import torch
import transformers

# See https://stackoverflow.com/questions/58216000/get-total-amount-of-free-gpu-memory-and-available-using-pytorch
def cudastats():
    s = 1.0e9
    r = {
        "total_memory": torch.cuda.get_device_properties(0).total_memory / s,
        "reserved_memory": torch.cuda.memory_reserved(0) / s,
        "allocated_memory": torch.cuda.memory_allocated(0) / s
    }
    r["free_memory"] = r["reserved_memory"] - r["allocated_memory"]
    return r

def hf(m1):
    return {
        "tokenizer": transformers.AutoTokenizer.from_pretrained(m1),
        "model": transformers.AutoModelForCausalLM.from_pretrained(
            m1,
            device_map="auto",
            offload_folder="/tmp/offload",
            torch_dtype=torch.float16
        )
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
    "/home/c/output/llmexp2",
    "facebook/galactica-125m"
]

print("CUDA: {}".format(torch.cuda.is_available()))
print("bfloat16: {}".format(torch.cuda.is_bf16_supported()))
for m1 in modellist:
    j1 = {}
    j1["model"] = m1
    j1["bfloat16"] = torch.cuda.is_bf16_supported()
    j1["cuda_version"] = torch.version.cuda
    print(m1)
    x1 = hf(m1)
    print(cudastats())
    j1["total_memory"] = torch.cuda.get_device_properties(0).total_memory
    j1["reserved_memory"] = torch.cuda.memory_reserved(0)
    j1["allocated_memory"] = torch.cuda.memory_allocated(0)
    t0 = time.time()
    j1["prompt"] = "hope:"
    j1["result"] = infer(x1, j1["prompt"])
    print(j1["result"])
    t1 = time.time()
    j1["time"] = t1 - t0
    print("Time spent inferring: ", t1 - t0)
    torch.cuda.empty_cache()

