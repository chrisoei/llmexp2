import torch
import transformers

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
    "facebook/galactica-1.3b",
    "Salesforce/codegen-2B-mono",
    "Salesforce/codegen-6B-mono",
    "facebook/galactica-6.7b",
    "Salesforce/codegen-16B-mono"
]

for m1 in modellist:
    print(m1)
    x1 = hf(m1)
    print(infer(x1, "def helloworld():"))
