import T5utils
import torch

from transformers import T5Tokenizer, FlaxT5ForConditionalGeneration

text = [
    "there are more than 5000 different species of potatoes",
    "python is the meaning of life the universe and everything"
]

name = "t5-large"
max_len = 512

tokenizer = T5Tokenizer.from_pretrained(name, model_max_length=max_len)
model = FlaxT5ForConditionalGeneration.from_pretrained(name)

encoding = tokenizer(
    text,
    padding="max_length", # do longest to pad to largest in batch
    max_target_length=max_len, # or max_length
    truncation=True,
    return_tensors="np"
)

input_ids, attention_mask = encoding.input_ids, encoding.attention_mask

outputs = model.encode(
    input_ids=input_ids,
    attention_mask=attention_mask,
    return_dict=False,
    output_attentions=False
)

print(len(outputs))
print(outputs[0].shape)
print(attention_mask.shape)
