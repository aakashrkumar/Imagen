import T5utils
import torch

from transformers import T5Tokenizer, FlaxT5ForConditionalGeneration

# name = T5utils.DEFAULT_T5_NAME
# model, tokenizer = T5utils.get_model_and_tokenizer(name)

text = [
    "there are more than 5000 different species of potatoes",
    "python is the meaning of life the universe and everything"
]

# outputs = T5utils.t5_encode_text(text, name)
# print(outputs.shape)
# print(torch.device)

name = "t5-large"

tokenizer = T5Tokenizer.from_pretrained(name)
model = FlaxT5ForConditionalGeneration.from_pretrained(name)
# model.model_max_length = 512

text_tokens = tokenizer(text[0], return_tensors="np")
outputs = model.encode(**text_tokens, return_dict = False)
print(outputs[0].shape)

text = outputs[0].numpy()
