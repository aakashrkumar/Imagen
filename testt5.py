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

inputs = tokenizer(text[0], return_tensors="np")
outputs = model.encode(**inputs, return_dict=True, output_attentions=True)
print(outputs["last_hidden_state"].shape)
print(outputs["attentions"][-1].shape)


# def pool_sequences(input_id):

def get_tokenizer_and_model():
    name = "t5-large"
    tokenizer = T5Tokenizer.from_pretrained(name)
    model = FlaxT5ForConditionalGeneration.from_pretrained(name)

def encode_text(text, tokenizer, model):
    max_sequence_length = 512
    max_target_length = 128
    encoding = tokenizer(
        text,
        padding="longest", 
        max_length=max_sequence_length, 
        truncation=True, 
        return_tensors="np")
    
    input_ids, attention_mask = encoding.input_ids, encoding.attention_mask

    outputs = model.encode(
        input_ids=input_ids, 
        attention_mask=attention_mask,
        return_dict=True,
        output_attentions=False)

    return outputs, attention_mask