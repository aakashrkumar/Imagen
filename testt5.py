import T5utils

name = T5utils.DEFAULT_T5_NAME
model, tokenizer = T5utils.get_model_and_tokenizer(name)

text = [
    "there are more than 5000 different species of potateos"
]

outputs = T5utils.t5_encode_text(text, name)
print(outputs.shape)