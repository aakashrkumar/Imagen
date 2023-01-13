from transformers import T5Tokenizer, FlaxT5ForConditionalGeneration
import numpy as np
import jax
import tqdm

max_sequence_length = 512

def get_tokenizer_and_model():
    name = "t5-large"
    tokenizer = T5Tokenizer.from_pretrained(name)
    model = FlaxT5ForConditionalGeneration.from_pretrained(name)
    return tokenizer, model

def tokenize_text(text, tokenizer):
    assert tokenizer is not None
    encoding = tokenizer(
        text,
        padding="max_length", #longest to match largest input
        max_length=max_sequence_length, 
        truncation=True, 
        return_tensors="np")
    return encoding.input_ids, encoding.attention_mask

@jax.jit
def encode_text(input_ids, attention_mask, model):
    assert model is not None     
    
    outputs = model.encode(
        input_ids=input_ids, 
        attention_mask=attention_mask,
        return_dict=True,
        output_attentions=False)
    
    return outputs[0], attention_mask

def test():
    tokenizer, model = get_tokenizer_and_model()
    text = "This is a test"
    for i in tqdm.tqdm(range(100)):
        input_ids, attention_mask = tokenize_text(text, tokenizer)
        encoded, attention_mask = encode_text(input_ids, attention_mask, model)
if __name__ == "__main__":
    test()