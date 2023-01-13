from functools import partial
from transformers import T5Tokenizer, FlaxT5ForConditionalGeneration
import numpy as np
import jax
import tqdm
from flax import struct, jax_utils

max_sequence_length = 512
name = "t5-large"
def get_tokenizer_and_model():
    tokenizer = T5Tokenizer.from_pretrained(name)
    model = FlaxT5ForConditionalGeneration.from_pretrained(name)
    return tokenizer, model

def tokenize_texts(text, tokenizer):
    assert tokenizer is not None
    encoding = tokenizer(
        text,
        padding="max_length", #longest to match largest input
        max_length=max_sequence_length, 
        truncation=True, 
        return_tensors="np")
    return encoding.input_ids, encoding.attention_mask

@partial(jax.pmap, in_axes=(0, 0, None))
def encode_texts(input_ids, attention_mask, model):
    assert model is not None     
    
    outputs = model.encode(
        input_ids=input_ids, 
        attention_mask=attention_mask,
        return_dict=True,
        output_attentions=False)
    
    return outputs[0], attention_mask

def test():
    tokenizer, model = get_tokenizer_and_model()
    text = ["This is a test"] * 1024
    model = jax_utils.replicate(model)
    for i in tqdm.tqdm(range(100)):
        input_ids, attention_mask = tokenize_texts(text, tokenizer)
        input_ids = np.array(input_ids).reshape(8, -1, 512)
        attention_mask = np.array(attention_mask).reshape(8, -1, 512)
        encoded, attention_mask = encode_texts(input_ids, attention_mask, model)
        print(encoded.shape)
        encoded = np.array(encoded).reshape(-1, 512, 1024)
        attention_mask = np.array(attention_mask).reshape(-1, 512)
        
if __name__ == "__main__":
    test()