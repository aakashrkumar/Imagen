from transformers import T5Tokenizer, FlaxT5ForConditionalGeneration
import jax

def get_tokenizer_and_model():
    name = "t5-large"
    tokenizer = T5Tokenizer.from_pretrained(name)
    model = FlaxT5ForConditionalGeneration.from_pretrained(name)
    return tokenizer, model


def encode_text(text, tokenizer, model):
    if tokenizer is None or model is None:
        return None, None
    
    max_sequence_length = 512
    encoding = tokenizer(
        text,
        padding="max_length", #longest to match largest input
        max_length=max_sequence_length, 
        truncation=True, 
        return_tensors="np")
    
    input_ids, attention_mask = encoding.input_ids, encoding.attention_mask

    outputs = model.encode(
        input_ids=input_ids, 
        attention_mask=attention_mask,
        return_dict=True,
        output_attentions=False)

    return outputs[0], attention_mask
