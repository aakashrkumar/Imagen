import ray
from ray import serve
from ray.serve.gradio_integrations import GradioIngress

import gradio as gr

from transformers import pipeline

from ray.serve.gradio_integrations import GradioServer

generator1 = pipeline("text-generation", model="gpt2")


def model1(text):
    generated_list = generator1(text, do_sample=True, min_length=20, max_length=100)
    generated = generated_list[0]["generated_text"]
    return generated

demo = gr.Interface(
    lambda text: f"{model1(text)}}", "textbox", "textbox"
)
demo.launch(share=True)
