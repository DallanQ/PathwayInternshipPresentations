import os
import io
from IPython.display import Image, display, HTML
from PIL import Image
import base64 
from dotenv import load_dotenv, find_dotenv
from transformers import pipeline
import gradio as gr


_ = load_dotenv(find_dotenv()) # read local .env file

os.environ["TOKENIZERS_PARALLELISM"] = "false"

get_completion = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")


def summarize(input):
    output = get_completion(input)
    return output[0]['summary_text']


demo = gr.Interface(fn=summarize, 
                    inputs=[gr.Textbox(label="Text to summarize", lines=6)],
                    outputs=[gr.Textbox(label="Result", lines=3)],
                    title="Text summarization with distilbart-cnn",
                    description="Summarize any text using the `shleifer/distilbart-cnn-12-6` model under the hood!"
                   )

if __name__ == "__main__":
    demo.launch()
