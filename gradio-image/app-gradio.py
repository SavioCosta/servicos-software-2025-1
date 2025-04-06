import gradio as gr
import requests
from PIL import Image
import io

def envia(imagem):
    url="http://backend-image:8081/build_nose/"
    with open(imagem,'rb') as f:
        r = requests.post(url, files={"image_file":f}, stream=True)
        pilImage = Image.open(io.BytesIO(r.content))
        return pilImage

ui = gr.Interface(fn=envia, inputs=gr.Image(type="filepath", width=256, height=256),outputs=gr.Image(type="pil", width=256, height=256))

if __name__ == "__main__":
    ui.launch(server_name="0.0.0.0",server_port=8080, show_api=False)
