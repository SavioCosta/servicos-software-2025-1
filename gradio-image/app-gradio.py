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

#ui = gr.Interface(fn=envia, inputs=gr.Image(type="filepath", width=256, height=256),outputs=gr.Image(type="pil", width=256, height=256))
with gr.Blocks() as ui:
    with gr.Row() as header:
        gr.Markdown(
            """
            # <span style='text-align:center'>Gerador de narizes :nose:</span>
            ### <span style='text-align:center'>Saiu com pressa e esqueceu o nariz em casa?</span>
            ### <span style='text-align:center'>Seus problemas acabaram!!!</span>
            <span style='text-align:center'>Entre na cabine da esquerda e seja teletransportado para a direita com o nariz novo!</span>
            """)
            
    with gr.Row() as body:
        with gr.Column() as col1:
            inputs = gr.Image(type="filepath", width=256, height=256)
            btn = gr.Button(value="Gerar Nariz")
        with gr.Column() as col2:
            outputs = gr.Image(type="pil", width=256, height=256)
    
    btn.click(fn=envia, inputs=inputs,outputs=outputs)

if __name__ == "__main__":
    ui.launch(server_name="0.0.0.0",server_port=8080, show_api=False)
