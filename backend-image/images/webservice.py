import os
import io
import subprocess
from os import path
import tensorflow as tf
import keras as keras
from tensorflow.keras import layers
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D
from PIL import Image
import numpy as np
from fastapi import FastAPI, File, UploadFile, Query, applications
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import StreamingResponse, RedirectResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.encoders import jsonable_encoder

app = FastAPI()

inception_net = tf.keras.applications.MobileNetV2()
import requests

response = requests.get("https://git.io/JJkYN")
labels = response.text.split("\n")

assets_path = os.getcwd() + "/swagger-ui-assets"
if path.exists(assets_path + "/swagger-ui.css") and path.exists(assets_path + "/swagger-ui-bundle.js"):
    app.mount("/assets", StaticFiles(directory=assets_path), name="static")


    def swagger_monkey_patch(*args, **kwargs):
        return get_swagger_ui_html(
            *args,
            **kwargs,
            swagger_favicon_url="",
            swagger_css_url="/assets/swagger-ui.css",
            swagger_js_url="/assets/swagger-ui-bundle.js",
        )


    applications.get_swagger_ui_html = swagger_monkey_patch

def bloco_residual(X, n, dim=(5, 5)):
    """`
    Entradas:
        X = tensor de entrada
        n = número de filtros
        dim = dimensão dos filtros

    Saída:
        A2 -> tensor de saída
    """

    # Configuraçao do bloco
    # Inclua o seu código aqui
    A1 = layers.Conv2D(n, dim, strides=1, padding='same', activation='relu')(X)
    Z2 = layers.Conv2D(n, dim, strides=1, padding='same', activation='linear')(A1)
    ZX = layers.Add()([Z2, X])
    A2 = layers.Activation('relu')(ZX)

    # Retorna saída
    return A2

def build_model():
  filter_size = (5, 5)
  input_shape = (64, 64, 3)
  X0 = layers.Input(shape=input_shape)

  # Inclua seu código aqui
  # Camada convolucional para ajustar número de canais para poder ser somada dentro do bloco residual
  X1 = layers.Conv2D(128, filter_size, strides=1, padding='same', activation='relu')(X0)

  # Primeiro bloco residual com 128 filtros
  X2 = bloco_residual(X1,128)

  # Camada convolucional para ajustar número de canais para poder ser somada dentro do bloco residual
  X3 = layers.Conv2D(256, filter_size, strides=1, padding='same', activation='relu')(X2)

  # Segundo bloco residual com 256 filtros
  X4 = bloco_residual(X3,256)

  # Terceiro bloco residual com 256 filtros
  X5 = bloco_residual(X4,256)

  # Camada convolucional para ajustar número de canais para poder ser somada dentro do bloco residual
  X6 = layers.Conv2D(128, filter_size, strides=1, padding='same', activation='relu')(X5)

  # Quarto bloco residual com 128 filtros
  X7 = bloco_residual(X6,128)

  # Quinto bloco residual com 128 filtros
  X8 = bloco_residual(X7,128)

  # Camada convolucional para ajustar número de canais para poder ser somada dentro do bloco residual
  X9 = layers.Conv2D(64, filter_size, strides=1, padding='same', activation='relu')(X8)

  # Sexto bloco residual com 64 filtros
  X10 = bloco_residual(X9,64)

  # Camada convolutional para acertar profundidade da imagem resultante no padrão RGB
  Y = layers.Conv2D(3, filter_size, strides=1, padding='same', activation='linear')(X10)


  # Criação da RNA
  # Inclua seu código aqui
  rna = Model(X0, Y)

  return rna

@app.get("/", response_class=RedirectResponse, include_in_schema=False)
async def index():
    return "/docs"

@app.post("/build_nose",
          tags=["Endpoints"], 
          responses = { 
              200 : { 
                  "content": {"image/png" : {}}
                    } 
                    },
                    response_class=Response)
async def build_nose(
    image_file: UploadFile = File(...)):
    image = Image.open(image_file.file)
    image = image.resize((64, 64))
    inp = np.array(image) / 255
    inp = inp.reshape((-1, 64, 64, 3))
    
    
    build_nose_model = build_model()
    build_nose_model.load_weights(os.getcwd() + "/images/model/model.weights.h5")
    
    prediction = build_nose_model.predict(inp).reshape((64, 64, 3))
    pixels = (prediction * 255).astype(np.uint8)
    img = Image.fromarray(pixels, 'RGB')
    img.save(os.getcwd() + "/images/model/sample.png")
    image_bytes = io.BytesIO()
    img.save(image_bytes, format='PNG')
    image_bytes.seek(0)
    buffer = image_bytes.read()


    return Response(
        content = buffer, media_type="image/png"
        #content = prediction.tolist(), media_type="text/json"
    )

