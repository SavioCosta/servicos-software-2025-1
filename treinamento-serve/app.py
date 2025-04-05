from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.staticfiles import StaticFiles
import aiofiles
import os
from os import path

app = FastAPI()

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


@app.get("/", response_class=RedirectResponse, include_in_schema=False)
async def index():
    return "/docs"

@app.get("/predicao")
def usa_modelo():
    #Carga do dataset
    iris = datasets.load_iris()
    X= iris.data 
    y = iris.target
    clf = joblib.load('/ml/iris_model.pkl')

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)

    y_pred = clf.predict(X_test)
    resposta= (f"Accuracy: {accuracy_score(y_test, y_pred)}")
    json = "{'resultado':'"+ resposta + "'}"

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        async with aiofiles.open(path.join("ml", file.filename), 'wb') as f:
            await f.write(contents)
        return JSONResponse(content={"message": "Arquivo salvo"})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
        await file.close()
    