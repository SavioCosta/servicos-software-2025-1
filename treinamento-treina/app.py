from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse

app = FastAPI()

def treina():
    #Carga do dataset
    iris = datasets.load_iris()
    X= iris.data 
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)

    clf = RandomForestClassifier()

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

    joblib.dump(clf, 'iris_model.pkl')
    print("Model saved!")


@app.get("/treina")
async def main():
    treina()
    return JSONResponse({"Criado: iris_model.pkl"})

@app.get("/download")
async def download():
    return FileResponse('iris_model.pkl')