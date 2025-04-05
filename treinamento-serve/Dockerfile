FROM python:3.10

WORKDIR /ml

COPY . .

RUN pip install --no-cache-dir --upgrade -r /ml/requirements.txt

EXPOSE 7000

CMD gunicorn --bind 0.0.0.0:7000 --workers 1 --timeout 0 app:app -k uvicorn.workers.UvicornWorker
