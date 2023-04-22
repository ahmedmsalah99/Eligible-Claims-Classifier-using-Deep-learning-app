FROM tiangolo/uvicorn-gunicorn-fastapi:python3.10
RUN pip install --upgrade pip

COPY ./requirements.txt /app/requirements.txt
COPY ./setup.py /setup.py
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt
RUN pip install -U nltk
RUN python /setup.py
RUN apt-get install unzip

COPY ./app /app/app
RUN unzip /app/app/quantized_setfitonnx_model.zip -d /app/app
RUN rm -rf /app/app/quantized_setfitonnx_model.zip
