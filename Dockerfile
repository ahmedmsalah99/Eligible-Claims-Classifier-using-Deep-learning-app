FROM tiangolo/uvicorn-gunicorn-fastapi:python3.10
# FROM python:3.10.10
RUN pip install --upgrade pip

COPY ./requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt
RUN apt-get install unzip

COPY ./app /app/app
RUN unzip /app/app/quantized_setfitonnx_model.zip -d /app/app
RUN rm -rf /app/app/quantized_setfitonnx_model.zip

# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]