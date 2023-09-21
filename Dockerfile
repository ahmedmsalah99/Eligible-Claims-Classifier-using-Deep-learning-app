FROM tiangolo/uvicorn-gunicorn-fastapi:python3.10
# RUN pip install --upgrade pip


COPY ./app /app/app

RUN pip install -U --no-cache-dir --upgrade -r /app/app/requirements.txt
# RUN apt-get install unzip


RUN python /app/app/setup.py && apt-get install unzip && unzip /app/app/quantized_setfitonnx_model.zip -d /app/app && rm -rf /app/app/quantized_setfitonnx_model.zip
RUN python  unzip /app/app/0.69-0.69-all.quant.zip -d /app/app && rm -rf /app/app/0.69-0.69-all.quant.zip

# RUN unzip /app/app/quantized_setfitonnx_model.zip -d /app/app
# RUN rm -rf /app/app/quantized_setfitonnx_model.zip
