FROM python:3.8-slim

WORKDIR /app
COPY ./models /app/models
COPY app.py /app/
COPY requirements.txt /app/

RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

EXPOSE 5005

CMD ["python", "app.py"]