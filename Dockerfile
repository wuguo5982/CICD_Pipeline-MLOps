FROM python:3.8-slim-buster
RUN pip install --upgrade pip  
RUN pip install --root-user-action=ignore requests

WORKDIR /app
COPY . /app

RUN apt update -y && apt-get install apt-utils && apt install awscli -y

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 unzip -y && pip install -r requirements.txt
CMD ["python3", "app.py"]
