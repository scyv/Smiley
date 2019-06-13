FROM python:3.7.3-slim

COPY ./ /opt/smiley

WORKDIR /opt/smiley

RUN apt-get update -y && apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6 && \
  pip install -r requirements.txt 

EXPOSE 5000

CMD ["python", "main.py"]
