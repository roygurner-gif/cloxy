FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY cloxy.py .

ENV CLOXY_DATA_DIR=/data
VOLUME /data
EXPOSE 9055

CMD ["python", "cloxy.py"]
