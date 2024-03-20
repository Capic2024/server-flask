FROM python:latest

RUN apt-get update && apt-get install -y \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libgtk-3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir myapp/
COPY app.py myapp/app.py
COPY requirements.txt myapp/requirements.txt
WORKDIR /myapp/
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000
CMD ["python", "app.py"]
