FROM python:3.11-slim-bookworm

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# CPU-only wheels. The default index resolves torch to the CUDA build, which drags
# in several gigabytes of NVIDIA runtime that never gets used here.
RUN pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cpu \
      "torch>=2.0.0"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Generate the training set and fit the model at build time, then drop the raw
# data. Keeps model artifacts out of the repo and makes every image reproducible
# from source rather than from a committed binary.
RUN python -m simulator.make_dataset \
 && python main.py train \
 && rm -f data/*.h5 data/*.hdf5

ENV API_URL=http://127.0.0.1:5051 \
    API_KEY=local-simulator \
    SIM_PORT=5051 \
    PORT=8080

EXPOSE 8080

CMD ["./docker-entrypoint.sh"]
