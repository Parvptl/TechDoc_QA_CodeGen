FROM python:3.11-slim

WORKDIR /app

COPY requirements-deploy.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python data/generate_dataset_simple.py
RUN python scripts/prepare_runtime_dataset.py --source data/dataset.csv --target data/runtime_dataset.csv
RUN python scripts/prepare_codet5_model.py --target models/codet5_finetuned --source models/codet5_finetuned_train_smoke

EXPOSE 8000

ENV DATASET_PATH=data/runtime_dataset.csv
ENV USE_CODET5=1

CMD ["uvicorn", "services.api:app", "--host", "0.0.0.0", "--port", "8000"]
