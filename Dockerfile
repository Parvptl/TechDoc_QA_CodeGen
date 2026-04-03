FROM python:3.11-slim

WORKDIR /app

COPY requirements-deploy.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python data/generate_dataset_simple.py

EXPOSE 8000

CMD ["uvicorn", "services.api:app", "--host", "0.0.0.0", "--port", "8000"]
