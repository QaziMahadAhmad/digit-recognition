FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt huggingface_hub

COPY . .

RUN python3 -c "from huggingface_hub import hf_hub_download; import os; os.makedirs('mnist_model', exist_ok=True); hf_hub_download(repo_id='Mahad0007/digit-recognition', repo_type='space', filename='mnist_model/scaler.pkl', local_dir='.'); hf_hub_download(repo_id='Mahad0007/digit-recognition', repo_type='space', filename='mnist_model/pca.pkl', local_dir='.'); hf_hub_download(repo_id='Mahad0007/digit-recognition', repo_type='space', filename='mnist_model/knn_model.pkl', local_dir='.'); print('Models ready!')"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]