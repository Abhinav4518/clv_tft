FROM python:3.10-slim

WORKDIR /app

# Copy the requirements from the api folder
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the ENTIRE project (including src, data, and saved_models)
COPY . .

EXPOSE 8000

# Run uvicorn exactly how we ran it locally (from the root)
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]