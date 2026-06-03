FROM python:3.10-slim

WORKDIR /app

# install system dependencies
RUN apt-get update && apt-get install -y build-essential

# copy requirements first (for caching)
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# copy project
COPY . .

# expose FastAPI port
EXPOSE 8000

# run app
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]