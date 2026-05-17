# 1. Base image Python 3.11 
FROM python:3.11-slim

# 2. Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Install dependencies sistem 
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. Working directory
WORKDIR /app

# 5. Copy file requirements dan install dependencies Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 6. Copy seluruh kode proyek ke dalam container
COPY . .

# 7. Buka port 8000 
EXPOSE 8000

# 8. Perintah untuk menjalankan Uvicorn saat container dinyalakan
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]