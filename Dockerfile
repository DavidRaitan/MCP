FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY library_backend.py wrapper_mcp.py .
CMD ["python", "wrapper_mcp.py"]
