# Use an explicit, lightweight, stable base image layer
FROM python:3.12-slim

# Establish the secure operational directory context
WORKDIR /workspace

# Copy configuration manifests first to leverage optimal layer caching mechanics
COPY requirements.txt .

# Install production dependencies cleanly with no background cache bloating
RUN pip install --no-cache-dir -r requirements.txt

# Copy the core service layout layers into the container boundaries
COPY app/ ./app/

# Expose the correct application portal gateway mapping port
EXPOSE 8000

# Start the high-availability Uvicorn production worker thread pool
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
