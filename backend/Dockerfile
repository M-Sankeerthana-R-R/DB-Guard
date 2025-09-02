# Use official Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files into container
COPY . .

# Expose ports
# 5000 -> Flask GUI
# 3307 -> MySQL (if running inside)
EXPOSE 5000

# Default command (run Flask GUI)
CMD ["python", "gui/app.py"]
