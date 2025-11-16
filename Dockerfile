# Use a stable Python base
FROM python:3.10-slim

# Set working dir
WORKDIR /app

# Upgrade pip/build tools first
COPY requirements.txt .
RUN python -m pip install --upgrade pip setuptools wheel

# Install numpy first (binary wheels) then rest of requirements
RUN pip install --no-cache-dir numpy==1.25.0
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code and model
COPY . .

# Expose port and run with gunicorn
EXPOSE 5000
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app", "--workers", "2"]


