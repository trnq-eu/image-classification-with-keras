# python image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install build dependencies and pkg-config
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    libhdf5-dev

# Copy dependency files into the container
COPY Pipfile Pipfile.lock ./

# Install dependencies
RUN pip install pipenv && pipenv install --system --deploy

# Copy files into the container
COPY app.py ./
COPY models/xception_v3_05_0.606.keras ./
COPY pictures ./

# Expose the port the app runs on
EXPOSE 5000

# Command to run the app when the container starts
CMD ["python", "app.py"]