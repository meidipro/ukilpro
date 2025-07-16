# Stage 1: Use an official Python image as the base
# Using a slim version to keep the image size down
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /code

# Install system dependencies that some Python packages might need
# (e.g., for Pillow or other libraries)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the Python requirements file into the container
COPY ./requirements.txt /code/requirements.txt

# Install the Python dependencies
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy your Python source code into the container
# This copies the 'api' and 'py_src' directories
COPY ./api /code/api
COPY ./py_src /code/py_src

# Expose the port that the app will run on
# Note: Fly.io automatically maps this to ports 80 and 443 externally
EXPOSE 8080

# The command to run when the container starts
# We use gunicorn for a production-ready server
# It will run the 'app' instance from the 'backend' module in the 'api' directory
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:$PORT", "api.backend:app"]