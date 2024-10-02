FROM python:3.10
WORKDIR /api

# Copy the requirements file from the root directory
COPY requirements.txt /api/requirements.txt

# Install the required dependencies
RUN pip install --no-cache-dir --upgrade -r /api/requirements.txt

# Copy the entire project into the Docker container
COPY . /api

# Expose port to the outside world
EXPOSE 8000

# Command to run the FastAPI app
CMD ["uvicorn", "code.deployment.api.main:app", "--host", "0.0.0.0", "--port", "8000"]


