# Use the AWS Lambda base image for Python
FROM python:alphine

# Copy the app files into the container
COPY app/ ./app

# Install necessary build tools
RUN apt-get update && \
    apt-get install -y build-essential g++-10 gcc cmake
    
# Install dependencies
RUN pip install -r app/requirements.txt

# Run the Download Script
RUN python app/download_model.py

# Command to run the FastAPI app through Mangum for AWS Lambda
CMD ["app.main.handler"]