# Use the AWS Lambda base image for Python
FROM python:alpine

# Copy the app files into the container
COPY app/ ./app

# Install necessary build tools
RUN apk update && \
    apk add --no-cache build-base g++ gcc cmake
    
# Install dependencies
RUN pip install -r app/requirements.txt

# Run the Download Script
RUN python app/download_model.py

# Command to run the FastAPI app through Mangum for AWS Lambda
CMD ["app.main.handler"]