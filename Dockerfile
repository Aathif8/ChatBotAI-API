# Use the AWS Lambda base image for Python
FROM public.ecr.aws/lambda/python:latest

# Copy the app files into the container
COPY app/ ./app

# Install dependencies
RUN pip install -r app/requirements.txt

# Run the Download Script
RUN Python app/download.py

# Command to run the FastAPI app through Magnum for AWS Lambda
CMD ["app.main.handler"]