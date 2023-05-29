FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file to the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install -r requirements.txt

# Copy the application code to the container
COPY . .

# Expose the port your Flask app will run on
EXPOSE 5000

# Define the command to start the Flask app
CMD ["python", "app.py"]
