# Object Detection Microservice

A fast and easy-to-use microservice for detecting objects in images using YOLOv8, built with FastAPI and Docker.

## Features
- Quick and accurate object detection using YOLOv8
- Simple REST API for image processing
- GPU acceleration for faster inference
- Docker containerization for easy deployment
- Interactive API documentation with Swagger UI

## Setup Instructions

### Build the Docker image
```bash
cd ai_backend
docker build -t object-detection-gpu .
### Run the Container 
mkdir -p output_images
chmod 777 output_images
docker run --gpus all -p 8000:8000 -v "$(pwd)/output_images:/app/output_images" object-detection-gpu
###Access the API
##API endpoint: http://localhost:8000
##Interactive documentation: http://localhost:8000/docs

## Step 3: Create a .gitignore file

```bash
cat > .gitignore << 'EOL'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
ENV/

# Output images
ai_backend/output_images/

# OS specific
.DS_Store
Thumbs.db
