# 🔍 Object Detection Microservice

A powerful, production-ready microservice for real-time object detection in images using YOLOv8. Built with FastAPI and Docker, this service provides a simple REST API that can detect and identify objects in uploaded images with high accuracy and speed.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=flat&logo=docker&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-FF6B6B?style=flat&logo=ultralytics)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## ✨ Features

- **🚀 Fast Object Detection**: Powered by YOLOv8 for state-of-the-art accuracy and speed
- **🌐 REST API**: Simple and intuitive endpoints for easy integration
- **🔧 GPU Acceleration**: CUDA support for lightning-fast inference
- **📦 Docker Ready**: Fully containerized for seamless deployment anywhere
- **📚 Interactive Docs**: Auto-generated API documentation with Swagger UI
- **🎯 Configurable Confidence**: Adjustable detection thresholds
- **💾 Image Storage**: Automatic saving of processed images with bounding boxes
- **🔒 Production Ready**: Built with best practices and error handling

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Client App    │───▶│   FastAPI API    │───▶│   YOLOv8 Model │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │  Output Storage  │
                       └──────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- NVIDIA GPU with CUDA support (optional, for GPU acceleration)
- Python 3.8+ (for local development)

### 🐳 Docker Deployment (Recommended)

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd "Object Detection First"
   ```

2. **Build the Docker image**
   ```bash
   cd ai_backend
   docker build -t object-detection-microservice .
   ```

3. **Run with GPU support** (Recommended for production)
   ```bash
   # Create output directory
   mkdir -p output_images
   chmod 777 output_images
   
   # Run with GPU acceleration
   docker run --gpus all -p 8000:8000 \
     -v "$(pwd)/output_images:/app/output_images" \
     object-detection-microservice
   ```

4. **Run without GPU** (CPU only)
   ```bash
   docker run -p 8000:8000 \
     -v "$(pwd)/output_images:/app/output_images" \
     object-detection-microservice
   ```

### 🐍 Local Development Setup

1. **Install dependencies**
   ```bash
   cd ai_backend
   pip install -r requirements.txt
   ```

2. **Run the application**
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

## 📖 API Documentation

Once the service is running, access the interactive API documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI Spec**: http://localhost:8000/openapi.json

### 🔗 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check and basic info |
| `POST` | `/detect` | Upload image for object detection |
| `GET` | `/output/{image_name}` | Retrieve processed output image |

### 📝 Usage Examples

#### Basic Object Detection

```bash
# Upload an image for detection
curl -X POST "http://localhost:8000/detect" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_image.jpg"
```

#### With Custom Confidence Threshold

```bash
curl -X POST "http://localhost:8000/detect?confidence=0.5&save_image=true" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_image.jpg"
```

#### Python Example

```python
import requests

# Upload image for detection
with open('test_image.jpg', 'rb') as f:
    files = {'file': f}
    params = {'confidence': 0.3, 'save_image': True}
    response = requests.post('http://localhost:8000/detect', 
                           files=files, params=params)
    
results = response.json()
print(f"Detected {len(results['labels'])} objects:")
for i, (label, score) in enumerate(zip(results['labels'], results['scores'])):
    print(f"  {i+1}. {label}: {score:.2f}")
```

## 📊 Response Format

The API returns detection results in the following JSON format:

```json
{
  "boxes": [
    [x1, y1, x2, y2],
    [x1, y1, x2, y2]
  ],
  "labels": [
    "person",
    "car"
  ],
  "scores": [
    0.95,
    0.87
  ]
}
```

Where:
- `boxes`: Bounding box coordinates [x1, y1, x2, y2]
- `labels`: Object class names
- `scores`: Confidence scores (0-1)

## 🛠️ Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CONFIDENCE_THRESHOLD` | `0.25` | Minimum confidence for detections |
| `OUTPUT_DIR` | `output_images` | Directory for saving processed images |
| `MODEL_PATH` | `yolov8n.pt` | Path to YOLOv8 model file |

### Model Options

The service supports different YOLOv8 model variants:

- `yolov8n.pt` - Nano (fastest, smallest)
- `yolov8s.pt` - Small
- `yolov8m.pt` - Medium  
- `yolov8l.pt` - Large
- `yolov8x.pt` - Extra Large (most accurate)

To use a different model, modify the model loading in `main.py`:

```python
model = YOLO("yolov8s.pt")  # Change to desired model
```

## 📁 Project Structure

```
Object Detection First/
├── README.md                    # This documentation
├── .gitignore                   # Git ignore rules
├── test.jpg                     # Sample test image
└── ai_backend/                  # Main application
    ├── main.py                  # FastAPI application
    ├── requirements.txt         # Python dependencies
    ├── Dockerfile              # Docker configuration
    ├── .dockerignore           # Docker ignore rules
    └── output_images/          # Processed images storage
```

## 🔧 Development

### Adding New Features

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-feature`
3. Make your changes
4. Test thoroughly
5. Submit a pull request

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run tests
pytest tests/
```

## 🚨 Troubleshooting

### Common Issues

**GPU not detected**
```bash
# Check NVIDIA Docker runtime
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

**Memory issues**
- Reduce batch size or use smaller model variant
- Ensure sufficient GPU/system memory

**Permission errors**
```bash
# Fix output directory permissions
chmod 777 output_images/
```

### Performance Optimization

- Use GPU acceleration for 10-50x speedup
- Choose appropriate model size for your use case
- Implement image preprocessing/resizing for large images
- Use batch processing for multiple images

## 📋 Requirements

- **Python**: 3.8+
- **CUDA**: 11.0+ (for GPU acceleration)
- **Memory**: 4GB+ RAM recommended
- **Storage**: 2GB+ for models and output images

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

This project is inspired by and gives credit to the original microservice implementation by [mskhan793](https://github.com/mskhan793/Microservice-for-Object-Detection). Special thanks for the foundational architecture and approach.

---

<div align="center">
  <strong>🔍 Happy Object Detecting! 🔍</strong>
</div>
