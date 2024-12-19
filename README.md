# YouTube Video Processing Pipeline

## Overview
Automated video processing pipeline using free tools:
- Gemini API for insights
- OpenCV for video analysis
- FFmpeg for video processing

## Prerequisites
- Python 3.8+
- Gemini API Key
- FFmpeg installed

## Setup
1. Clone the repository
2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install requirements
```bash
pip install -r requirements.txt
```

4. Configure API Key
- Replace `YOUR_API_KEY_HERE` in `src/video_processor.py`

## Running the Pipeline
```bash
python src/video_processor.py
```

## Features
- Automatic video file detection
- Video analysis with Gemini AI
- Video resizing and optimization
- Logging support

## Workflow
1. Place videos in `input/` directory
2. Pipeline processes videos automatically
3. Processed videos and insights stored in `output/`

## Limitations
- Free Gemini API rate limits
- Basic video processing
