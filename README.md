# Image to Text and Search Project

## Overview
This project is a Python-based application that extracts frames from videos, generates text descriptions and embeddings for each frame using the BLIP model, and enables semantic search based on user prompts. It is designed to analyze video content, save the results as JSON files, and retrieve relevant frames with timestamps and confidence scores.

## Features
- **Frame Extraction**: Extract keyframes from video files at specified intervals using OpenCV.
- **Image Captioning**: Generate natural language descriptions for each frame using the Salesforce BLIP model.
- **Embedding Generation**: Create semantic embeddings for descriptions using SentenceTransformers (all-MiniLM-L6-v2) to enable intelligent search.
- **JSON Storage**: Save frame metadata (path, timestamp, description, keywords, embedding) in JSON files organized by video name.
- **Semantic Search**: Search for frames based on text prompts with confidence scores and timestamps.
- **Memory Efficiency**: Optimized to handle large numbers of frames with memory management.

## Prerequisites
- Python 3.6 or higher
- Required libraries:
  - `opencv-python`
  - `transformers`
  - `pillow`
  - `sentence-transformers`
  - `keybert`
  - `numpy`
- FFmpeg (for `ffprobe` to get video metadata)

Install dependencies:
```bash
pip3 install opencv-python transformers pillow sentence-transformers keybert numpy
```

## Installation
1. Clone the repository (or copy the scripts):
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```
2. Ensure the `frames` and `json_saves` directories exist or will be created automatically.

## Usage

### 1. Extract Frames from Video
Use `frame.py` to extract keyframes from a video:
```bash
python3 frame.py /path/to/video.mp4
```
- Output: Frames are saved in `./frames/<video_name>/<video_name>_frameN.jpg`.
- Default interval is 30 frames; adjust in the script if needed.

### 2. Generate JSON Files
Use `image2text.py` to process frames and create JSON files:
```bash
python3 image2text.py ./frames/<video_name> [/path/to/video.mp4]
```
- Optional video path provides accurate FPS for timestamp calculation.
- Output: JSON files in `./json_saves/<video_name>/<video_name>_frameN.json` with metadata.

### 3. Search for Frames
Search for frames based on a prompt:
```bash
python3 image2text.py --search "your prompt here"
```
- Example: `python3 image2text.py --search "police officer with green shirt"`.
- Output: Displays matching frames with path, timestamp, and confidence score (if confidence ≥ 50%).

## File Structure
```
project/
├── frames/              # Directory for extracted frames
│   └── <video_name>/
│       ├── <video_name>_frame1.jpg
│       ├── <video_name>_frame2.jpg
│       └── ...
├── json_saves/          # Directory for JSON files
│   └── <video_name>/
│       ├── <video_name>_frame1.json
│       ├── <video_name>_frame2.json
│       └── ...
├── frame.py             # Script to extract frames
├── image2text.py        # Script for captioning and searching
└── README.md            # This file
```

## Configuration
- **FPS**: Automatically detected from video if provided; defaults to 30.0.
- **Interval**: Defaults to 30 frames in `frame.py`; adjust for more/less frames.
- **Confidence Threshold**: Defaults to 50% in searches; modify in `search_frame_by_prompt` if needed.

## Troubleshooting
- **Missing JSON Files**: Ensure frames are extracted and `image2text.py` is run successfully.
- **Timestamp Issues**: Verify video FPS with `ffprobe -i /path/to/video.mp4 -show_streams -select_streams v:0 -print_format json` and provide the video path.
- **Errors**: Check console output and ensure all dependencies are installed.

## Contributing
Feel free to fork this repository, submit issues, or send pull requests to improve the project.

## License
This project is open-source under the MIT License (see LICENSE file for details).

## Acknowledgments
- Built with help from xAI's Grok.
- Utilizes models from Hugging Face and libraries like OpenCV and SentenceTransformers.