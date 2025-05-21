# Video Summarization Pipeline

This repository contains a pipeline for video summarization using deep learning techniques.

## Updates

### Latest Updates
- Added intermediate result saving to prevent data loss in case of errors
- Added error handling and recovery mechanisms for LLM processing steps
- Fixed NumPy array serialization issues in JSON storage

## How to Run

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure the pipeline in `config_pipeline.json`

3. Run the pipeline:
```bash
python main.py --config config_pipeline.json
```

## Features

- Frame extraction from videos
- Sample building for analysis
- LLM-based frame content analysis
- Data cleaning and processing
- Video summarization evaluation

## Notes

- Intermediate results are saved in subdirectories to prevent data loss
- The pipeline can recover from failures and resume processing
