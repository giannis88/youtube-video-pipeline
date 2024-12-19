#!/bin/bash

# Initialize Git repository
git init

# Add remote origin
git remote add origin https://github.com/giannis88/youtube-video-pipeline.git

# Stage all files
git add .

# Commit with initial message
git commit -m "Initial commit: YouTube Video Processing Pipeline"

# Push to main branch
git branch -M main
git push -u origin main

echo "Repository initialized and pushed to GitHub"
