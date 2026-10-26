# Face Emotion Recognition

Real-time facial emotion recognition from a webcam, built with Python, OpenCV and DeepFace.
Developed with the help of AI tools (Claude) for code generation and debugging.

## How it works
1. OpenCV captures webcam frames and detects faces (Haar cascade).
2. Each face crop is classified by a pretrained CNN (DeepFace) into 7 emotions:
   happy, sad, angry, fear, surprise, disgust, neutral.
3. Predictions are smoothed over the last 8 results to avoid flicker.
4. A live bar panel shows the score of every emotion; a session summary prints on exit.

## Run
```bash
pip install -r requirements.txt
python app.py
```
Keys: `q` quit, `s` save screenshot. The first run downloads the model weights (needs internet once).

## Ideas for extension
- Log emotions over time and plot them with matplotlib
- Add a Streamlit UI
- Combine with the text emotion detector (Arabic/English)
