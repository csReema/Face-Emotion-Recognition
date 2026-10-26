"""
Real-time Face Emotion Recognition
Webcam -> face detection (OpenCV) -> emotion classification (DeepFace CNN).

Keys:  q = quit   s = save screenshot
"""
import time
from collections import deque, Counter

import cv2
from deepface import DeepFace

EMOJI = {
    "happy": "Happy", "sad": "Sad", "angry": "Angry", "fear": "Fear",
    "surprise": "Surprise", "disgust": "Disgust", "neutral": "Neutral",
}
COLORS = {
    "happy": (0, 200, 0), "sad": (200, 100, 0), "angry": (0, 0, 220),
    "fear": (160, 0, 160), "surprise": (0, 200, 220),
    "disgust": (0, 120, 80), "neutral": (180, 180, 180),
}

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
history = deque(maxlen=8)          # smooth predictions over last frames
log = Counter()                    # session statistics


def analyze(face_img):
    result = DeepFace.analyze(
        face_img, actions=["emotion"], enforce_detection=False, silent=True
    )
    r = result[0] if isinstance(result, list) else result
    return r["dominant_emotion"], r["emotion"]


def draw_panel(frame, scores):
    y = 20
    for emo, val in sorted(scores.items(), key=lambda kv: -kv[1]):
        w = int(val * 1.2)
        cv2.rectangle(frame, (10, y), (10 + w, y + 12), COLORS.get(emo, (255, 255, 255)), -1)
        cv2.putText(frame, f"{emo} {val:.0f}%", (140, y + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        y += 20


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Cannot open webcam")
    frame_i, scores = 0, {}
    print("Running... press 'q' to quit, 's' to save a screenshot.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5, minSize=(80, 80))

        for (x, y, w, h) in faces:
            frame_i += 1
            if frame_i % 3 == 0:   # analyze every 3rd frame for speed
                try:
                    emo, scores = analyze(frame[y:y + h, x:x + w])
                    history.append(emo)
                except Exception as e:
                    print("analysis error:", e)
            if history:
                stable = Counter(history).most_common(1)[0][0]
                log[stable] += 1
                color = COLORS.get(stable, (255, 255, 255))
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, EMOJI.get(stable, stable), (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        if scores:
            draw_panel(frame, scores)

        cv2.imshow("Face Emotion Recognition", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            name = f"shot_{int(time.time())}.png"
            cv2.imwrite(name, frame)
            print("saved", name)

    cap.release()
    cv2.destroyAllWindows()

    total = sum(log.values()) or 1
    print("\nSession summary:")
    for emo, c in log.most_common():
        print(f"  {emo:9s} {c / total * 100:5.1f}%")


if __name__ == "__main__":
    main()
