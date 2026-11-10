"""Flask server for the Emotion Detector application."""

from flask import Flask, render_template, request

from EmotionDetection.emotion_detection import emotion_detector, emotion_predictor

app = Flask("Emotion Detector")


@app.route("/")
def render_index_page():
    """Render the main page of the application."""
    return render_template("index.html")


@app.route("/emotionDetector")
def sent_detector():
    """Analyze the supplied text and return the emotion scores."""
    text_to_detect = request.args.get("textToAnalyze", "")
    response = emotion_predictor(emotion_detector(text_to_detect))

    if response["dominant_emotion"] is None:
        return "Invalid text! Please try again."

    return (
        f"For the given statement, the system response is "
        f"'anger': {response['anger']}, 'disgust': {response['disgust']}, "
        f"'fear': {response['fear']}, 'joy': {response['joy']} and "
        f"'sadness': {response['sadness']}. "
        f"The dominant emotion is {response['dominant_emotion']}."
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
