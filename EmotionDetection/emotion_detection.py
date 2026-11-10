"""Emotion detection using the IBM Watson NLP EmotionPredict service."""

import json
import os

import requests

DEFAULT_URL = (
    "https://sn-watson-emotion.labs.skills.network/"
    "v1/watson.runtime.nlp.v1/NlpService/EmotionPredict"
)
MODEL_ID = "emotion_aggregated-workflow_lang_en_stock"
REQUEST_TIMEOUT = 10


def _error_response():
    """Return the standard response used when the input is invalid."""
    return {
        "anger": None,
        "disgust": None,
        "fear": None,
        "joy": None,
        "sadness": None,
        "dominant_emotion": None,
    }


def emotion_detector(text_to_analyze):
    """Send text to Watson NLP and return the raw emotion prediction."""
    url = os.getenv("WATSON_URL", DEFAULT_URL)
    headers = {
        "grpc-metadata-mm-model-id": MODEL_ID,
        "Content-Type": "application/json",
    }
    payload = {"raw_document": {"text": text_to_analyze}}

    try:
        response = requests.post(
            url, json=payload, headers=headers, timeout=REQUEST_TIMEOUT
        )
    except requests.RequestException:
        return _error_response()

    if response.status_code == 200:
        return json.loads(response.text)
    if response.status_code == 400:
        return _error_response()
    return _error_response()


def emotion_predictor(detected_text):
    """Format the Watson NLP response into the expected output dictionary."""
    if not detected_text or "emotionPredictions" not in detected_text:
        return _error_response()

    emotions = detected_text["emotionPredictions"][0]["emotion"]
    dominant_emotion = max(emotions, key=emotions.get)

    return {
        "anger": emotions["anger"],
        "disgust": emotions["disgust"],
        "fear": emotions["fear"],
        "joy": emotions["joy"],
        "sadness": emotions["sadness"],
        "dominant_emotion": dominant_emotion,
    }
