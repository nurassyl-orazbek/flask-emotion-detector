"""Unit tests for the Emotion Detection application."""

import unittest

from EmotionDetection.emotion_detection import emotion_detector, emotion_predictor


class TestEmotionDetection(unittest.TestCase):
    """Test the dominant emotion returned for representative statements."""

    def test_emotion_predictor(self):
        """Check the dominant emotion for five statements."""
        statements = {
            "I am glad this happened": "joy",
            "I am really mad about this": "anger",
            "I feel disgusted just hearing about this": "disgust",
            "I am so sad about this": "sadness",
            "I am really afraid that this will happen": "fear",
        }
        for statement, expected in statements.items():
            result = emotion_predictor(emotion_detector(statement))
            self.assertEqual(result["dominant_emotion"], expected)


if __name__ == "__main__":
    unittest.main()
