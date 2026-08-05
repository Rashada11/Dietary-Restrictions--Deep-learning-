from django.test import SimpleTestCase

from unittest.mock import patch

from .model_inference import predict_all
from .restrictions import analyze_text


class ModelInferenceTests(SimpleTestCase):
    def test_model_returns_a_supported_label_and_confidence(self):
        results, errors = predict_all("Ingredients: milk, sugar, wheat flour")
        self.assertTrue(results, errors)
        self.assertIn("main", results)
        self.assertTrue(results["main"]["is_main"])

    def test_dietary_analysis_reports_matches(self):
        result = analyze_text("Ingredients: milk, wheat flour, sugar", ["lactose", "gluten"])
        self.assertEqual(result["status"], "AVOID")
        self.assertEqual(len(result["warnings"]), 2)
