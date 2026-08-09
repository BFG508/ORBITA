import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from visualize_benchmark import _arch_cdf_phrase, _arch_gen_phrase, _arch_name_es, ARCH_ORDER


class TestVisualizationFormatting(unittest.TestCase):
    """Test suite for visualization grammar and preposition logic."""

    def test_arch_name_es(self):
        """Verify Spanish architecture name mappings for all 5 architectures."""
        expected = {
            "resnet": "ResNet",
            "mlp": "MLP",
            "lstm": "LSTM",
            "linear": "modelo lineal",
            "tree": "árbol de decisión",
        }
        for arch in ARCH_ORDER:
            self.assertIn(arch, expected)
            self.assertEqual(_arch_name_es(arch), expected[arch])

    def test_arch_gen_phrase_preposition(self):
        """Verify correct Spanish prepositions ('de' vs 'del') for model names."""
        # 'del' for 'modelo lineal' and 'árbol de decisión'
        self.assertEqual(_arch_gen_phrase("linear"), "Generalización del modelo lineal")
        self.assertEqual(_arch_gen_phrase("tree"), "Generalización del árbol de decisión")

        # 'de' for proper nouns / acronyms: ResNet, MLP, LSTM
        self.assertEqual(_arch_gen_phrase("resnet"), "Generalización MoE de ResNet")
        self.assertEqual(_arch_gen_phrase("mlp"), "Generalización de MLP")
        self.assertEqual(_arch_gen_phrase("lstm"), "Generalización de LSTM")

    def test_arch_cdf_phrase_preposition(self):
        """Verify Spanish CDF title phrases with 'de' vs 'del' for all 5 architectures."""
        self.assertEqual(
            _arch_cdf_phrase("resnet"),
            "Función de distribución acumulada de ResNet de error espacial",
        )
        self.assertEqual(
            _arch_cdf_phrase("mlp"),
            "Función de distribución acumulada de MLP de error espacial",
        )
        self.assertEqual(
            _arch_cdf_phrase("lstm"),
            "Función de distribución acumulada de LSTM de error espacial",
        )
        self.assertEqual(
            _arch_cdf_phrase("linear"),
            "Función de distribución acumulada del modelo lineal de error espacial",
        )
        self.assertEqual(
            _arch_cdf_phrase("tree"),
            "Función de distribución acumulada del árbol de decisión de error espacial",
        )

    def test_arch_order_completeness(self):
        """Ensure canonical ARCH_ORDER contains all 5 supported architectures."""
        self.assertEqual(len(ARCH_ORDER), 5)
        self.assertListEqual(ARCH_ORDER, ["resnet", "lstm", "mlp", "tree", "linear"])


if __name__ == "__main__":
    unittest.main()
