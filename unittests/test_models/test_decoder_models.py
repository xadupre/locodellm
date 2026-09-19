import json
import os
import tempfile
import unittest

import onnx

from locodellm.ext_test_case import ExtTestCase, requires_onnxruntime_genai
from locodellm.test_models import mock_generate_model, tiny_model


class TestDecoderModels(ExtTestCase):
    def test_opset_compatible_ir(self):
        """Checks that both decoders use the IR version required by their opsets."""
        for builder in (mock_generate_model, tiny_model):
            with self.subTest(builder=builder.__name__):
                model = builder.make_decoder_model()
                self.assertEqual(
                    model.ir_version, onnx.helper.find_min_ir_version_for(model.opset_import)
                )
                onnx.checker.check_model(model)

    @requires_onnxruntime_genai()
    def test_genai_loads_decoders(self):
        """Checks that GenAI loads both decoders without downloading tokenizers."""
        import onnxruntime_genai

        for builder in (mock_generate_model, tiny_model):
            with self.subTest(builder=builder.__name__), tempfile.TemporaryDirectory() as folder:
                onnx.save(builder.make_decoder_model(), os.path.join(folder, "model.onnx"))
                with open(os.path.join(folder, "genai_config.json"), "w") as stream:
                    json.dump(builder.make_genai_config(), stream)
                model = onnxruntime_genai.Model(folder)
                self.assertIsNotNone(model)


if __name__ == "__main__":
    unittest.main()
