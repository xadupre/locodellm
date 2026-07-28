import os
import subprocess
import sys
import time
import unittest

from locodellm import __file__ as locodellm_file
from locodellm.ext_test_case import ExtTestCase, has_onnxruntime_genai, is_windows

VERBOSE = 0
ROOT = os.path.realpath(os.path.abspath(os.path.join(locodellm_file, "..", "..")))

# Strings printed by transformers / huggingface_hub when the machine cannot
# reach the HuggingFace Hub.  The mock model still downloads the tokenizer, so
# the example is skipped rather than failed when there is no connectivity.
_CONNECTIVITY_MARKERS = (
    "We couldn't connect to 'https://huggingface.co'",
    "Cannot access content at: https://huggingface.co/",
    "Can't load the configuration of",
    "No address associated with hostname",
    "Failed to resolve 'huggingface.co'",
)


class TestDocumentationExamples(ExtTestCase):
    def run_test(self, fold: str, name: str, verbose: int = 0) -> int:
        """Runs a documentation example as a subprocess under ``UNITTEST_GOING=1``.

        Returns:
            ``1`` when the example ran successfully.
        """
        env = os.environ.copy()
        env["UNITTEST_GOING"] = "1"
        ppath = env.get("PYTHONPATH", "")
        if not ppath:
            env["PYTHONPATH"] = ROOT
        elif ROOT not in ppath:
            sep = ";" if is_windows() else ":"
            env["PYTHONPATH"] = ppath + sep + ROOT

        perf = time.perf_counter()
        cmds = [sys.executable, "-u", os.path.join(fold, name)]
        p = subprocess.Popen(cmds, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
        out, err = p.communicate()
        st = err.decode("utf-8", errors="ignore")
        if p.returncode != 0:
            if any(marker in st for marker in _CONNECTIVITY_MARKERS):
                raise unittest.SkipTest(f"Connectivity issues while running {name!r}:\n{st}")
            raise AssertionError(
                f"Example {name!r} (cmd: {cmds}) failed due to\n{st}"
                f"\n--stdout--\n{out.decode('utf-8', errors='ignore')}"
            )
        dt = time.perf_counter() - perf
        if verbose:
            print(f"{dt:.3f}: run {name!r}")
        return 1

    def test_check_unittest_going_is_true(self):
        """Checks that ``UNITTEST_GOING`` is set while running the tests."""
        self.assertIn("UNITTEST_GOING", os.environ)
        self.assertEqual(os.environ["UNITTEST_GOING"], "1")

    @classmethod
    def add_test_methods(cls):
        """Registers one test method per ``plot_*.py`` documentation example."""
        this = os.path.abspath(os.path.dirname(__file__))
        fold = os.path.normpath(os.path.join(this, "..", "docs", "examples"))
        found = []
        if os.path.isdir(fold):
            for name in sorted(os.listdir(fold)):
                if name.endswith(".py") and name.startswith("plot_"):
                    found.append(name)

        for name in found:
            if not has_onnxruntime_genai():

                @unittest.skip("onnxruntime-genai is missing")
                def _test_(self, fold=fold, name=name):
                    res = self.run_test(fold, name, verbose=VERBOSE)
                    self.assertTrue(res)

            else:

                def _test_(self, fold=fold, name=name):
                    res = self.run_test(fold, name, verbose=VERBOSE)
                    self.assertTrue(res)

            short_name = os.path.splitext(name)[0]
            setattr(cls, f"test_{short_name}", _test_)


TestDocumentationExamples.add_test_methods()


if __name__ == "__main__":
    unittest.main(verbosity=2)
