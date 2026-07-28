import sys

import locodellm

project = "locodellm"
author = "locodellm contributors"
release = locodellm.__version__

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.duration",
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_runpython.epkg",
    "sphinx_runpython.runpython",
]

exclude_patterns = ["_build"]
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_theme_options = {
    "github_url": "https://github.com/xadupre/locodellm",
    "header_links_before_dropdown": 10,
}

intersphinx_mapping = {
    "numpy": ("https://numpy.org/doc/stable", None),
    "python": (f"https://docs.python.org/{sys.version_info.major}", None),
}

suppress_warnings = ["intersphinx.external"]

epkg_dictionary = {
    "numpy": "https://numpy.org/",
    "onnx": "https://github.com/onnx/onnx",
    "onnxruntime": "https://github.com/microsoft/onnxruntime",
    "onnxruntime-genai": "https://github.com/microsoft/onnxruntime-genai",
    "python": "https://www.python.org/",
}
