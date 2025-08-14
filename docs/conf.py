# Configuration file for the Sphinx documentation builder.

import os
from importlib import metadata

# -- Project information -----------------------------------------------------

project = "abaco"
copyright = "2025, Edir Vidal and MoNA group"
author = "Edir Vidal, Angel L.P., Henry Webel, Atieh Gharib, Juliana Assis, Sebastián Ayala-Ruano, Alberto Santos"
PACKAGE_VERSION = metadata.version("abaco")
version = PACKAGE_VERSION
release = PACKAGE_VERSION


# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",  # Core extension for generating documentation from docstrings
    "sphinx.ext.autodoc.typehints",  # Automatically document type hints in function signatures
    "sphinx.ext.viewcode",  # Include links to the source code in the documentation
    "sphinx.ext.napoleon",  # Support for Google and NumPy style docstrings
    "sphinx.ext.intersphinx",  # allows linking to other projects' documentation in API
    "sphinx_new_tab_link",  # each link opens in a new tab
    "myst_nb",  # Markdown and Jupyter Notebook support
    "sphinx_copybutton",  # add copy button to code blocks
]

#  https://myst-nb.readthedocs.io/en/latest/computation/execute.html
nb_execution_mode = "auto"
nb_execution_timeout = -1  # -1 means no timeout
myst_enable_extensions = ["dollarmath", "amsmath"]

# Plotly support through require javascript library
# https://myst-nb.readthedocs.io/en/latest/render/interactive.html#plotly
html_js_files = [
    "https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.4/require.min.js"
]

# https://myst-nb.readthedocs.io/en/latest/configuration.html
# Execution
nb_execution_raise_on_error = True
# Rendering
nb_merge_streams = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# Ignore
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "jupyter_execute",
    "conf.py",
]

# Intersphinx options
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "scikit-learn": ("https://scikit-learn.org/stable/", None),
    "torch": ("https://docs.pytorch.org/docs/stable/", None),
    "transformers": ("https://huggingface.co/docs/transformers/main/en/", None),
}

# -- Options for HTML output -------------------------------------------------
html_title = "ABaCo Documentation"
html_theme = "sphinx_book_theme"
html_logo = "https://raw.githubusercontent.com/Multiomics-Analytics-Group/abaco/HEAD/docs/images/logo/abaco_logo.svg"
html_favicon = "https://raw.githubusercontent.com/Multiomics-Analytics-Group/abaco/HEAD/docs/images/logo/abaco_logo.svg"
html_theme_options = {
    "github_url": "https://github.com/Multiomics-Analytics-Group/abaco",
    "repository_url": "https://github.com/Multiomics-Analytics-Group/abaco",
    "repository_branch": "main",
    "home_page_in_toc": True,
    "path_to_docs": "docs",
    "show_navbar_depth": 1,
    "use_edit_page_button": True,
    "use_repository_button": True,
    "use_download_button": True,
    "launch_buttons": {
        "colab_url": "https://colab.research.google.com"
    },
    "navigation_with_keys": False,
}

# -- Setup for sphinx-apidoc -------------------------------------------------

if os.environ.get("READTHEDOCS") == "True":
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).parent.parent
    PACKAGE_ROOT = PROJECT_ROOT / "src" / "abaco"

    def run_apidoc(_):
        from sphinx.ext import apidoc

        apidoc.main(
            [
                "--force",
                "--implicit-namespaces",
                "--module-first",
                "--separate",
                "-o",
                str(PROJECT_ROOT / "docs" / "reference"),
                str(PACKAGE_ROOT),
                str(PACKAGE_ROOT / "*.c"),
                str(PACKAGE_ROOT / "*.so"),
            ]
        )

    def setup(app):
        app.connect("builder-inited", run_apidoc)
