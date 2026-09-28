"""Sphinx configuration for the documentation published from main/docs."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
project = "vkexec"
copyright = "vkexec contributors"
extensions = ["myst_parser", "breathe", "sphinx.ext.graphviz"]
source_suffix = {".md": "markdown"}
master_doc = "index"
exclude_patterns = ["_build"]
html_theme = "furo"
html_baseurl = "https://janagor.github.io/vkexec/"
html_theme_options = {
    "source_repository": "https://github.com/janagor/vkexec/",
    "source_branch": "main",
    "source_directory": "docs/",
}
myst_heading_anchors = 3
breathe_projects = {"vkexec": str(ROOT / "docs" / "_build" / "doxygen" / "xml")}
breathe_default_project = "vkexec"
nitpicky = True
