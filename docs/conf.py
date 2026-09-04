import os
import sys
import yaml

sys.path.insert(0, os.path.abspath('../packages/navigation/src'))
sys.path.insert(0, os.path.abspath('mock_imports'))

with open(os.path.join(os.path.dirname(__file__), 'config.yaml')) as f:
    _cfg = yaml.safe_load(f)

project = _cfg['project']
copyright = _cfg['copyright']
author = _cfg['author']
release = _cfg['version']
version = _cfg['version']

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

napoleon_google_docstring = _cfg['napoleon_google_docstring']
napoleon_numpy_docstring = _cfg['napoleon_numpy_docstring']
napoleon_include_init_with_doc = _cfg['napoleon_include_init_with_doc']
napoleon_include_private_with_doc = _cfg['napoleon_include_private_with_doc']
napoleon_include_special_with_doc = _cfg['napoleon_include_special_with_doc']
napoleon_use_admonition_for_examples = _cfg['napoleon_use_admonition_for_examples']
napoleon_use_admonition_for_notes = _cfg['napoleon_use_admonition_for_notes']
napoleon_use_admonition_for_references = _cfg['napoleon_use_admonition_for_references']
napoleon_use_ivar = _cfg['napoleon_use_ivar']
napoleon_use_param = _cfg['napoleon_use_param']
napoleon_use_rtype = _cfg['napoleon_use_rtype']
napoleon_use_keyword = _cfg['napoleon_use_keyword']

add_module_names = _cfg['add_module_names']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'mock_imports']

html_theme = 'sphinx_rtd_theme'
html_theme_options = _cfg['html_theme_options']
html_static_path = []