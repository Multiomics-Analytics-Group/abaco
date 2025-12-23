<!-- https://myst-parser.readthedocs.io/en/latest/faq/index.html
#include-a-file-from-outside-the-docs-folder-like-readme-md -->

```{include} ../README.md
:caption: Overview
:relative-docs: docs
:relative-images:
```

```{toctree}
:maxdepth: 3
:caption: "Getting started tutorials"

tutorial/tutorial-anaerobic-digestion
tutorial/demo-mgnify-mgys-1135
tutorial/demo-mgnify-tomatoes
```

```{toctree}
:maxdepth: 4
:caption: "Reference"

reference/abaco
```

```{toctree}
:maxdepth: 1
:caption: "Indices and Tables"

genindex
modindex
search
```

---