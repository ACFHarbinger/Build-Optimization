# Python Middleware Reference

The full `middleware/src` reference is generated with **sphinx-autoapi**, which
discovers modules from source without needing the package installed or
importable.

## Build locally

```bash
uv sync --extra docs
sphinx-build -b html docs/sphinx docs/_build/api/python
```

Open `docs/_build/api/python/index.html`, or run `bash docs/build_docs.sh` to
build everything and copy it into `site/api/python/`.
