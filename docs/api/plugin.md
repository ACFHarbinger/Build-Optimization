# Unreal Engine Plugin Reference

The `unreal-plugin/` in-editor build-optimization reference is generated with
**Doxygen** from `unreal-plugin/Source/BuildOptimization/Public` and `Private`.

## Build locally

```bash
doxygen docs/plugin/Doxyfile
```

Open `docs/_build/api/plugin/index.html`, or run `bash docs/build_docs.sh` to
build everything and copy it into `site/api/plugin/`. The Doxygen input only
scans the module's own sources; it does not require an Unreal Engine install.
