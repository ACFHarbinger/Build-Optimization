# Browser Extension Reference

The `extension/src` (Manifest V3 browser extension) reference is generated
with **TypeDoc**.

## Build locally

```bash
npm install
npx typedoc --options docs/extension/typedoc.json
```

Open `docs/_build/api/extension/index.html`, or run `bash docs/build_docs.sh`
to build everything and copy it into `site/api/extension/`.
