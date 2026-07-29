# Docker

The only container defined here builds the **Tauri Studio** app into distributable Linux bundles — there is no server component to containerize (the CLI and dashboard run locally against `uv`/`pixi` environments; see the root [README](../README.md#installation--setup)).

```bash
docker build -f docker/Dockerfile --output type=local,dest=dist-studio .
```

Produces `dist-studio/*.deb` and `dist-studio/*.AppImage`.
