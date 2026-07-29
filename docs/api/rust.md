# Tauri Rust Shell Reference

The `frontend/src-tauri` (Rust) reference is generated with **rustdoc** via
`cargo doc`.

## Build locally

```bash
cd frontend/src-tauri
cargo doc --no-deps
```

Open `frontend/src-tauri/target/doc/app/index.html`, or run
`bash docs/build_docs.sh` to build everything and copy it into
`site/api/rust/`.
