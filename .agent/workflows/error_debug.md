# Workflow: Debugging

1. Reproduce with the smallest possible Hydra override (`uv run python main.py policy=... game=...`).
2. Check `middleware/src/tracking` for the failing run's logged metrics before adding print statements.
3. For C++ crashes, rebuild with `-DCMAKE_BUILD_TYPE=Debug` and reproduce under `gdb`/`lldb`.
4. For Tauri/React issues, `npm run tauri:dev` opens devtools automatically — check both the webview console and the Rust process's stdout.
5. Once root-caused, add a regression test (see [`test_writing.md`](test_writing.md)) before closing the issue.
