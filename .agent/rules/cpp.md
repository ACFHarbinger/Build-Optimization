# C++ Rules

- C++17, headers in `backend/include/build_optimizer/`, implementation in `backend/src/`.
- Every new exported function needs a corresponding `pybind11` binding in `backend/src/bindings.cpp` — see [`.agent/skills/add-cpp-export.md`](../skills/add-cpp-export.md).
- No raw owning pointers; use value types, `std::unique_ptr`, or `std::shared_ptr`.
- Rebuild and run tests after any change: `cd backend && pixi run build && pixi run test`.
