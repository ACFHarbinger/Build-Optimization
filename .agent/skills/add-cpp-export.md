# Skill: Add a C++ Export

1. Declare the function in a header under `backend/include/build_optimizer/`.
2. Implement it in a matching `.cpp` file under `backend/src/`.
3. Add the source file to `pybind11_add_module(...)` in `backend/CMakeLists.txt`.
4. Bind it in `backend/src/bindings.cpp` with `m.def(...)`.
5. Rebuild (`pixi run build`) and call it from Python: `import build_optimizer_backend`.
