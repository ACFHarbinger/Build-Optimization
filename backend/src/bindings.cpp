#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "build_optimizer/knapsack.hpp"

namespace py = pybind11;
using namespace build_optimizer;

PYBIND11_MODULE(build_optimizer_backend, m) {
    m.doc() = "Build-Optimization C++ solver core";

    py::class_<KnapsackItem>(m, "KnapsackItem")
        .def(py::init<>())
        .def(py::init<std::int64_t, double>(), py::arg("weight"), py::arg("value"))
        .def_readwrite("weight", &KnapsackItem::weight)
        .def_readwrite("value", &KnapsackItem::value);

    py::class_<KnapsackResult>(m, "KnapsackResult")
        .def_readonly("total_value", &KnapsackResult::total_value)
        .def_readonly("selected_indices", &KnapsackResult::selected_indices);

    m.def("solve_knapsack", &solve_knapsack, py::arg("items"), py::arg("capacity"),
          "Exact 0-1 knapsack solve via dynamic programming.");
}
