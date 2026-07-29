#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "build_optimizer/knapsack.hpp"
#include "build_optimizer/mckp.hpp"

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

    py::class_<MckpOption>(m, "MckpOption")
        .def(py::init<>())
        .def(py::init<std::size_t, std::int64_t, double>(), py::arg("class_index"), py::arg("weight"),
             py::arg("value"))
        .def_readwrite("class_index", &MckpOption::class_index)
        .def_readwrite("weight", &MckpOption::weight)
        .def_readwrite("value", &MckpOption::value);

    py::class_<MckpResult>(m, "MckpResult")
        .def_readonly("total_value", &MckpResult::total_value)
        .def_readonly("total_weight", &MckpResult::total_weight)
        .def_readonly("selected_option_indices", &MckpResult::selected_option_indices);

    m.def("solve_mckp_branch_and_bound", &solve_mckp_branch_and_bound, py::arg("options"), py::arg("num_classes"),
          py::arg("capacity"),
          "Exact Multiple-Choice Knapsack solve (>=0 options per class, at most one selected per "
          "class) via branch-and-bound with a fractional-relaxation upper bound. Models equipment "
          "slots directly: each class is a slot, each option a candidate item.");
}
