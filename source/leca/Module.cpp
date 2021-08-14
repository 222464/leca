#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "Leca.h"

namespace py = pybind11;

PYBIND11_MODULE(leca, m) {
    m.def("setNumThreads", [](int numThreads) {
        omp_set_num_threads(numThreads);        
    });

    m.def("getNumThreads", []() -> int {
        return omp_get_num_threads();
    });

    py::class_<Leca>(m, "Leca")
        .def(py::init<>())
        .def_readwrite("lr", &Leca::lr)
        .def_readwrite("discount", &Leca::discount)
        .def_readwrite("traceDecay", &Leca::traceDecay)
        .def("init", &Leca::init,
            py::arg("width"),
            py::arg("height"),
            py::arg("seed") = 1234
        )
        .def("step", &Leca::step,
            py::arg("inputs"),
            py::arg("reward"),
            py::arg("learnEnabled") = true
        )
        .def("getWidth", &Leca::getWidth)
        .def("getHeight", &Leca::getHeight)
        .def("getOn", &Leca::getOn)
        .def("getLastOn", &Leca::getLastOn);
}
