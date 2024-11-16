#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "linear_regression.cpp"  // Подключаем вашу реализацию

namespace py = pybind11;

// Создаём Python модуль
PYBIND11_MODULE(simple_linear_regression, m) {
    m.doc() = "Python bindings for Simple Linear Regression";  // Описание модуля
    m.def("linear_regression", &linear_regression, "Compute linear regression",
          py::arg("X"), py::arg("Y"));  // Объявляем функцию для Python
}