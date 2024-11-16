cmake_minimum_required(VERSION 3.10)

# Название проекта
project(SimpleLinearRegression LANGUAGES CXX)

# Установить стандарт C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Найти Pybind11
find_package(pybind11 REQUIRED)

# Добавить модуль Pybind11
pybind11_add_module(
    simple_linear_regression  # Название модуля
    bindings.cpp              # Файл с биндингами
)

# Вывод информации
message(STATUS "Project: ${PROJECT_NAME}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Pybind11 include directories: ${pybind11_INCLUDE_DIRS}")
