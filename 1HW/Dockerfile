# Используем базовый образ Python
FROM python:3.9-slim

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y build-essential cmake pybind11-dev

# Устанавливаем numpy через pip
RUN pip install numpy

# Указываем рабочую директорию
WORKDIR /app

# Копируем все файлы проекта
COPY . .

# Удаляем старую build-директорию, если она существует, и собираем проект
RUN rm -rf build && mkdir build && cd build && cmake .. && make && cp simple_linear_regression*.so /app/

# Выполняем Python-скрипт для тестирования
CMD ["python3", "test_script.py"]
