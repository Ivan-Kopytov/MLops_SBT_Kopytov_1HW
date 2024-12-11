import torch
from torch.utils.data import Dataset
# Импортируем биндинги, установленный пакет из ДЗ-1
# Предполагается, что у вас есть пакет simple_linear_regression с функцией linear_regression(X, Y)
from simple_linear_regression import linear_regression

class RegressionDataset(Dataset):
    def __init__(self, X, Y):
        # X, Y - списки или тензоры
        # Для простоты X и Y - Python list или torch.Tensor
        # Если torch.Tensor, убедимся что они идут по оси = 0
        self.X = torch.tensor(X, dtype=torch.float32) if not isinstance(X, torch.Tensor) else X.float()
        self.Y = torch.tensor(Y, dtype=torch.float32) if not isinstance(Y, torch.Tensor) else Y.float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Демонстрация использования биндингов в методе __getitem__
        # Вызовем linear_regression на всех данных (X, Y), получим b0, b1
        # Это неэффективно, но отвечает требованиям задания.
        X_list = self.X.tolist()
        Y_list = self.Y.tolist()
        b0, b1 = linear_regression(X_list, Y_list)

        # Возвращаем:
        # вход X[idx],
        # целевое значение Y[idx],
        # а также коэффициенты регрессии для демонстрации, что мы использовали биндинги.
        # В реальном проекте, возможно, это не нужно, но для демонстрации оставим.
        return self.X[idx], self.Y[idx], torch.tensor([b0, b1], dtype=torch.float32)
