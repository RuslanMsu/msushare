"""Модуль базовых алгоритмов линейной алгебры.
Задание состоит в том, чтобы имплементировать класс Matrix
(следует воспользоваться кодом из семинара ООП), учтя рекомендации pylint.
Для проверки кода следует использовать команду pylint matrix.py.
Pylint должен показывать 10 баллов.
Кроме того, следует добавить поддержку исключений в отмеченных местах.
Для проверки корректности алгоритмов следует сравнить результаты с соответствующими функциями numpy.
"""
import json
import random
import numpy as np
from numpy.linalg import inv, det


class Matrix:
    """
    Matrix class
    """
    def __init__(self,  nrows, ncols, init = "zeros"):
        """
            nrows - количество строк матрицы
            ncols - количество столбцов матрицы
            init - метод инициализации элементов матрицы:
                "zeros" - инициализация нулями
                "ones" - инициализация единицами
                "random" - случайная инициализация
                "eye" - матрица с единицами на главной диагонали
        """
        if nrows<0 or ncols<0:
            raise ValueError('nrows and ncols must by a positive number.')
        if init not in ["zeros", "ones", "eye", "random"]:
            raise ValueError(f'init has not atribute {init}')
        self.nrows = nrows
        self.ncols = ncols
        self.data = []
        if init == "zeros":
            for i in range(0,nrows):
                row = []
                for _ in range(0,ncols):
                    row.append(0)
                self.data.append(row)
        if init == "ones":
            for i in range(0,nrows):
                row = []
                for _ in range(0,ncols):
                    row.append(1)
                self.data.append(row)
        if init == "random":
            for i in range(0,nrows):
                row = []
                for _ in range(0,ncols):
                    row.append(random.randint(1,100))
                self.data.append(row)
        if init == "eye":
            for i in range(0,nrows):
                row = []
                for _ in range(0,ncols):
                    row.append(0)
                row[i] = 1
                self.data.append(row)


    @staticmethod
    def fromDict(data):
        """
            Десеарилизация матрицы из словаря
        """
        ncols = data["ncols"]
        nrows = data["nrows"]
        items = data["data"]
        assert len(items) == ncols*nrows
        m = Matrix(nrows, ncols)
        for row in range(nrows):
            for col in range(ncols):
                m[(row, col)] = items[ncols*row + col]
        return m


    @staticmethod
    def toDict(M):
        """
            Сериализация матрицы в словарь
        """
        assert isinstance(M, Matrix)
        nrows, ncols = M.shape()
        data = []
        for row in range(nrows):
            for col in range(ncols):
                data.append(M[(row, col)])
        return {"nrows": nrows, "ncols": ncols, "data": data}


    def __str__(self):
        "Строковое представление матрицы"
        return '\n'.join(' '.join(map(str, row)) for row in self.data)


    def __repr__(self):
        return  '%s' % (self.data)


    def shape(self):
        "Вернуть кортеж размера матрицы (nrows, ncols)"
        return (self.nrows,self.ncols)


    def __getitem__(self, index):
        "Вернуть элемент с индесом index (кортеж (row, col))"
        row, col = index
        return self.data[row][col]


    def __setitem__(self, index, value):
        "Присвоить значение value элементу с индесом index (кортеж (row, col))"
        check_list = isinstance(index, list)
        check_tuple = isinstance(index, tuple)
        if not (check_list or check_tuple):
            raise ValueError('index is not list or typle.')
        if len(index)!=2:
            raise ValueError('len index must by 2')
        if self.nrows < index[0] or self.ncols < index[1]:
            raise IndexError('IndexError')
        row, col = index
        self.data[row][col] = value


    def __sub__(self, rhs):
        "Вычесть матрицу rhs и вернуть результат"
        if self.nrows != rhs.nrows or self.ncols != rhs.ncols:
            raise ValueError('Wrong size')

        for i in range(self.nrows):
            for j in range(self.ncols):
                self.data[i][j] = self.data[i][j] - rhs.data[i][j]
        return self


    def __add__(self, rhs):
        "Сложить с матрицей rhs и вернуть результат"
        if self.nrows != rhs.nrows or self.ncols != rhs.ncols:
            raise ValueError('Wrong size')

        for i in range(self.nrows):
            for j in range(self.ncols):
                self.data[i][j] = self.data[i][j] + rhs.data[i][j]
        return self


    def __mul__(self, rhs):
        "Умножить на матрицу rhs и вернуть результат"
        if self.nrows != rhs.ncols:
            raise ValueError('Wrong size')

        res = Matrix(self.nrows, rhs.ncols)
        for i in range(self.nrows):
            for j in range (rhs.ncols):
                summ = 0
                for k in range(self.ncols):
                    summ+=self.data[i][k]*rhs.data[k][j]
                res[(i,j)] = summ
        return res


    def __pow__(self, power):
        "Возвести все элементы в степень pow и вернуть результат"
        for i in range(self.nrows):
            for j in range(self.ncols):
                self.data[i][j] = self.data[i][j]**power
        return self


    def sum(self):
        "Вернуть сумму всех элементов матрицы"
        summ = 0
        for i in self.data:
            summ+=sum(i)
        return summ


    def __minor(self,i,j):
        res = Matrix(self.ncols-1,self.nrows-1)
        #print(res)
        for row in range(self.nrows):
            for col in range(self.ncols):
                #print(row,col)
                if(row != i and col != j):
                    ind_row = row if row < i else row -1
                    ind_col = col if col < j else col -1
                    res[(ind_row,ind_col)] = self.data[row][col]
        return res


    def det(self):
        "Вычислить определитель матрицы"
        if len(self.data) == 2:
            return self.data[0][0]*self.data[1][1]-self.data[0][1]*self.data[1][0]
        determinant = 0
        for i in range(0,len(self.data)):
            determinant+=(-1)**i*self.data[0][i]*(self.__minor(0,i)).det()
        return determinant


    def transpose(self):
        "Транспонировать матрицу и вернуть результат"
        res = Matrix(self.ncols,self.nrows)
        for i in range(self.ncols):
            for j in range(self.nrows):
                res[(i,j)] = self.data[j][i]
        return res


    def inv(self):
        "Вычислить обратную матрицу и вернуть результат"
        if self.ncols != self.nrows:
            raise ArithmeticError('матрица не квадратная')

        res = Matrix(self.ncols,self.nrows)

        determinant = self.det()
        #special case for 2x2 matrix:
        if len(self.data) == 2:
            res[(0,0)] = self.data[1][1]/determinant
            res[(0,1)] = -1*self.data[0][1]/determinant
            res[(1,0)] = -1*self.data[1][0]/determinant
            res[(1,1)] = self.data[0][0]/determinant
            return res

        cofactors = []
        for row in range(len(self.data)):
            cofactorRow = []
            for col in range(len(self.data)):
                minor = self.__minor(row,col)
                cofactorRow.append(((-1)**(row+col)) * minor.det())
            cofactors.append(cofactorRow)
        for row in range(len(self.data)):
            for col in range(len(self.data)):
                res[(row,col)] = cofactors[row][col]
        res = res.transpose()
        for row in range(len(self.data)):
            for col in range(len(self.data)):
                res[(row,col)] = res[(row,col)]/determinant
        return res


    def tonumpy(self):
        "Приведение к массиву numpy"
        return np.array(self.data)


    def raund(self):
        "raund"
        for i in range(len(self.data)):
            for j in range(len(self.data)):
                self.data[i][j] = round(self.data[i][j], 3)
        return self


def test(matrixA,MatrixB):
    "test"
    A_numpy = matrixA.tonumpy()
    B_numpy = MatrixB.tonumpy()

    C_matrix = matrixA*MatrixB
    C_matrix_np = A_numpy@B_numpy

    C_matrix_t = C_matrix.transpose()
    C_matrix_t_np = C_matrix_np.T

    C_matrix_inv = C_matrix.inv()
    C_matrix_inv_np =  np.linalg.inv(C_matrix_np)


    Deter = C_matrix.det()
    Deter_numpy = np.linalg.det(C_matrix_np)

    assert (np.all(C_matrix.tonumpy().round(5) == C_matrix_np.round(5)))
    assert (np.all(C_matrix_t.tonumpy().round(5) == C_matrix_t_np.round(5)))
    assert (np.all(C_matrix_inv.tonumpy().round(5) == C_matrix_inv_np.round(5)))
    assert (round(Deter,5) == round(Deter_numpy,5))


def load_file(filename):
    "load_file"
    with open(filename, "r") as f:
        input_file = json.load(f)
        matrixA = Matrix.fromDict(input_file["A"])
        matrixB = Matrix.fromDict(input_file["B"])
    return matrixA, matrixB


if __name__ == "__main__":
    FILE = "input_085.json"
    A, B = load_file(FILE)
    test(A,B)
        