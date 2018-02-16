import numpy as np
import sys
import os

def GetDataFromFile(filename, n):
    Matrix = np.fromfile(filename, dtype=int, sep="\t")
    if n*n > Matrix.size:
        print("Not enough elements in " + filename + ".txt.")
        sys.exit()
    else:
        Matrix = Matrix[:(n*n)]
        return np.reshape(Matrix, (n,n))

def MatrixProd(Matrix1, Matrix2):
    if Matrix1.shape[0] == Matrix2.shape[0] & Matrix1.shape[1] == Matrix2.shape[1]:
        return np.matmul(Matrix1, Matrix2)

def MatrixDotProd(Matrix1, Matrix2):
    if Matrix1.shape[0] == Matrix2.shape[0] & Matrix1.shape[1] == Matrix2.shape[1]:
        return np.dot(Matrix1, Matrix2)

def Transpose(OrigMatrix):
    return np.transpose(OrigMatrix)

def OutputFile(M1, M2, M3, M4, M5, M6, M7, M8):
    with open(os.path.join(os.path.dirname(__file__), "Output.txt"), 'w') as fout:
        fout.write("Matrix from file1.txt:" + "\n")
        fout.write(str(M1))
        fout.write("\n" + "Matrix from file2.txt:" + "\n")
        fout.write(str(M2))
        fout.write("\n" + "Product of M1 and M2:" + "\n")
        fout.write(str(M3))
        fout.write("\n" + "Dot Product of M1 and M2:" + "\n")
        fout.write(str(M4))
        fout.write("\n" + "Transposed M1:" + "\n")
        fout.write(str(M5))
        fout.write("\n" + "Transposed M2:" + "\n")
        fout.write(str(M6))
        fout.write("\n" + "Product of Transposed M1 and Transposed M2:" + "\n")
        fout.write(str(M7))
        fout.write("\n" + "Dot Product of Transposed M1 and Transposed M2:" + "\n")
        fout.write(str(M8))