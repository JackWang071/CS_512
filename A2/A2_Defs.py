import numpy as np
import sys
import os

# Reads in an input file and returns an n*n matrix composed of elements from that input file.
def GetDataFromFile(filename, n):
    # File contents are first copied to an array to enable simpler counting and manipulation.
    Array = np.fromfile(filename, dtype=int, sep="\t")
    # Returns an error message if the file did not contain enough elements to build an n*n matrix.
    if n*n > Array.size:
        print("Not enough elements in " + filename + ".txt.")
        sys.exit()
    else:
        Matrix = Array[:(n*n)]
        return np.reshape(Matrix, (n,n))

# Finds the product of two matrices.
def MatrixProd(Matrix1, Matrix2):
    # Checks that Matrix1 has as many columns as Matrix2 has rows
    if Matrix1.shape[1] == Matrix2.shape[0]:
        return np.matmul(Matrix1, Matrix2)

# Finds the dot product of two matrices
def MatrixDotProd(Matrix1, Matrix2):
    # Checks that Matrix1 has as many columns as Matrix2 has rows
    if Matrix1.shape[1] == Matrix2.shape[0]:
        return np.dot(Matrix1, Matrix2)

# Returns the transposed form of the original matrix.
def Transpose(OrigMatrix):
    return np.transpose(OrigMatrix)

# Writes all parameter Matrices to an output file.
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