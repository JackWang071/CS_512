import numpy as np
import sys
import os

class SquareMatrix:
    # Reads in an input file and returns an n*n matrix composed of elements from that input file.
    def GetDataFromFile(self, filename, n):
        n = int(n)
        # If n is less than 3, return an error message.
        if n <= 3:
            print("Error: Input out of bounds.")
            sys.exit()
        # File contents are first copied to a Matrix.
        Matrix = np.loadtxt(filename, dtype=int)
        # Returns an error message if the Matrix's dimensions are less than n*n.
        if n*n > np.prod(Matrix.shape):
            print("Not enough elements in " + filename + ".")
            sys.exit()
        else:
            return Matrix[:n, :n]

    # Finds the product of two matrices.
    def MatrixProd(self, Matrix1, Matrix2):
        # Checks that Matrix1 has as many columns as Matrix2 has rows
        if Matrix1.shape[1] == Matrix2.shape[0]:
            return np.matmul(Matrix1, Matrix2)

    # Finds the dot product of two matrices
    def MatrixDotProd(self, Matrix1, Matrix2):
        # Checks that Matrix1 has as many columns as Matrix2 has rows
        if Matrix1.shape[1] == Matrix2.shape[0]:
            return np.dot(Matrix1, Matrix2)

    # Returns the transposed form of the original matrix.
    def Transpose(self, OrigMatrix):
        return np.transpose(OrigMatrix)

    # Divides one matrix by another.
    def DivideMatrix(self, Matrix1, Matrix2):
        # Ensuring that Matrix1 and Matrix2 are both square.
        if (Matrix1.shape[0] == Matrix1.shape[1]) & (Matrix2.shape[0] == Matrix2.shape[1]):
            # Ensuring that Matrix1 and Matrix2 are the same dimensions
            if Matrix1.shape == Matrix2.shape:

                det = np.linalg.det(Matrix2)
                print(det)

                # Attempting to invert Matrix2 to multiply it with Matrix1.
                if np.linalg.det(Matrix2) != 0:
                    # InverseMat2 = np.linalg.inv(Matrix2)
                    # return np.matmul(Matrix1, InverseMat2)
                    np.seterr(divide = 'print')
                    return np.divide(Matrix1, Matrix2)

        return "\tUndefined"

    # Writes the parameter matrix to the output file.
    def OutputFile(self, Matrix = "No matrix", message = "No message"):
        with open(os.path.join(os.path.dirname(__file__), "Output.txt"), 'a') as fout:
            # print(str(message) + "\n" + str(Matrix) + "\n")
            fout.write(str(message) + "\n" + str(Matrix) + "\n")