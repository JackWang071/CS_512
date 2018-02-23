import A2_Defs
import sys

n = input("Please enter a positive number: ")

MyMat = A2_Defs.SquareMatrix()

M1 = MyMat.GetDataFromFile("file1.txt", n)
M2 = MyMat.GetDataFromFile("file2.txt", n)
M1_prod_M2 = MyMat.MatrixProd(M1,M2)
M1_dotprod_M2 = MyMat.MatrixDotProd(M1, M2)
M1_trans = MyMat.Transpose(M1)
M2_trans = MyMat.Transpose(M2)
M1_trans_prod_M2_trans = MyMat.MatrixProd(M1_trans, M2_trans)
M1_trans_dotprod_M2_trans = MyMat.MatrixDotProd(M1_trans, M2_trans)
M1_dividedby_M2 = MyMat.DivideMatrix(M1, M2)




MyMat.OutputFile(M1, "Matrix from file1.txt:")
MyMat.OutputFile(M2, "Matrix from file2.txt:")
MyMat.OutputFile(M1_prod_M2, "Product of M1 and M2:")
MyMat.OutputFile(M1_dotprod_M2, "Dot Product of M1 and M2:")
MyMat.OutputFile(M1_trans, "Transposed M1:")
MyMat.OutputFile(M2_trans, "Transposed M2:")
MyMat.OutputFile(M1_trans_prod_M2_trans, "Product of Transposed M1 and Transposed M2:")
MyMat.OutputFile(M1_trans_dotprod_M2_trans, "Dot Product of Transposed M1 and Transposed M2:")
MyMat.OutputFile(M1_dividedby_M2, "Quotient of M1 and M2")