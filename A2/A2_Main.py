import A2_Defs
import sys

n = input("Please enter a positive number: ")
n = int(n)
if n <= 3:
    print("Error: Input out of bounds.")
    sys.exit()
else:
    M1 = A2_Defs.GetDataFromFile("file1.txt", n)

M2 = A2_Defs.GetDataFromFile("file2.txt", n)
M1_prod_M2 = A2_Defs.MatrixProd(M1,M2)
M1_dotprod_M2 = A2_Defs.MatrixDotProd(M1, M2)
M1_trans = A2_Defs.Transpose(M1)
M2_trans = A2_Defs.Transpose(M2)
M1_trans_prod_M2_trans = A2_Defs.MatrixProd(M1_trans, M2_trans)
M1_trans_dotprod_M2_trans = A2_Defs.MatrixDotProd(M1_trans, M2_trans)

print("Matrix from file1.txt:")
print(M1)
print("Matrix from file2.txt:")
print(M2)
print("Product of M1 and M2:")
print(M1_prod_M2)
print("Dot Product of M1 and M2:")
print(M1_dotprod_M2)
print("Transposed M1:")
print(M1_trans)
print("Transposed M2:")
print(M2_trans)
print("Product of Transposed M1 and Transposed M2:")
print(M1_trans_prod_M2_trans)
print("Dot Product of Transposed M1 and Transposed M2:")
print(M1_trans_dotprod_M2_trans)

A2_Defs.OutputFile(M1, M2, M1_prod_M2, M1_dotprod_M2, M1_trans, M2_trans, M1_trans_prod_M2_trans, M1_trans_dotprod_M2_trans)