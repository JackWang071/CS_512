import random
import os

def InitMatrix(rows, cols):
    return [[0 for c in range(cols)] for r in range(rows)]

#Getting Data From the Matrix
def GetDataFromDataFile():
    Matrix = InitMatrix(10, 10)
    row = 0; col = 0
    with open(os.path.join(os.path.dirname(__file__), "Data.txt")) as fin:
        for line in fin:
            col = 0
            token = ""
            for c in range(len(line)):
                if (c == len(line) - 1) | (line[c] == " ") | (line[c] == "\t"):
                    # if token is not a blank string, copy it to the appropriate element of Matrix
                    if token != "":
                        Matrix[row][col] = int(token)
                        token = ""
                        col += 1
                else:
                    token = token + line[c]
            row += 1
    return Matrix
#This makes a 10 by 3 matrix by selecting a random of 3 matrixs.
def MakeMatrix(OriginalMatrix, ColA, ColB, ColC):
    NewMatrix = InitMatrix(10,3)
    columns = [ColA, ColB, ColC]
    # first for loop sorts the elements in columns
    for i in range(len(columns)):
        for j in range(i, len(columns)):
            if columns[i] > columns[j]:
                temp = columns[i]
                columns[i] = columns[j]
                columns[j] = temp
    for col in range(len(columns)):
        for row in range(len(OriginalMatrix)):
            NewMatrix[row][col] = OriginalMatrix[row][columns[col]]
    return NewMatrix

def GetThreeRandomNumbers(OriginalMatrix, Num1, Num2, Num3):
    ColA = Num1; ColB = Num2; ColC = Num3
    while(ColA == Num1) | (ColA == Num2) | (ColA == Num3):
        ColA = random.randrange(0,len(OriginalMatrix[0]),1)
    while(ColB == ColA) | (ColB == Num1) | (ColB == Num2) | (ColB == Num3):
        ColB = random.randrange(0,len(OriginalMatrix[0]),1)
    while(ColC == ColB) | (ColC == ColA) | (ColC == Num1) | (ColC == Num2) | (ColC == Num3):
        ColC = random.randrange(0,len(OriginalMatrix[0]),1)
    return ColA, ColB, ColC

def AddingMatrices(Matrix1, Matrix2):
    Matrix3 = InitMatrix(len(Matrix1), len(Matrix1[0]))
    for row in range(len(Matrix1)):
        for col in range (len(Matrix1[row])):
            Matrix3[row][col] = Matrix1[row][col] + Matrix2[row][col]
    return Matrix3

def AddingContentOfEachRow(Matrix):
    NewMatrix = InitMatrix(10,1)
    for row in range (len(Matrix)):
        ColTotal = 0
        for col in range (len(Matrix[row])):
            ColTotal += Matrix[row][col]
        NewMatrix[row] = ColTotal
    return NewMatrix

# try implementing quicksort
def SortElements(Matrix):
    if isinstance(Matrix[0], int):
        Vector = InitMatrix(len(Matrix), 1)
        for r in range(len(Matrix)):
            Vector[r] = Matrix[r]
        for i in range(len(Vector)):
            for j in range(i, len(Vector)):
                if Vector[i] > Vector[j]:
                    temp = Vector[i]
                    Vector[i] = Vector[j]
                    Vector[j] = temp
        return Vector

def PrintMatrix(Matrix):
    if isinstance(Matrix[0], int):
        for row in range(len(Matrix)):
            print(Matrix[row])
    else:
        for row in range(len(Matrix)):
            line = ""
            for col in range(len(Matrix[0])):
                line = line + str(Matrix[row][col]) + "\t"
            print(line)

def PrintOutput(OriginalMatrix, Matrix1, Matrix2, Matrix3, Matrix4, Matrix5):
    PrintMatrix(OriginalMatrix)
    print("\nMatrix 1:")
    PrintMatrix(Matrix1)
    print("\nMatrix 2:")
    PrintMatrix(Matrix2)
    print("\nMatrix 3:")
    PrintMatrix(Matrix3)
    print("\nMatrix 4:")
    PrintMatrix(Matrix4)
    print("\nMatrix 5:")
    PrintMatrix(Matrix5)
