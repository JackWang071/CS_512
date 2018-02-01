import random
import os

def InitMatrix(rows, cols):
    return [[0 for c in range(cols)] for r in range(rows)]

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

def MakeMatrix(OriginalMatrix, ColA, ColB, ColC):
    NewMatrix = InitMatrix(10,3)
    # for loop fills in the elements of NewMatrix with the designated columns from OriginalMatrix
    for col in range(0,2):
        index = ColA
        if (index > ColB):
            index = ColB
        if (index > ColC):
            index = ColB
        for row in range (len(OriginalMatrix)):
            NewMatrix[row][col] = OriginalMatrix[row][index]
    return NewMatrix

def GetThreeRandomNumbers(OriginalMatrix, Num1, Num2, Num3):
    ColA = Num1; ColB = Num2; ColC = Num3
    while(ColA != Num1) & (ColA != Num2) & (ColA != Num3):
        ColA = random.randrange(0,len(OriginalMatrix[0]),1)
    while(ColB != ColA) & (ColB != Num1) & (ColB != Num2) & (ColB != Num3):
        ColB = random.randrange(0,len(OriginalMatrix[0]),1)
    while(ColC != ColB) & (ColC != ColA) & (ColC != Num1) & (ColC != Num2) & (ColC != Num3):
        ColC = random.randrange(0,len(OriginalMatrix[0]),1)
    return ColA, ColB, ColC

def AddingMatrices(Matrix1, Matrix2):
    for row in range(len(Matrix1)):
        for col in range (len(Matrix1[row])):
            Matrix1[row][col] += Matrix2[row][col]
    return Matrix1

def AddingContentOfEachRow(Matrix):
    NewMatrix = InitMatrix(10,1)
    ColTotal = 0
    for row in range (len(Matrix)):
        for col in range (len(Matrix[row])):
            ColTotal += Matrix[row][col]
        NewMatrix[row] = ColTotal
    return NewMatrix

# try implementing quicksort
def SortElements(Vector):
    if len(Vector[0]) == 1:
        NewMatrix = InitMatrix(10,1)
        for row in range(len(Vector)):
            NewMatrix[row][0] = Vector[row][0]
            for row2 in range (row, len(Vector)-1):
                if (NewMatrix[row][0] > Vector[row2][0]):
                    NewMatrix[row][0] = Vector[row2][0]
        return NewMatrix
    else:
        return Vector

def PrintMatrix(Matrix):
    for row in range(len(Matrix)):
        line = ""
        for col in range(len(Matrix[0])):
            line = line + str(Matrix[row][col]) + "\t"
        print(line)

def PrintOutput(OriginalMatrix, Matrix1, Matrix2, Matrix3, Matrix4, Matrix5):
    print("")