import random
import array

def GetDataFromDataFile():
    Matrix = array.array('i')
    token = ""
    row = 0; col = -1
    with open('Data.txt') as fin:
        for line in fin:
            # Set row to 0 and increment col by 1
            row = 0
            col = col + 1
            # Initialize token as the first character of the line
            token = line[0]
            for chara in line:
                if (chara == " "):
                    # if token is not a blank string, copy it to the appropriate element of Matrix
                    if (token != ""):
                        row = row + 1
                        Matrix[col][row] = int(token)
                        token = ""
                else:
                    token = token + chara
    return Matrix

def MakeMatrix(OriginalMatrix, ColA, ColB, ColC):
    NewMatrix = array.array('i')
    # for loop fills in the elements of NewMatrix with the designated columns from OriginalMatrix
    for col in range(0,2):
        index = ColA
        if (index > ColB):
            index = ColB
        if (index > ColC):
            index = ColB
        for row in range (0, len(OriginalMatrix[index])):
            NewMatrix[col][row] = OriginalMatrix[index][row]
    return NewMatrix

def GetThreeRandomNumbers(OriginalMatrix, Num1, Num2, Num3):
    ColA = Num1; ColB = Num2; ColC = Num3
    while(ColA != Num1 & ColA != Num2 & ColA != Num3):
        ColA = random.randrange(0,len(OriginalMatrix),1)
    while(ColB != ColA & ColB != Num1 & ColB != Num2 & ColB != Num3):
        ColB = random.randrange(0,len(OriginalMatrix),1)
    while(ColC != ColB & ColC != ColA & ColC != Num1 & ColC != Num2 & ColC != Num3):
        ColC = random.randrange(0,len(OriginalMatrix),1)
    return ColA, ColB, ColC

def AddingMatrices(Matrix1, Matrix2):
    for col in Matrix1:
        for row in range (0, len(Matrix1[col])):
            Matrix1[col][row] = Matrix1[col][row] + Matrix2[col][row]
    return Matrix1

def AddingContentOfEachRow(Matrix):
    NewMatrix = array.array('i')
    ColTotal = 0
    for col in range (0, len(Matrix)):
        for row in range (0, len(Matrix[col])):
            ColTotal = ColTotal + Matrix[col][row]
        NewMatrix[col] = ColTotal
    return NewMatrix

# try implementing quicksort
def SortElements(Matrix):
    NewMatrix = array.array('i')
    newInd = 0
    for oldInd in range (0, len(Matrix)):
        NewMatrix[newInd] = Matrix[oldInd]
        for oldInd2 in range (oldInd, len(Matrix)):
            if (NewMatrix[newInd] > Matrix[oldInd2]):
                NewMatrix[newInd] = Matrix[oldInd2]
    return NewMatrix

def PrintMatrix(Matrix):
    for col in range(0, len(Matrix)):
        for row in range(0, len(Matrix)):
            print(Matrix[col][row])

def PrintOutput(OriginalMatrix, Matrix1, Matrix2, Matrix3, Matrix4, Matrix5):
    print("")