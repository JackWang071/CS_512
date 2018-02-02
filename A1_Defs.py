import random
import os

# Initializes a Matrix of any size
def InitMatrix(rows, cols):
    return [[0 for c in range(cols)] for r in range(rows)]

# Copies data from an outside file into a Matrix
def GetDataFromDataFile():
    Matrix = InitMatrix(10, 10)
    row = 0; col = 0
    # Opens the correct path to Data.txt
    with open(os.path.join(os.path.dirname(__file__), "Data.txt")) as fin:
        # First for loop reads in each line from fin
        for line in fin:
            col = 0
            token = ""
            # Nested second loop reads in each character from each line
            for c in range(len(line)):
                # If the current character is a whitespace or the end of the line has been reached:
                if (c == len(line) - 1) | (line[c] == " ") | (line[c] == "\t"):
                    # Copy token to the next element of the current Matrix row and reset it to a blank string
                    if token != "":
                        Matrix[row][col] = int(token)
                        token = ""
                        col += 1
                # Otherwise concatenate the current character to token
                else:
                    token = token + line[c]
            row += 1
    return Matrix

# Creates a Matrix from three columns of an original
def MakeMatrix(OriginalMatrix, ColA, ColB, ColC, sortorder):
    NewMatrix = InitMatrix(10,3)
    columns = [ColA, ColB, ColC]
    # first for loop copies the selected columns from OriginalMatrix to NewMatrix
    for col in range(len(columns)):
        for row in range(len(OriginalMatrix)):
            NewMatrix[row][col] = OriginalMatrix[row][columns[col]]
    SortingCol = InitMatrix(len(NewMatrix),1)
    # second for loop sorts the elements of each column of NewMatrix
    for col in range(len(NewMatrix[0])):
        for row in range(len(NewMatrix)):
            SortingCol[row] = NewMatrix[row][col]
        SortingCol = SortElements(SortingCol, sortorder)
        for row in range(len(NewMatrix)):
            NewMatrix[row][col] = SortingCol[row]
    return NewMatrix

# Generates three random non-equal numbers, with an option to avoid three other numbers
def GetThreeRandomNumbers(OriginalMatrix, Num1, Num2, Num3):
    ColA = Num1; ColB = Num2; ColC = Num3
    while(ColA == Num1) | (ColA == Num2) | (ColA == Num3):
        ColA = random.randrange(0,len(OriginalMatrix[0]),1)
    while(ColB == ColA) | (ColB == Num1) | (ColB == Num2) | (ColB == Num3):
        ColB = random.randrange(0,len(OriginalMatrix[0]),1)
    while(ColC == ColB) | (ColC == ColA) | (ColC == Num1) | (ColC == Num2) | (ColC == Num3):
        ColC = random.randrange(0,len(OriginalMatrix[0]),1)
    return ColA, ColB, ColC

# Adds two matrices together and returns the result as a third matrix
def AddingMatrices(Matrix1, Matrix2):
    Matrix3 = InitMatrix(len(Matrix1), len(Matrix1[0]))
    # Nested for loops adds the elements of each Matrix and copies the result to Matrix3
    for row in range(len(Matrix1)):
        for col in range (len(Matrix1[row])):
            Matrix3[row][col] = Matrix1[row][col] + Matrix2[row][col]
    return Matrix3

# Adds together all the elements in each of this Matrix's rows and returns the result as a new matrix
def AddingContentOfEachRow(Matrix):
    NewMatrix = InitMatrix(10,1)
    for row in range (len(Matrix)):
        ColTotal = 0
        for col in range (len(Matrix[row])):
            ColTotal += Matrix[row][col]
        NewMatrix[row] = ColTotal
    return NewMatrix

# Sorts the elements of the parameter Matrix in ascending or descending order
def SortElements(Matrix, order):
    if isinstance(Matrix[0], int):
        Vector = InitMatrix(len(Matrix), 1)
        # Copying all Matrix elements to Vector
        for r in range(len(Matrix)):
            Vector[r] = Matrix[r]
        # Iterating through all elements of Vector to sort them
        for i in range(len(Vector)):
            for j in range(i, len(Vector)):
                # If order is A or a, sort elements in ascending order
                if ((order == 'A') | (order == 'a')) & (Vector[i] > Vector[j]):
                    temp = Vector[i]
                    Vector[i] = Vector[j]
                    Vector[j] = temp
                # If order is D or d, sort elements in descending order
                if ((order == 'D') | (order == 'd')) & (Vector[i] < Vector[j]):
                    temp = Vector[i]
                    Vector[i] = Vector[j]
                    Vector[j] = temp
        return Vector

# Prints each element of the parameter Matrix
def PrintMatrix(Matrix):
    # Matrix is a vector, then run a for loop that iterates down each element
    if isinstance(Matrix[0], int):
        for row in range(len(Matrix)):
            print(Matrix[row])
    # Otherwise run nested for loops that concatenate each row's elements into one string and print that string
    else:
        for row in range(len(Matrix)):
            line = ""
            for col in range(len(Matrix[0])):
                line = line + str(Matrix[row][col]) + "\t"
            print(line)

# Calls PrintMatrix() for each Matrix in the parameters.
def PrintOutput(OriginalMatrix, Matrix1, Matrix2, Matrix3, Matrix4, Matrix5):
    print("Original Matrix:")
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