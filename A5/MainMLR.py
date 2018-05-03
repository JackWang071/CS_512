import time                 #provides timing for benchmarks
from numpy  import *        #provides complex math and array functions
from sklearn import svm     #provides Support Vector Regression
import csv

#Local files created by me
import mlr
import FromDataFileMLR
import FromFinessFileMLR

#------------------------------------------------------------------------------
def CreateInitialVelocity(numOfPop, numOfFea):
    VelocityM = ndarray((numOfPop, numOfFea))
    # Each element in initial VelocityMatrix is randomly determined
    for i in range(numOfPop):
        for j in range(numOfFea):
            VelocityM[i][j] = random.random()
    return VelocityM
#------------------------------------------------------------------------------
def getAValidrow(numOfFea, eps=0.015):
    # Returns a row with at least three features
    sum = 0
    while (sum < 3):
       V = zeros(numOfFea)
       for j in range(numOfFea):
          r = random.uniform(0,1)
          if (r < eps):
             V[j] = 1
          else:
             V[j] = 0
       sum = V.sum()
    return V
#------------------------------------------------------------------------------
def Create_A_Population(numOfPop, numOfFea):
    # Initializes the first population using getAValidRow
    population = random.random((numOfPop,numOfFea))
    for i in range(numOfPop):
        V = getAValidrow(numOfFea)
        for j in range(numOfFea):
            population[i][j] = V[j]
    return population
#------------------------------------------------------------------------------
# The following creates an output file. Every time a model is created the
# descriptors of the model, the ame of the model (ex: "MLR" for multiple
# linear regression of "SVM" support vector machine) the R^2 of training, Q^2
# of training,R^2 of validation, and R^2 of test is placed in the output file
def createAnOutputFile():
    file_name = None
    algorithm = None

    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    if ( (file_name == None) and (algorithm != None)):
        file_name = "{}_{}_gen{}_{}.csv".format(algorithm.__class__.__name__,
                        algorithm.model.__class__.__name__, algorithm.gen_max,timestamp)
    elif file_name==None:
        file_name = "{}.csv".format(timestamp)
    fileOut = open(file_name, 'w')
    fileW = csv.writer(fileOut)

    fileW.writerow(['Descriptor ID', 'Fitness', 'Model','R2', 'Q2',
            'R2Pred_Validation', 'R2Pred_Test'])

    return fileW
#-------------------------------------------------------------------------------------------
def createANewPopulation(numOfPop, numOfFea, OldPopulation, VelocityM, LocalBestM,
                         GlobalBestRow, alpha, NumIterations):
    NewPopulation = ndarray((numOfPop, numOfFea))
    # When alpha reaches 0.33, the data mining should be ended
    alpha -= (0.17 / NumIterations)
    p = 0.5 * (1 + alpha)
    # Each element of NewPopulation will be determined based on VelocityMatrix values
    # compared to p
    for i in range(numOfPop):
        for j in range(numOfFea):
            if VelocityM[i][j] <= alpha:
                NewPopulation[i][j] = OldPopulation[i][j]
            elif (VelocityM[i][j] > alpha) & (VelocityM[i][j] <= p):
                NewPopulation[i][j] = LocalBestM[i][j]
            elif (VelocityM[i][j] > p) & (VelocityM[i][j] <= 1):
                NewPopulation[i][j] = GlobalBestRow[j]
            else:
                NewPopulation[i][j] = OldPopulation[i][j]
    return NewPopulation
#-------------------------------------------------------------------------------------------
def FindGlobalBestRow(LocalMatrix, LocalMatFitness, GlobalBestRow, GlobalBestFitness):
    IndexOfBest = argmin(LocalMatFitness)
    if GlobalBestFitness > LocalMatFitness[IndexOfBest]:
        # Update GlobalBestRow to the LocalBestMatrix row with best fitness
        GlobalBestRow = LocalMatrix[IndexOfBest].copy()
        GlobalBestFitness = LocalMatFitness[IndexOfBest]
#-------------------------------------------------------------------------------------------
def UpdateLocalMatrix(NewPopulation, NewPopFitness, LocalMatrix, LocalMatFitness):
    numOfPop = LocalMatrix.shape[0]
    # Go through each row in LocalBestMatrix
    for i in range(numOfPop):
        # If the ith LocalBestMatrix row has worse fitness than ith NewPopulation row:
        if LocalMatFitness[i] > NewPopFitness[i]:
            # Update LocalBestMatrix with NewPopulation row
            LocalMatrix[i] = NewPopulation[i].copy()
            # Update fitness for this LocalBestMatrix row
            LocalMatFitness[i] = NewPopFitness[i]
#-------------------------------------------------------------------------------------------
def UpdateVelocityMatrix(NewPop, VelocityM, LocalBestM, GlobalBestRow, c1=2, c2=2, inertiaWeight=0.9):
    numOfPop = VelocityM.shape[0]
    numOfFea = VelocityM.shape[1]
    # Go through element in VelocityMatrix
    for i in range(numOfPop):
        for j in range(numOfFea):
            # Each element will be updated using terms based on the current
            # LocalBestMatrix and GlobalBestRow
            term1 = c1 * random.random() * (LocalBestM[i][j] - NewPop[i][j])
            term2 = c2 * random.random() * (GlobalBestRow[j] - NewPop[i][j])
            VelocityM[i][j]=term1+term2+(inertiaWeight*VelocityM[i][j])
#-------------------------------------------------------------------------------------------
def PerformOneMillionIteration(numOfPop, numOfFea, population, fitness, model, fileW,
                               TrainX, TrainY, ValidateX, ValidateY, TestX, TestY,
                               VelocityM, LocalMatrix, LocalMatFitness, GlobalBestRow,
                               GlobalBestFit, NumIterations):
    NumOfGenerations = 1
    alpha = 0.5
    while NumOfGenerations < NumIterations:
        OldPopulation = population.copy()
        population = createANewPopulation(numOfPop, numOfFea, OldPopulation, VelocityM,
                                          LocalMatrix, GlobalBestRow, alpha, NumIterations)
        fittingStatus, fitness = FromFinessFileMLR.validate_model(model,fileW, population,
                                TrainX, TrainY, ValidateX, ValidateY, TestX, TestY)

        UpdateLocalMatrix(population, fitness, LocalMatrix, LocalMatFitness)
        FindGlobalBestRow(LocalMatrix, LocalMatFitness, GlobalBestRow, GlobalBestFit)
        UpdateVelocityMatrix(population, VelocityM, LocalMatrix, GlobalBestRow)

        NumOfGenerations = NumOfGenerations + 1
        print(NumOfGenerations)
    return

#--------------------------------------------------------------------------------------------
#Main program
def main():
    #Setting the number of iterations
    NumIterations = 10000

    # Number of descriptor should be 385 and number of population should be 50 or more
    numOfPop = 50
    numOfFea = 385

    # create an object of Multiple Linear Regression model.
    # The class is located in mlr file
    model = mlr.MLR()

    # create an output file. Name the object to be FileW
    fileW = createAnOutputFile()

    # we continue exhancing the model; however if after 1000 iteration no
    # enhancement is done, we can quit
    unfit = 1000

    # Final model requirements: The following is used to evaluate each model. The minimum
    # values for R^2 of training should be 0.6, R^2 of Validation should be 0.5 and R^2 of
    # test should be 0.5
    R2req_train    = .6
    R2req_validate = .5
    R2req_test     = .5

    # getAllOfTheData is in FromDataFileMLR file. The following places the data
    # (training data, validation data, and test data) into associated matrices
    TrainX, TrainY, ValidateX, ValidateY, TestX, TestY = FromDataFileMLR.getAllOfTheData()
    TrainX, ValidateX, TestX = FromDataFileMLR.rescaleTheData(TrainX, ValidateX, TestX)

    fittingStatus = unfit
    # Initializing and finding the fitness of the first population
    population = Create_A_Population(numOfPop,numOfFea)
    fittingStatus, fitness = FromFinessFileMLR.validate_model(model,fileW, population,
        TrainX, TrainY, ValidateX, ValidateY, TestX, TestY)

    # Initializing GlobalBestRow
    GlobalBestRow = zeros((numOfFea))
    GlobalBestFitness = unfit
    # Initializing the first VelocityMatrix
    VelocityM = CreateInitialVelocity(numOfPop, numOfFea)
    # initializing LocalBestMatrix as the initial population
    LocalBestM = population.copy()
    # initializing LocalBestMatrix's fitness as the initial population's fitness
    LocalBestM_Fit = fitness.copy()
    FindGlobalBestRow(LocalBestM, LocalBestM_Fit, GlobalBestRow, GlobalBestFitness)

    PerformOneMillionIteration(numOfPop, numOfFea, population, fitness, model, fileW,
                               TrainX, TrainY, ValidateX, ValidateY, TestX, TestY,
                               VelocityM, LocalBestM, LocalBestM_Fit, GlobalBestRow,
                               GlobalBestFitness, NumIterations)
#main routine ends in here
#------------------------------------------------------------------------------
main()
#------------------------------------------------------------------------------
