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
def InitializePopulation(numOfPop, numOfFea, VelocityM, lmbd = 0.01):
    newpop = ndarray((numOfPop, numOfFea))
    # Each element in initial Population is based on VelocityMatrix values
    # in the same index position.
    for p in range(numOfPop):
        for f in range(numOfFea):
            if VelocityM[p][f] <= lmbd:
                newpop[p][f] = 1
            else:
                newpop[p][f] = 0
        if sum(newpop[p]) < 3:
            newpop[p] = getAValidrow(numOfFea)
    return newpop
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
                         GlobalBestRow, NumIterations, alpha, beta=0.004):
    NewPopulation = OldPopulation.copy()
    # When alpha reaches 0.33, the data mining should be ended
    alpha -= (0.17 / NumIterations)
    a = 0.5 * (1 + alpha)
    b = 1 - beta
    popchanges = 0
    # Each element of NewPopulation will be determined based on VelocityMatrix values
    # compared to a and b
    for i in range(numOfPop):
        for j in range(numOfFea):
            if (alpha < VelocityM[i][j]) & (VelocityM[i][j] <= a):
                NewPopulation[i][j] = LocalBestM[i][j]
            elif (a < VelocityM[i][j]) & (VelocityM[i][j] <= b):
                NewPopulation[i][j] = GlobalBestRow[j]
            elif (b < VelocityM[i][j]) & (VelocityM[i][j] <= 1):
                NewPopulation[i][j] = 1 - OldPopulation[i][j]
            else:
                NewPopulation[i][j] = OldPopulation[i][j]

            if NewPopulation[i][j] != OldPopulation[i][j]:
                popchanges += 1
    print(sum(NewPopulation))
    return alpha, popchanges, NewPopulation
#-------------------------------------------------------------------------------------------
def FindGlobalBestRow(GlobalBestRow, GlobalBestFitness, LocalBestM, LocalBestM_Fit):
    IndexOfBest = argmin(LocalBestM_Fit)
    if GlobalBestFitness > LocalBestM_Fit[IndexOfBest]:
        # Update GlobalBestRow to the LocalBestMatrix row with best fitness
        GlobalBestRow = LocalBestM[IndexOfBest].copy()
        GlobalBestFitness = LocalBestM_Fit[IndexOfBest]
    return GlobalBestRow, GlobalBestFitness
#-------------------------------------------------------------------------------------------
def UpdateLocalMatrix(NewPopulation, NewPopFitness, LocalBestM, LocalBestM_Fit):
    numOfPop = LocalBestM.shape[0]
    # Go through each row in LocalBestMatrix
    changes = 0
    for i in range(numOfPop):
        # If the ith LocalBestMatrix row has worse fitness than ith NewPopulation row:
        if LocalBestM_Fit[i] > NewPopFitness[i]:
            # Update LocalBestMatrix with NewPopulation row
            LocalBestM[i] = NewPopulation[i].copy()
            # Update fitness for this LocalBestMatrix row
            LocalBestM_Fit[i] = NewPopFitness[i]

            changes += 1

    print(changes)
    return LocalBestM, LocalBestM_Fit
#-------------------------------------------------------------------------------------------
def UpdateVelocityMatrix(VelocityM, NewPop, F=0.7, CR=0.7):
    numOfPop = VelocityM.shape[0]
    numOfFea = VelocityM.shape[1]
    avgV = 0
    # Go through each row in VelocityMatrix
    for i in range(numOfPop):
        # Ensuring that values of r1, r2, and r3 are all random and distinct
        # Each value will indicate a row from NewPop
        # Each row will be used to generate updated values for VelocityMatrix
        while True:
            r1 = random.randint(0, numOfPop)
            if r1 != i:
                break
        while True:
            r2 = random.randint(0, numOfPop)
            if r2 != i & r2 != r1:
                break
        while True:
            r3 = random.randint(0, numOfPop)
            if r3 != i & r3 != r2 & r3 != r1:
                break
        #For every element in the ith row of Velocity Matrix:
        for j in range(numOfFea):
            #If random() returns a value under CR, update this element using this equation
            if random.random() < CR:
                VelocityM[i][j] = NewPop[r1][j] + (F * (NewPop[r2][j] - NewPop[r3][j]))
            #Otherwise just keep the old value
            else:
                VelocityM[i][j] = VelocityM[i][j]
            avgV += VelocityM[i][j]
    return VelocityM, (avgV / (numOfPop * numOfFea))
#-------------------------------------------------------------------------------------------
def PerformOneMillionIteration(numOfPop, numOfFea, population, fitness, model, fileW,
                               TrainX, TrainY, ValidateX, ValidateY, TestX, TestY,
                               NumIterations, VelocityM, LocalMat, LocalMatFit,
                               GlobalBestRow, GlobalBestFitness):
    NumOfGenerations = 1
    alpha = 0.5
    waittime = 0
    #OldPopulation = population
    while (NumOfGenerations < NumIterations):
        print("Generation", end=' ')
        print(NumOfGenerations)

        OldPopulation = population.copy()
        alpha, popchanges, population = createANewPopulation(numOfPop, numOfFea,
                                        OldPopulation, VelocityM, LocalMat,
                                        GlobalBestRow, NumIterations, alpha)
        fittingStatus, fitness = FromFinessFileMLR.validate_model(model,fileW,
                    population, TrainX, TrainY, ValidateX, ValidateY, TestX, TestY)

        LocalMat, LocalMatFit = UpdateLocalMatrix(population, fitness,
                                                  LocalMat, LocalMatFit)
        GlobalBestRow, GlobalBestFitness = FindGlobalBestRow(GlobalBestRow,
                                            GlobalBestFitness, LocalMat, LocalMatFit)
        VelocityM, avgV = UpdateVelocityMatrix(VelocityM, population)

        print(popchanges)
        print(waittime)

        print(avgV)

        # If population models have not changed much in a while, scatter the models
        if popchanges < 5:
            waittime = waittime + 1
            if waittime >= 4:
                # self.CreateInitialVelocity(numOfPop, numOfFea)
                population = InitializePopulation(numOfPop, numOfFea, VelocityM)
                waittime = 0
        elif waittime > 0:
            waittime = 0

        NumOfGenerations = NumOfGenerations + 1
    return

#--------------------------------------------------------------------------------------------
#Main program
def main():
    NumIterations = 1000

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

    # Initializing GlobalBestRow
    GlobalBestRow = zeros((numOfFea))
    GlobalBestFitness = unfit
    # Creating initial Velocity Matrix
    VelocityM = CreateInitialVelocity(numOfPop, numOfFea)
    # Creating initial population based on initial VelocityMatrix
    population = InitializePopulation(numOfPop,numOfFea, VelocityM)
    # Determining fitness of initial population
    fittingStatus, fitness = FromFinessFileMLR.validate_model(model,fileW, population,
        TrainX, TrainY, ValidateX, ValidateY, TestX, TestY)

    # initializing LocalBestMatrix as the initial population
    LocalBestM = population.copy()
    # initializing LocalBestMatrix's fitness values
    LocalBestM_Fit = fitness.copy()
    # finding the GlobalBestRow of the initial population
    FindGlobalBestRow(GlobalBestRow, GlobalBestFitness, LocalBestM, LocalBestM_Fit)

    PerformOneMillionIteration(numOfPop, numOfFea, population, fitness, model, fileW,
                               TrainX, TrainY, ValidateX, ValidateY, TestX, TestY,
                               NumIterations, VelocityM, LocalBestM, LocalBestM_Fit,
                               GlobalBestRow, GlobalBestFitness)
#main routine ends in here
#------------------------------------------------------------------------------
main()
#------------------------------------------------------------------------------