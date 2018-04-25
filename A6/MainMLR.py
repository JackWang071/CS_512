import time                 #provides timing for benchmarks
from numpy  import *        #provides complex math and array functions
from sklearn import svm     #provides Support Vector Regression
import csv

#Local files created by me
import mlr
import FromDataFileMLR
import FromFinessFileMLR

class DE_BPSO:
    def __init__(self, numOfPop, numOfFea):
        # Acquires and formats data from Train, Validation, Test .csv files
        self.filedata = FromDataFileMLR.DataFromFile()
        # Performs data analysis on training, validation, and test data
        self.analyzer = FromFinessFileMLR.FitnessResults()
        self.NumIterations = 1000
        self.alpha = 0.5 #starting alpha value
        self.GlobalBestRow = ndarray(numOfFea) #best-fitting population yet found
        self.GlobalBestFitness = 10000 #fitness of GlobalBestRow, initialized very high
        self.VelocityM = ndarray((numOfPop, numOfFea)) # Velocity matrix
        self.LocalBestM = ndarray((numOfPop, numOfFea)) # local best matrix
        self.LocalBestM_Fit = ndarray(numOfPop) # local best matrix fitnesses
    #------------------------------------------------------------------------------
    def CreateInitialVelocity(self, numOfPop, numOfFea):
        # Each element in initial VelocityMatrix is randomly determined
        for i in range(numOfPop):
            for j in range(numOfFea):
                self.VelocityM[i][j] = random.random()
    #------------------------------------------------------------------------------
    def InitializePopulation(self, numOfPop, numOfFea, lmbd = 0.01):
        newpop = ndarray((numOfPop, numOfFea))
        # Each element in initial Population is based on VelocityMatrix values
        # in the same index position.
        for p in range(numOfPop):
            for f in range(numOfFea):
                if self.VelocityM[p][f] <= lmbd:
                    newpop[p][f] = 1
                else:
                    newpop[p][f] = 0
        return newpop
    #------------------------------------------------------------------------------
    # The following creates an output file. Every time a model is created the
    # descriptors of the model, the ame of the model (ex: "MLR" for multiple
    # linear regression of "SVM" support vector machine) the R^2 of training, Q^2
    # of training,R^2 of validation, and R^2 of test is placed in the output file
    def createAnOutputFile(self):
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
    def createANewPopulation(self, numOfPop, numOfFea, OldPopulation, beta=0.004):
        NewPopulation = ndarray((numOfPop, numOfFea))
        # When alpha reaches 0.33, the data mining should be ended
        self.alpha -= (0.17 / self.NumIterations)
        a = 0.5 * (1 + self.alpha)
        b = 1 - beta
        # Each element of NewPopulation will be determined based on VelocityMatrix values
        # compared to a and b
        for i in range(numOfPop):
            for j in range(numOfFea):
                if (self.alpha < self.VelocityM[i][j]) & (self.VelocityM[i][j] <= a):
                    NewPopulation[i][j] = self.LocalBestM[i][j]
                elif (a < self.VelocityM[i][j]) & (self.VelocityM[i][j] <= b):
                    NewPopulation[i][j] = self.GlobalBestRow[j]
                elif (b < self.VelocityM[i][j]) & (self.VelocityM[i][j] <= 1):
                    NewPopulation[i][j] = 1 - OldPopulation[i][j]
                else:
                    NewPopulation[i][j] = OldPopulation[i][j]
        return NewPopulation
    #-------------------------------------------------------------------------------------------
    def FindGlobalBestRow(self):
        IndexOfBest = 0 # represents index position of best-fitness row in LocalBestMatrix
        numOfPop = self.LocalBestM.shape[0]
        # Find the row with best fitness value in LocalBestMatrix
        for i in range(numOfPop):
            if (self.LocalBestM_Fit[i] < self.GlobalBestFitness) and (self.LocalBestM_Fit[i] > 0):
                self.GlobalBestFitness = self.LocalBestM_Fit[i]
                IndexOfBest = i
                print(IndexOfBest, " ")
        # Update GlobalBestRow to the LocalBestMatrix row with best fitness
        copyto(self.GlobalBestRow, self.LocalBestM[IndexOfBest])
    #-------------------------------------------------------------------------------------------
    def UpdateLocalMatrix(self, NewPopulation, NewPopFitness):
        numOfPop = self.LocalBestM.shape[0]
        # Go through each row in LocalBestMatrix
        for i in range(numOfPop):
            # If the ith LocalBestMatrix row has worse fitness than ith NewPopulation row:
            if self.LocalBestM_Fit[i] > NewPopFitness[i]:
                # Update LocalBestMatrix with NewPopulation row
                copyto(self.LocalBestM[i], NewPopulation[i])
                # Update fitness for this LocalBestMatrix row
                self.LocalBestM_Fit[i] = NewPopFitness[i]
                print(self.LocalBestM_Fit[i])
    #-------------------------------------------------------------------------------------------
    def UpdateVelocityMatrix(self, NewPop, F=0.7, CR=0.7):
        numOfPop = self.VelocityM.shape[0]
        numOfFea = self.VelocityM.shape[1]
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
                    self.VelocityM[i][j] = NewPop[r1][j] + (F * (NewPop[r2][j] - NewPop[r3][j]))
                #Otherwise just keep the old value
                else:
                    self.VelocityM[i][j] = self.VelocityM[i][j]
    #-------------------------------------------------------------------------------------------
    def PerformOneMillionIteration(self, numOfPop, numOfFea, population, fitness, model, fileW,
                                   TrainX, TrainY, ValidateX, ValidateY, TestX, TestY):
        NumOfGenerations = 1
        OldPopulation = population
        while (NumOfGenerations < self.NumIterations):
            print(NumOfGenerations)
            population = self.createANewPopulation(numOfPop, numOfFea, OldPopulation)
            fittingStatus, fitness = self.analyzer.validate_model(model,fileW,
                        population, TrainX, TrainY, ValidateX, ValidateY, TestX, TestY)

            self.UpdateLocalMatrix(population, fitness)
            self.FindGlobalBestRow()
            self.UpdateVelocityMatrix(population)

            #print(self.LocalBestM_Fit)

            NumOfGenerations = NumOfGenerations + 1
        return
#end of DE_BPSO class

#--------------------------------------------------------------------------------------------
#Main program
def main():
    # Number of descriptor should be 385 and number of population should be 50 or more
    numOfPop = 50
    numOfFea = 385

    # create an object of Multiple Linear Regression model.
    # The class is located in mlr file
    model = mlr.MLR()
    # Creates new populations and updates VelocityMatrix, LocalBestMatrix, and GlobalBestRow
    dataminer = DE_BPSO(numOfPop, numOfFea)

    # create an output file. Name the object to be FileW 
    fileW = dataminer.createAnOutputFile()

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
    TrainX, TrainY, ValidateX, ValidateY, TestX, TestY = dataminer.filedata.getAllOfTheData()
    TrainX, ValidateX, TestX = dataminer.filedata.rescaleTheData(TrainX, ValidateX, TestX)

    fittingStatus = unfit
    dataminer.CreateInitialVelocity(numOfPop, numOfFea)
    # Creating initial population based on initial VelocityMatrix
    population = dataminer.InitializePopulation(numOfPop,numOfFea)
    # Determining fitness of initial population
    fittingStatus, fitness = dataminer.analyzer.validate_model(model,fileW, population,
        TrainX, TrainY, ValidateX, ValidateY, TestX, TestY)

    # initializing LocalBestMatrix as the initial population
    copyto(dataminer.LocalBestM, population)
    # initializing LocalBestMatrix's fitness values
    copyto(dataminer.LocalBestM_Fit, fitness)
    # finding the GlobalBestRow of the initial population
    dataminer.FindGlobalBestRow()

    dataminer.PerformOneMillionIteration(numOfPop, numOfFea, population, fitness, model, fileW,
                               TrainX, TrainY, ValidateX, ValidateY, TestX, TestY)
#main routine ends in here
#------------------------------------------------------------------------------
main()
#------------------------------------------------------------------------------



