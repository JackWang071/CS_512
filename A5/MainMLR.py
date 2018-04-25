import time                 #provides timing for benchmarks
from numpy  import *        #provides complex math and array functions
from sklearn import svm     #provides Support Vector Regression
import csv

#Local files created by me
import mlr
import FromDataFileMLR
import FromFinessFileMLR

class BPSO:
    def __init__(self, numOfPop, numOfFea):
        # Acquires and formats data from Train, Validation, Test .csv files
        self.filedata = FromDataFileMLR.DataFromFile()
        # Performs data analysis on training, validation, and test data
        self.analyzer = FromFinessFileMLR.FitnessResults()
        self.NumIterations = 1000
        self.alpha = 0.5  # starting alpha value
        self.GlobalBestRow = ndarray(numOfFea)  # best-fitting population yet found
        self.GlobalBestFitness = 10000  # fitness of GlobalBestRow, initialized very high
        self.VelocityM = ndarray((numOfPop, numOfFea))  # Velocity matrix
        self.LocalBestM = ndarray((numOfPop, numOfFea))  # local best matrix
        self.LocalBestM_Fit = ndarray(numOfPop)  # local best matrix fitnesses
    #------------------------------------------------------------------------------
    def CreateInitialVelocity(self, numOfPop, numOfFea):
        # Each element in initial VelocityMatrix is randomly determined
        for i in range(numOfPop):
            for j in range(numOfFea):
                self.VelocityM[i][j] = random.random()
    #------------------------------------------------------------------------------
    def getAValidrow(self, numOfFea, eps=0.015):
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
    def Create_A_Population(self, numOfPop, numOfFea):
        # Initializes the first population using getAValidRow
        population = random.random((numOfPop,numOfFea))
        for i in range(numOfPop):
            V = self.getAValidrow(numOfFea)
            for j in range(numOfFea):
                population[i][j] = V[j]
        return population
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
    def createANewPopulation(self, numOfPop, numOfFea, OldPopulation, fitness):
        NewPopulation = ndarray((numOfPop, numOfFea))
        # When alpha reaches 0.33, the data mining should be ended
        self.alpha -= (0.17 / self.NumIterations)
        p = 0.5 * (1 + self.alpha)
        # Each element of NewPopulation will be determined based on VelocityMatrix values
        # compared to p
        for i in range(numOfPop):
            for j in range(numOfFea):
                if self.VelocityM[i][j] <= self.alpha:
                    NewPopulation[i][j] = OldPopulation[i][j]
                elif (self.VelocityM[i][j] > self.alpha) & (self.VelocityM[i][j] <= p):
                    NewPopulation[i][j] = self.LocalBestM[i][j]
                elif (self.VelocityM[i][j] > p) & (self.VelocityM[i][j] <= 1):
                    NewPopulation[i][j] = self.GlobalBestRow[j]
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
    #-------------------------------------------------------------------------------------------
    def UpdateVelocityMatrix(self, NewPop, c1=2, c2=2, inertiaWeight=0.9):
        numOfPop = self.VelocityM.shape[0]
        numOfFea = self.VelocityM.shape[1]
        # Go through element in VelocityMatrix
        for i in range(numOfPop):
            for j in range(numOfFea):
                # Each element will be updated using terms based on the current
                # LocalBestMatrix and GlobalBestRow
                term1 = c1 * random.random() * (self.LocalBestM[i][j] - NewPop[i][j])
                term2 = c2 * random.random() * (self.GlobalBestRow[j] - NewPop[i][j])
                self.VelocityM[i][j]=term1+term2+(inertiaWeight*self.VelocityM[i][j])
    #-------------------------------------------------------------------------------------------
    def PerformOneMillionIteration(self, numOfPop, numOfFea, population, fitness, model, fileW,
                                   TrainX, TrainY, ValidateX, ValidateY, TestX, TestY):
        NumOfGenerations = 1
        OldPopulation = population
        while NumOfGenerations < self.NumIterations:
            population = self.createANewPopulation(numOfPop, numOfFea, OldPopulation, fitness)
            fittingStatus, fitness = self.analyzer.validate_model(model,fileW, population,
                                    TrainX, TrainY, ValidateX, ValidateY, TestX, TestY)

            self.UpdateLocalMatrix(population, fitness)
            self.FindGlobalBestRow()
            self.UpdateVelocityMatrix(population)

            NumOfGenerations = NumOfGenerations + 1
            print(NumOfGenerations)
        return
#end of BPSO class

#--------------------------------------------------------------------------------------------
#Main program
def main():
    # Number of descriptor should be 385 and number of population should be 50 or more
    numOfPop = 50
    numOfFea = 385

    # create an object of Multiple Linear Regression model.
    # The class is located in mlr file
    model = mlr.MLR()
    dataminer = BPSO(numOfPop, numOfFea)

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
    # Initializing and finding the fitness of the first population
    population = dataminer.Create_A_Population(numOfPop,numOfFea)
    fittingStatus, fitness = dataminer.analyzer.validate_model(model,fileW, population,
        TrainX, TrainY, ValidateX, ValidateY, TestX, TestY)

    # Initializing the first VelocityMatrix
    dataminer.CreateInitialVelocity(numOfPop, numOfFea)
    # initializing LocalBestMatrix as the initial population
    copyto(dataminer.LocalBestM, population)
    # initializing LocalBestMatrix's fitness as the initial population's fitness
    copyto(dataminer.LocalBestM_Fit, fitness)
    dataminer.FindGlobalBestRow()

    dataminer.PerformOneMillionIteration(numOfPop, numOfFea, population, fitness, model, fileW,
                               TrainX, TrainY, ValidateX, ValidateY, TestX, TestY)
#main routine ends in here
#------------------------------------------------------------------------------
main()
#------------------------------------------------------------------------------



