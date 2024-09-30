# -*- coding: utf-8 -*-
"""
Created on Fri Mar  20 16:53 2020

@author: Omer
"""
### New encoder / decoder implementation using numba + cuda
#if you uncomment the next line the world will end
from typing import Iterator
import numpy as np
import os
import time
#import math
import copy
import operator
import math
import concurrent.futures
from multiprocessing import Lock

import threading
compiler_lock = threading.Lock()

# Trying an adapted version of https://github.com/ContinuumIO/numbapro-examples/blob/master/multigpu/multigpu_mt.py
compilerLock = Lock()
from numba import cuda, float32, int32


projectDir = os.environ.get('LDPC')
if projectDir == None:
    import pathlib
    projectDir = pathlib.Path(__file__).parent.absolute()

projectDirEvals = str(projectDir) + "evaluations/"
import fileHandler
import common

import sys
sys.path.insert(1, projectDir)

def evaluateCodeCuda(seed, SNRpoints, numberOfIterations, parityMatrix, numOfTransmissions, G = 'None' , cudaDeviceNumber = 0):
    
    cuda.select_device(cudaDeviceNumber)
    device = cuda.get_current_device()
    print("*** debugging mp issues: "+ str(device))
    print("*** debugging mp issues: "+ str(cuda.gpus))
    
    LDPC_LOCAL_PRNG = np.random.RandomState(7134066)
    LDPC_MAX_SEED = 2**31 - 1
    LDPC_SEED_DATA_TYPE = np.int64
    # Omer Sella: in an ideal world, the value written in BIG_NUMBER would be the maximal value of float32, where float32 is on the GPU.
    BIG_NUMBER = float32(10000)
    MATRIX_DIM0 = np.int32(1022)
    MATRIX_DIM1 = 8176
    CODEWORDSIZE = MATRIX_DIM1
    VECTOR_DIM = MATRIX_DIM0
    SHARED_SIZE_0 = 1024 # AND NOT MATRIX_DIM0
    SHARED_SIZE_1 = 8196
    CONST1 = MATRIX_DIM0
    CONST2 = MATRIX_DIM1
    THREADS_PER_BLOCK = 512
    BLOCKS_PER_GRID_DIM0 = (MATRIX_DIM0 + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK
    BLOCKS_PER_GRID_DIM1 = (MATRIX_DIM1 + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK
    LDPC_CUDA_INT_DATA_TYPE = np.int32

    SHARED_MEM_VERTICAL_RECUTION = 512
    THREADS_PER_BLOCK_VERTICAL_SUM = (512, 1)

    BLOCKS_PER_GRID_X_VERTICAL_SUM = math.ceil(512 / THREADS_PER_BLOCK_VERTICAL_SUM[0])
    BLOCKS_PER_GRID_Y_VERTICAL_SUM = math.ceil(MATRIX_DIM1 / THREADS_PER_BLOCK_VERTICAL_SUM[1])
    BLOCKS_PER_GRID_VERTICAL_SUM = (BLOCKS_PER_GRID_X_VERTICAL_SUM, BLOCKS_PER_GRID_Y_VERTICAL_SUM)

    SHARED_MEMORY_SIZE_LOCATE_TWO_SMALLEST_HORIZONTAL = 1024
    THREADS_PER_BLOCK_LOCATE_TWO_SMALLEST_HORIZONTAL_2D = (1,1022)
    BLOCKS_PER_GRID_X_LOCATE_TWO_SMALLEST_HORIZONTAL_2D = math.ceil(MATRIX_DIM0 / THREADS_PER_BLOCK_LOCATE_TWO_SMALLEST_HORIZONTAL_2D[0])
    BLOCKS_PER_GRID_Y_LOCATE_TWO_SMALLEST_HORIZONTAL_2D = 1 #math.ceil(MATRIX_DIM1 / THREADS_PER_BLOCK_LOCATE_TWO_SMALLEST_HORIZONTAL_2D[1])
    BLOCKS_PER_GRID_LOCATE_TWO_SMALLEST_HORIZONTAL_2D = (BLOCKS_PER_GRID_X_LOCATE_TWO_SMALLEST_HORIZONTAL_2D, BLOCKS_PER_GRID_Y_LOCATE_TWO_SMALLEST_HORIZONTAL_2D)

    THREADS_PER_BLOCK_PRODUCE_NEW_MATRIX_2D = (2,511)
    # (511,2) 576.11 micro
    # (1022,1) 724.48 micro
    # (2,511) 472.58 micro
    BLOCKS_PER_GRID_X_PRODUCE_NEW_MATRIX_2D = math.ceil(MATRIX_DIM0 / THREADS_PER_BLOCK_PRODUCE_NEW_MATRIX_2D[0])
    BLOCKS_PER_GRID_Y_PRODUCE_NEW_MATRIX_2D = math.ceil(MATRIX_DIM1 / THREADS_PER_BLOCK_PRODUCE_NEW_MATRIX_2D[1])
    BLOCKS_PER_GRID_PRODUCE_NEW_MATRIX_2D = (BLOCKS_PER_GRID_X_PRODUCE_NEW_MATRIX_2D, BLOCKS_PER_GRID_Y_PRODUCE_NEW_MATRIX_2D)
    THREADS_PER_BLOCK_MATRIX_MINUS_2D = (1,1022)
    # (2,511) 295 micro
    # (511,2) 1.64 mili
    # (2,1022) Doesn't work
    # (1,1022) 276 micro
    #
    BLOCKS_PER_GRID_X_MATRIX_MINUS_2D = math.ceil(MATRIX_DIM0 / THREADS_PER_BLOCK_MATRIX_MINUS_2D[0])
    BLOCKS_PER_GRID_Y_MATRIX_MINUS_2D = math.ceil(MATRIX_DIM1 / THREADS_PER_BLOCK_MATRIX_MINUS_2D[1])
    BLOCKS_PER_GRID_MATRIX_MINUS_2D = (BLOCKS_PER_GRID_X_MATRIX_MINUS_2D, BLOCKS_PER_GRID_Y_MATRIX_MINUS_2D)

    THREADS_PER_BLOCK_BINARY_CALC = (1022,1)
    BLOCKS_PER_GRID_X_BINARY = math.ceil(MATRIX_DIM0 / THREADS_PER_BLOCK_BINARY_CALC[0])
    BLOCKS_PER_GRID_Y_BINARY = math.ceil(MATRIX_DIM1 / THREADS_PER_BLOCK_BINARY_CALC[1])
    BLOCKS_PER_GRID_BINARY_CALC = (BLOCKS_PER_GRID_X_BINARY, BLOCKS_PER_GRID_Y_BINARY)



    #https://docs.microsoft.com/en-us/sysinternals/downloads/process-explorer
    #https://en.wikipedia.org/wiki/Thread_block_(CUDA_programming)
    #https://numba.pydata.org/numba-doc/dev/cuda/examples.html

    #Omer Sella: see the following link regarding numba cuda caching
    #https://github.com/numba/numba/issues/1711
    
    
    with compilerLock:
        ############################
        @cuda.jit(device = True)
        def twoElementsLoad(lhs, lhsIndex, lhsMask, rhs, rhsIndex, rhsMask):
            lhsLocal = lhs
            rhsLocal = rhs
            if lhsMask == 0:
                lhsLocal = BIG_NUMBER
            if rhsMask == 0:
                rhsLocal = BIG_NUMBER
            if lhsLocal < rhsLocal:
                smallest = lhsLocal
                secondSmallest = rhsLocal
                argSmallest = lhsIndex
            else:
                smallest = rhsLocal
                secondSmallest = lhsLocal
                argSmallest = rhsIndex
            return smallest, secondSmallest, argSmallest


        @cuda.jit(device = True)
        def twoElementsMergeSort(lhsSmallest, lhsSecondSmallest, argminLhs, rhsSmallest, rhsSecondSmallest, argminRhs):
            if (lhsSmallest < rhsSmallest):
                resultSmallest = lhsSmallest
                resultSecondSmallest = min(rhsSmallest, lhsSecondSmallest)
                resultArgmin = argminLhs
            else:
                resultSmallest = rhsSmallest
                resultSecondSmallest = min(lhsSmallest, rhsSecondSmallest)
                resultArgmin = argminRhs
            return resultSmallest, resultSecondSmallest, resultArgmin
      

        @cuda.jit
        def locateTwoSmallestHorizontal2DV2(M, parityVector, smallest, secondSmallest, argmin):
        # Omer Sella: this kernel is highly specific to the 8176 X 1022 size.
        # It is intended to be used in blocks of 1X1022. Possibly could work for 2X1022 - untested.
            partialSmallest = cuda.shared.array(SHARED_MEMORY_SIZE_LOCATE_TWO_SMALLEST_HORIZONTAL, dtype = float32)
            partialSecondSmallest = cuda.shared.array(SHARED_MEMORY_SIZE_LOCATE_TWO_SMALLEST_HORIZONTAL, dtype = float32)
            argminSmallest = cuda.shared.array(SHARED_MEMORY_SIZE_LOCATE_TWO_SMALLEST_HORIZONTAL, dtype = float32)
            if (cuda.threadIdx.x == 0):
                partialSmallest[1022] = BIG_NUMBER
                partialSmallest[1023] = BIG_NUMBER
                partialSecondSmallest[1023] = BIG_NUMBER
                partialSecondSmallest[1023] = BIG_NUMBER
            cuda.syncthreads()
            row,col = cuda.grid(2)
            threadIdxX = cuda.threadIdx.y
            offset = cuda.blockDim.y
            i = cuda.blockIdx.y * cuda.blockDim.y * 2 + cuda.threadIdx.y
            i_offset = i + offset
            partialSmallest[threadIdxX], partialSecondSmallest[threadIdxX], argminSmallest[threadIdxX] = twoElementsLoad(abs(M[row, i]), i, parityVector[row, i], abs(M[row, i_offset]), i_offset, parityVector[row, i_offset])
            for j in range(1,4):
                i = j * cuda.blockDim.y * 2 + cuda.threadIdx.y
                a,b,c = twoElementsLoad(abs(M[row, i]), i, parityVector[row, i], abs(M[row, i + offset]), i + offset, parityVector[row, i + offset])
                partialSmallest[threadIdxX], partialSecondSmallest[threadIdxX], argminSmallest[threadIdxX] = twoElementsMergeSort(partialSmallest[threadIdxX], partialSecondSmallest[threadIdxX], argminSmallest[threadIdxX], a, b, c)
                cuda.syncthreads()
            s = 512
            if (threadIdxX < s):
                partialSmallest[threadIdxX], partialSecondSmallest[threadIdxX], argminSmallest[threadIdxX] = twoElementsMergeSort(partialSmallest[threadIdxX], partialSecondSmallest[threadIdxX], argminSmallest[threadIdxX], partialSmallest[threadIdxX + s], partialSecondSmallest[threadIdxX + s], argminSmallest[threadIdxX + s])
                cuda.syncthreads()
            s = 256
            while (s > 0):
                if (threadIdxX < s):
                    partialSmallest[threadIdxX], partialSecondSmallest[threadIdxX], argminSmallest[threadIdxX] = twoElementsMergeSort(partialSmallest[threadIdxX], partialSecondSmallest[threadIdxX], argminSmallest[threadIdxX], partialSmallest[threadIdxX + s], partialSecondSmallest[threadIdxX + s], argminSmallest[threadIdxX + s])
                    cuda.syncthreads()
                s = s // 2
            if (cuda.threadIdx.y == 0):
                smallest[row] = partialSmallest[0]#cuda.blockIdx.x#
                secondSmallest[row] = partialSecondSmallest[0]
                argmin[row] = argminSmallest[0]
                cuda.syncthreads()
            ############################
            return


        @cuda.jit(device = True)
        def extendedSign(operand):
            if operand < 0:
                return -1
            else:
                return 1
    
        @cuda.jit
        def signReduceHorizontal(matrix, signVector):
            pos = cuda.grid(1)
            localSign = 1        
            for j in range(MATRIX_DIM1):
                localSign = localSign * extendedSign(matrix[pos,j])
            signVector[pos] = localSign
            cuda.syncthreads()
            return

        # Omer Sella: this function takes a matrix as an input and finds :
        # 1. The two smallest elements in every row (in absolute values)
        # 2. The location of (the) minimum.
        # 3. The number of negatives in every row
        @cuda.jit
        def findMinimaAndNumberOfNegatives(parityMatrix, matrix, smallest, secondSmallest, locationOfMinimum):
            pos = cuda.grid(1)
            # Omer Sella: init local variables
            localLocationOfMinimum = -1
            localSmallest = BIG_NUMBER
            localSecondSmallest = BIG_NUMBER
            localNumberOfNegatives =  0
            # First implementation: go over the row and extract the information
            for j in range(MATRIX_DIM1):
                if parityMatrix[pos,j] == 1:
                    element = matrix[pos,j]
                    if element < 0:
                        element = -1 * matrix[pos,j]
                    else:
                        pass
                
                    if element < localSmallest:
                        localSecondSmallest = localSmallest
                        localSmallest = element
                        localLocationOfMinimum = j
                    elif element < localSecondSmallest:
                        localSecondSmallest = element
                    else:
                        pass
                else:
                    pass
            # Omer Sella: now communicate thread specific results into the arrays
            smallest[pos] = localSmallest
            secondSmallest[pos] = localSecondSmallest
            locationOfMinimum[pos] = localLocationOfMinimum
            cuda.syncthreads()
            return


        @cuda.jit
        def produceNewMatrix2D(parityMatrix, matrix, smallest, secondSmallest, locationOfMinimum, signProduct, newMatrix):
            i,j = cuda.grid(2)
            localSecondSmallest = secondSmallest[i]
            localSmallest = smallest[i]
            localLocationOfMinimum = locationOfMinimum[i]
            localSignProduct = signProduct[i]
            localMask = parityMatrix[i,j]
            if matrix[i,j] >= 0:
                localSign = 1
            else:
                localSign = -1
            temp = 0.0
            if j == localLocationOfMinimum:
                temp = localSecondSmallest
            else:
                temp = localSmallest
            newMatrix[i,j] = localMask * temp * localSignProduct * localSign
            return


        @cuda.jit
        def maskedFanOut(mask, vector, matrix):
            pos = cuda.grid(1)
            if pos > MATRIX_DIM1:
                return
            localValue = vector[pos]
            cuda.syncthreads()
            for k in range(MATRIX_DIM0):
            #matrix[k,pos] = vector[pos]
                if mask[k,pos] == 1:
                    matrix[k,pos] = localValue
                else:
                    matrix[k,pos] = 0.0
            cuda.syncthreads()
            return



        @cuda.jit
        def sumReduction2DVertical(M, v_r):
            partial_sum = cuda.shared.array(SHARED_MEM_VERTICAL_RECUTION, dtype = float32)
            row, col = cuda.grid(2) #cuda.blockIdx.x * (cuda.blockDim.x * 2) + cuda.threadIdx.x
            if ((row + cuda.blockDim.x) < M.shape[0]):
                partial_sum[cuda.threadIdx.x] = M[row, col] + M[row + cuda.blockDim.x, col]
            else:
                partial_sum[cuda.threadIdx.x] = M[row, col]
            cuda.syncthreads()
            #// Iterate of log base 2 the block dimension
            s = cuda.blockDim.x // 2
            while (s >= 1):
                if (cuda.threadIdx.x < s):
                    partial_sum[cuda.threadIdx.x] += partial_sum[cuda.threadIdx.x + s]
                s = s // 2
                cuda.syncthreads()
            #// Let the thread 0 for this block write it's result to main memory
            #// Result is inexed by this block
            if (cuda.threadIdx.x == 0):
                v_r[col] = partial_sum[0]
                cuda.syncthreads()
            return

        @cuda.jit
        def matrixSumVertical(matrix, result):
            pos = cuda.grid(1)
            if pos >= MATRIX_DIM1:
                return
            temp = 0.0
            for k in range(MATRIX_DIM0):
                temp = temp + matrix[k,pos]
            cuda.syncthreads()
            result[pos] = temp
            return
        
        @cuda.jit
        def slicerCuda(vector, slicedVector):
        ## Omer Sella: slicer puts a threshold, everything STRICTLY above 0 is translated to 1,  otherwise 0 (including equality). Do not confuse with the reserved function name slice !
            pos = cuda.grid(1)
            if pos >= MATRIX_DIM1:
                return
            if vector[pos] > 0:
                slicedVector[pos] = 1
            else:
                slicedVector[pos] = 0 
            cuda.syncthreads()
            return

    
        @cuda.jit        
        def resetVector(v):
            k = cuda.grid(1)
            v[k] = 0.0
            return
    


        @cuda.jit
        def calcBinaryProduct2(parityMatrix, binaryVector, resultVector):
            i,j = cuda.grid(2)
            if binaryVector[j] != 0:
                cuda.atomic.add(resultVector, i, parityMatrix[i,j])
            return

        @cuda.jit        
        def mod2Vector(v):
            k = cuda.grid(1)
            ## OSS 10/11/2021 added a boundary check, there seems to have been a bug caused by access to uncharted memory at v[MATRIX_DIM0]
            if k >= MATRIX_DIM0:
                return
            v[k] = v[k] % 2
            return
    

        @cuda.jit
        def numberOfNegativesToProductOfSigns(numberOfNegatives_device,productOfSigns_device):
            pos = cuda.grid(1)
            if pos >= MATRIX_DIM0:
                return
            else:
                productOfSigns_device[pos] = -1 * ( ( 2 * (numberOfNegatives_device[pos] % 2) ) - 1)
            return
        @cuda.jit
        def cudaPlusDim1(softVector_device,fromChannel_device):
            pos = cuda.grid(1)
            if pos >= MATRIX_DIM1:
                return
            else:
                softVector_device[pos] += fromChannel_device[pos]
            return


        @cuda.jit
        def cudaMatrixMinus2D(A,B):
            i, j = cuda.grid(2)
            A[i,j] -= B[i,j]
            return

        @cuda.jit
        def checkIsCodeword(vector, result):
            sdata = cuda.shared.array(MATRIX_DIM1, dtype = float32)
            pos = cuda.grid(1)
            if pos >= MATRIX_DIM0:
                return
            sdata[pos] = 0
            if vector[pos] != 0:
                sdata[0] = sdata[0] + vector[pos]
                cuda.syncthreads()
            cuda.syncthreads()
            if pos == 0:
                result[0] = sdata[0]
            return

        @cuda.jit
        def numberOfNonZeros(vector, result):
            pos = cuda.grid(1)
            if pos >= MATRIX_DIM1:
                return
    
            if vector[pos] != 0:
                cuda.atomic.add(result, 1 , 1)
            cuda.syncthreads()

            return

    def slicer(vector):
        ## Omer Sella: slicer puts a threshold, everything above 0 is translated to 1,  otherwise 0 (including equality). Do not confuse with the reserved function name slice !
        slicedVector = np.ones(MATRIX_DIM1, dtype = LDPC_CUDA_INT_DATA_TYPE)
        slicedVector[np.where(vector <= 0)] = 0
        return slicedVector


    def modulate(vector, length):
        modulatedVector = np.ones(length, dtype = np.float32)
        modulatedVector[np.where(vector == 0)] = -1
        return modulatedVector

    def addAWGN(vector, length, SNRdb, prng):
        ## The input SNR is in db so first convert:
        SNR = 10 ** (SNRdb/10)
        ## Now use the definition: SNR = signal^2 / sigma^2
        sigma = np.sqrt(0.5 / SNR)
        noise = np.float32(prng.normal(0, sigma, length))
        sigmaActual = np.sqrt((np.sum(noise ** 2)) / length)
        noisyVector = vector + noise
        return noisyVector, sigma, sigmaActual

    def AWGNarray(dim0, dim1, SNRdb, prng):
        ## The input SNR is in db so first convert:
        SNR = 10 ** (SNRdb/10)
        ## Now use the definition: SNR = signal^2 / sigma^2
        sigma = np.sqrt(0.5 / SNR)
        noise = np.float32(prng.normal(0, sigma, (dim0, dim1)))
        sigmaActual = np.sum(noise ** 2, axis = 1) / dim1
        sigmaActual = np.sqrt( sigmaActual )
        return noise, sigma, sigmaActual
    # Concurrent futures require the seed to be between 0 and 2**32 -1
    #assert (np.dtype(seed) == np.int32)
    if seed < 0:
	    raise ValueError("Seed must be greater than 0")
    assert hasattr(SNRpoints, "__len__")
    with cuda.defer_cleanup():
        
        localPrng = np.random.RandomState(seed)
        numberOfSNRpoints = len(SNRpoints)
        softVector_host = np.zeros(MATRIX_DIM1, dtype = np.float32)
        isCodewordVector_host = np.ones(MATRIX_DIM0, dtype = np.int32)
        binaryVector_host = np.ones(MATRIX_DIM1, dtype = np.int32)
        numberOfNegatives_host = np.zeros(MATRIX_DIM0, dtype = np.int32)
        productOfSigns_host = np.zeros(MATRIX_DIM0, dtype = np.int32)
        locationOfMinimum_host = np.zeros(MATRIX_DIM0, dtype = np.int32)
        smallest_host = np.zeros(MATRIX_DIM0, dtype = np.float32)
        secondSmallest_host = np.zeros(MATRIX_DIM0, dtype = np.float32)
        newMatrix_host = np.zeros((MATRIX_DIM0,MATRIX_DIM1), dtype = np.float32)
        matrix_host = np.zeros((MATRIX_DIM0, MATRIX_DIM1), dtype = np.float32)
        result_host = np.zeros(120, dtype = np.float32)
        temp_host = np.zeros(2, dtype = np.float)
        
        isCodewordVector_device = cuda.to_device(isCodewordVector_host)
        binaryVector_device = cuda.to_device(binaryVector_host)
        numberOfNegatives_device = cuda.to_device(numberOfNegatives_host)
        productOfSigns_device = cuda.to_device(productOfSigns_host)
        locationOfMinimum_device = cuda.to_device(locationOfMinimum_host)
        smallest_device = cuda.to_device(smallest_host)
        secondSmallest_device = cuda.to_device(secondSmallest_host)
        newMatrix_device = cuda.to_device(newMatrix_host)
        parityMatrix_device = cuda.to_device(parityMatrix)
        matrix_device = cuda.to_device(matrix_host)
        result_device = cuda.to_device(result_host)
        
        zro_device = cuda.to_device(np.zeros(1))
        # init a new berStatistics object to collect statistics
        berStats = common.berStatistics()#np.zeros(numberOfSNRpoints, dtype = LDPC_DECIMAL_DATA_TYPE)
        codeword = np.zeros(MATRIX_DIM1, dtype = np.int32)
        modulatedCodeword = modulate(codeword, MATRIX_DIM1)   
        
        numberOfSNRpoints_device = cuda.to_device(numberOfSNRpoints)
        numOfTransmissions_device = cuda.to_device(numOfTransmissions)

        for s in range(numberOfSNRpoints):
            start = 0
            end = 0
            totalTime = 0
            for t in range(numOfTransmissions):
                fromChannel_host, sigma, sigmaActual = addAWGN(modulatedCodeword, MATRIX_DIM1, SNRpoints[s], localPrng) 
                softVector_host = copy.copy(fromChannel_host)
                fromChannel_device = cuda.to_device(fromChannel_host)    
                softVector_device = cuda.to_device(softVector_host)    
                senseword = slicer(fromChannel_host)
                berUncoded = np.count_nonzero(senseword != codeword)
            
                ########################### Decoding happens here #######################
                #start = time.time()
                iterator = 0
                isCodeword = False
                maskedFanOut[BLOCKS_PER_GRID_DIM1, THREADS_PER_BLOCK](parityMatrix_device, softVector_device, matrix_device)
                # Check if fromChannel makes a codeword    
                slicerCuda[BLOCKS_PER_GRID_DIM1, THREADS_PER_BLOCK](softVector_device, binaryVector_device)
                resetVector[1022, 1](isCodewordVector_device)
                calcBinaryProduct2[BLOCKS_PER_GRID_BINARY_CALC, THREADS_PER_BLOCK_BINARY_CALC](parityMatrix_device, binaryVector_device, isCodewordVector_device)
                mod2Vector[16, 511](isCodewordVector_device)
                checkIsCodeword[BLOCKS_PER_GRID_DIM0, THREADS_PER_BLOCK](isCodewordVector_device, result_device)
                if result_device[0] == 0 :
                    isCodeword = True
                    
                while (iterator < numberOfIterations and not isCodeword):
                    findMinimaAndNumberOfNegatives[BLOCKS_PER_GRID_DIM0, THREADS_PER_BLOCK](parityMatrix_device, matrix_device, smallest_device, secondSmallest_device, locationOfMinimum_device)
                    numberOfNegativesToProductOfSigns[BLOCKS_PER_GRID_DIM0, THREADS_PER_BLOCK](numberOfNegatives_device,productOfSigns_device)  
                    locateTwoSmallestHorizontal2DV2[BLOCKS_PER_GRID_LOCATE_TWO_SMALLEST_HORIZONTAL_2D, THREADS_PER_BLOCK_LOCATE_TWO_SMALLEST_HORIZONTAL_2D](matrix_device, parityMatrix_device, smallest_device, secondSmallest_device, locationOfMinimum_device)
                    signReduceHorizontal[1022, 1](matrix_device, productOfSigns_device)  
                    produceNewMatrix2D[BLOCKS_PER_GRID_PRODUCE_NEW_MATRIX_2D, THREADS_PER_BLOCK_PRODUCE_NEW_MATRIX_2D](parityMatrix_device, matrix_device, smallest_device, secondSmallest_device, locationOfMinimum_device, productOfSigns_device, newMatrix_device)
                    matrixSumVertical[BLOCKS_PER_GRID_DIM1, THREADS_PER_BLOCK](newMatrix_device, softVector_device)
                    #sumReduction2DVertical[BLOCKS_PER_GRID_VERTICAL_SUM, THREADS_PER_BLOCK_VERTICAL_SUM](newMatrix_device, softVector_device)
                    # Omer Sella: Notice that the result of matrix summation using the cuda 
                    # kernel does not have to be exactly as the numpy sum, 
                    # even if the data types are identical. That's why the following two lines are commented:
                    #testSoftVector = np.sum(newMatrix, axis = 0)
                    #assert ( np.all(testSoftVector == softVector))
                    cudaPlusDim1[BLOCKS_PER_GRID_DIM1, THREADS_PER_BLOCK](softVector_device,fromChannel_device)
                    slicerCuda[BLOCKS_PER_GRID_DIM1, THREADS_PER_BLOCK](softVector_device, binaryVector_device)
                    resetVector[1022, 1](isCodewordVector_device)
                    calcBinaryProduct2[BLOCKS_PER_GRID_BINARY_CALC, THREADS_PER_BLOCK_BINARY_CALC](parityMatrix_device, binaryVector_device, isCodewordVector_device)
                    mod2Vector[16, 511](isCodewordVector_device)
                    checkIsCodeword[BLOCKS_PER_GRID_DIM0, THREADS_PER_BLOCK](isCodewordVector_device, result_device)
                    if iterator % 6 == 0:
                        if result_device[0] == 0:
                            isCodeword = True
                    maskedFanOut[BLOCKS_PER_GRID_DIM1, THREADS_PER_BLOCK](parityMatrix_device, softVector_device, matrix_device)
                    cudaMatrixMinus2D[BLOCKS_PER_GRID_MATRIX_MINUS_2D, THREADS_PER_BLOCK_MATRIX_MINUS_2D](matrix_device, newMatrix_device) # = matrix_device - newMatrix_device
                    iterator = iterator + 1
                slicerCuda[BLOCKS_PER_GRID_DIM1, THREADS_PER_BLOCK](softVector_device, binaryVector_device)
                result_device[1] = 0
                numberOfNonZeros[BLOCKS_PER_GRID_DIM1, THREADS_PER_BLOCK](binaryVector_device, result_device)
                #end = time.time()
                #totalTime += (end - start)
                #binaryVector_host = binaryVector_device.copy_to_host()                
                #softVector_host = softVector_device.copy_to_host()
                #########################################################################
                result_host = result_device.copy_to_host()
                berDecoded = result_host[1]
                berStats.addEntry(SNRpoints[s], sigma, sigmaActual, berUncoded, berDecoded, iterator, numberOfIterations, 'test')
                
    #Omer Sella: added cuda.close() to see if I can run with concurrent futures.
    #cuda.close()
    return berStats



def evaluateMatrixAndEpsilon(parityMatrix, epsilon, numberOfIterations = 50, cudaDeviceNumber = 0):
    cuda.select_device(cudaDeviceNumber)
    with cuda.defer_cleanup():
        ## Create assests
        softVector_host = np.zeros(MATRIX_DIM1, dtype = np.float32)
        isCodewordVector_host = np.ones(MATRIX_DIM0, dtype = np.int32)
        binaryVector_host = np.ones(MATRIX_DIM1, dtype = np.int32)
        numberOfNegatives_host = np.zeros(MATRIX_DIM0, dtype = np.int32)
        productOfSigns_host = np.zeros(MATRIX_DIM0, dtype = np.int32)
        locationOfMinimum_host = np.zeros(MATRIX_DIM0, dtype = np.int32)
        smallest_host = np.zeros(MATRIX_DIM0, dtype = np.float32)
        secondSmallest_host = np.zeros(MATRIX_DIM0, dtype = np.float32)
        newMatrix_host = np.zeros((MATRIX_DIM0,MATRIX_DIM1), dtype = np.float32)
        matrix_host = np.zeros((MATRIX_DIM0, MATRIX_DIM1), dtype = np.float32)
        result_host = np.zeros(120, dtype = np.float32)
        temp_host = np.zeros(2, dtype = np.float)
        # Move assests to device explicitly
        isCodewordVector_device = cuda.to_device(isCodewordVector_host)
        binaryVector_device = cuda.to_device(binaryVector_host)
        numberOfNegatives_device = cuda.to_device(numberOfNegatives_host)
        productOfSigns_device = cuda.to_device(productOfSigns_host)
        locationOfMinimum_device = cuda.to_device(locationOfMinimum_host)
        smallest_device = cuda.to_device(smallest_host)
        secondSmallest_device = cuda.to_device(secondSmallest_host)
        newMatrix_device = cuda.to_device(newMatrix_host)
        parityMatrix_device = cuda.to_device(parityMatrix)
        matrix_device = cuda.to_device(matrix_host)
        result_device = cuda.to_device(result_host)
        zro_device = cuda.to_device(np.zeros(1))
        codeword = np.zeros(MATRIX_DIM1, dtype = np.int32)
        modulatedCodeword = modulate(codeword, MATRIX_DIM1)   
        #####
        fromChannel_host = modulatedCodeword + epsilon
        softVector_host = copy.copy(fromChannel_host)
        fromChannel_device = cuda.to_device(fromChannel_host)    
        softVector_device = cuda.to_device(softVector_host)
        senseword = slicer(fromChannel_host)
        berUncoded = np.count_nonzero(senseword != codeword)
        berDecoded = MATRIX_DIM1
            
        ########################### Decoding happens here #######################
        totalTime = 0
        start = time.time()
        iterator = 0
        isCodeword = False
        maskedFanOut[BLOCKS_PER_GRID_DIM1, THREADS_PER_BLOCK](parityMatrix_device, softVector_device, matrix_device)
        # Check if fromChannel makes a codeword    
        slicerCuda[BLOCKS_PER_GRID_DIM1, THREADS_PER_BLOCK](softVector_device, binaryVector_device)
        resetVector[1022, 1](isCodewordVector_device)
        calcBinaryProduct2[BLOCKS_PER_GRID_BINARY_CALC, THREADS_PER_BLOCK_BINARY_CALC](parityMatrix_device, binaryVector_device, isCodewordVector_device)
        mod2Vector[16, 511](isCodewordVector_device)
        checkIsCodeword[BLOCKS_PER_GRID_DIM0, THREADS_PER_BLOCK](isCodewordVector_device, result_device)
        if result_device[0] == 0 :
            isCodeword = True
            berDecoded = 0   
        while (iterator < numberOfIterations and not isCodeword):
            findMinimaAndNumberOfNegatives[BLOCKS_PER_GRID_DIM0, THREADS_PER_BLOCK](parityMatrix_device, matrix_device, smallest_device, secondSmallest_device, locationOfMinimum_device)
            numberOfNegativesToProductOfSigns[BLOCKS_PER_GRID_DIM0, THREADS_PER_BLOCK](numberOfNegatives_device,productOfSigns_device)  
            locateTwoSmallestHorizontal2DV2[BLOCKS_PER_GRID_LOCATE_TWO_SMALLEST_HORIZONTAL_2D, THREADS_PER_BLOCK_LOCATE_TWO_SMALLEST_HORIZONTAL_2D](matrix_device, parityMatrix_device, smallest_device, secondSmallest_device, locationOfMinimum_device)
            signReduceHorizontal[1022, 1](matrix_device, productOfSigns_device)  
            produceNewMatrix2D[BLOCKS_PER_GRID_PRODUCE_NEW_MATRIX_2D, THREADS_PER_BLOCK_PRODUCE_NEW_MATRIX_2D](parityMatrix_device, matrix_device, smallest_device, secondSmallest_device, locationOfMinimum_device, productOfSigns_device, newMatrix_device)
            matrixSumVertical[BLOCKS_PER_GRID_DIM1, THREADS_PER_BLOCK](newMatrix_device, softVector_device)
            #sumReduction2DVertical[BLOCKS_PER_GRID_VERTICAL_SUM, THREADS_PER_BLOCK_VERTICAL_SUM](newMatrix_device, softVector_device)
            # Omer Sella: Notice that the result of matrix summation using the cuda 
            # kernel does not have to be exactly as the numpy sum, 
            # even if the data types are identical. That's why the following two lines are commented:
            cudaPlusDim1[BLOCKS_PER_GRID_DIM1, THREADS_PER_BLOCK](softVector_device,fromChannel_device)
            slicerCuda[BLOCKS_PER_GRID_DIM1, THREADS_PER_BLOCK](softVector_device, binaryVector_device)
            resetVector[1022, 1](isCodewordVector_device)
            calcBinaryProduct2[BLOCKS_PER_GRID_BINARY_CALC, THREADS_PER_BLOCK_BINARY_CALC](parityMatrix_device, binaryVector_device, isCodewordVector_device)
            mod2Vector[16, 511](isCodewordVector_device)
            checkIsCodeword[BLOCKS_PER_GRID_DIM0, THREADS_PER_BLOCK](isCodewordVector_device, result_device)
            # The choice of checking every 6 iteration is dependent on cuda version, GPU and host, and requires ad-hoc fine tuning 
            if iterator % 6 == 0:
                if result_device[0] == 0:
                    isCodeword = True
            maskedFanOut[BLOCKS_PER_GRID_DIM1, THREADS_PER_BLOCK](parityMatrix_device, softVector_device, matrix_device)
            cudaMatrixMinus2D[BLOCKS_PER_GRID_MATRIX_MINUS_2D, THREADS_PER_BLOCK_MATRIX_MINUS_2D](matrix_device, newMatrix_device) # = matrix_device - newMatrix_device
            iterator = iterator + 1
        slicerCuda[BLOCKS_PER_GRID_DIM1, THREADS_PER_BLOCK](softVector_device, binaryVector_device)
        result_device[1] = 0
        numberOfNonZeros[BLOCKS_PER_GRID_DIM1, THREADS_PER_BLOCK](binaryVector_device, result_device)
        end = time.time()
        result_host = result_device.copy_to_host()
        totalTime += (end - start)
        berDecoded = result_host[1]
    return berUncoded, berDecoded, iterator, totalTime



def testNearEarth(numOfTransmissions = 60, graphics = True):
    status = 'Near earth problem'
    print("*** in test near earth")
    nearEarthParity = np.int32(fileHandler.readMatrixFromFile(str(projectDir) + '/codeMatrices/nearEarthParity.txt', 1022, 8176, 511, True, False, False))
    roi = [3.0, 3.2 ,3.4, 3.6]#,3.6, 3.8]#[28, 29, 30, 31]##np.arange(3, 3.8, 0.2)
    codewordSize = 8176
    messageSize = 7154
    numOfIterations = 50

    start = time.time()
    bStats = evaluateCodeCuda(460101, roi, numOfIterations, nearEarthParity, numOfTransmissions)    
    end = time.time()
    print('Time it took for code evaluation == %d' % (end-start))
    print('Throughput == '+str((8176*len(roi)*numOfTransmissions)/(end-start)) + 'bits per second.')
    a, b, c, d = bStats.getStats(codewordSize)
    scatterSnr, scatterBer, scatterItr, snrAxis, averageSnrAxis, berData, averageNumberOfIterations = bStats.getStatsV2()
    pConst = np.poly1d([1])
    
    p = np.polyfit(scatterSnr, scatterBer, 1)
    # Omer Sella: 16/06/2021 decided to use np polynomials. Also changed the reward to the area between
    # the constant 1 and the fitted line.
    p1 = np.poly1d(p)
    pTotalInteg = (pConst - p1).integ()
    reward = pTotalInteg(roi[-1]) - pTotalInteg(roi[0])
    print(reward)
    if graphics == True:
        common.plotEvaluationData(scatterSnr, scatterBer)

    print("berDecoded " + str(c))
    
    if (c[-1] == 0) and (c[-2] == 0):
        status = 'OK'
    return bStats, status


def testConcurrentFutures(numberOfCudaDevices = 1):
    nearEarthParity = np.int32(fileHandler.readMatrixFromFile(str(projectDir) + '/codeMatrices/nearEarthParity.txt', 1022, 8176, 511, True, False, False))
    numOfTransmissions = 15
    roi = [3.0, 3.2 ,3.4, 3.6]
    numOfIterations = 50
    seeds = LDPC_LOCAL_PRNG.randint(0, LDPC_MAX_SEED, numberOfCudaDevices, dtype = LDPC_SEED_DATA_TYPE) 
    berStats = common.berStatistics()
    #################
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = {executor.submit(evaluateCodeCuda, seeds[deviceNumber], roi, numOfIterations, nearEarthParity, numOfTransmissions, 'None', deviceNumber): deviceNumber for deviceNumber in range(numberOfCudaDevices)}
        for result in concurrent.futures.as_completed(results):
            berStats = berStats.add(result.result())
    ##################
    return berStats


def evaluateCodeCudaWrapper(seeds, SNRpoints, numberOfIterations, parityMatrix, numOfTransmissions, G = 'None' , numberOfCudaDevices = 4):
    # This is a multiprocessing wrapper for evaluateCodeCuda.
    # No safety of len(seeds) == numberOfCudaDevices
    # No safety of cuda devices exist
    # Number of iterations must be divisible by numberOfCudaDevices
    berStats = common.berStatistics()
    newNumOfTransmissions = numOfTransmissions // numberOfCudaDevices
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = {executor.submit(evaluateCodeCuda, seeds[deviceNumber], SNRpoints, numberOfIterations, parityMatrix, newNumOfTransmissions, 'None', deviceNumber): deviceNumber for deviceNumber in range(numberOfCudaDevices)}
    for result in concurrent.futures.as_completed(results):
        berStats = berStats.add(result.result())
    return berStats
            

        
def main():
    print("*** In ldpcCUDA.py main function.")  
    #bStats, status = testNearEarth()
    start = time.time()
    bStats = testConcurrentFutures(numberOfCudaDevices = 1)
    end =  time.time()
    print("*** total running time == " + str(end - start))
    print(bStats.getStats())
    status = 0
    #testAsyncExec()
    return bStats, status


if __name__ == '__main__':
    main()
