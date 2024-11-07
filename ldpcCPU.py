# The min sum algorithm steps are defined as follows:
# Check node:
# eta[i,j] = min_{k \neq j}|lamda[i] - eta^{previous}[i,k]|\cdot \product_{k \neq j}sign(lamda[i] - eta^{previous}[i,k])
# Bit node:
# lamda[j] = r[j] + \sum_{c_i \in M_j} (eta[i,j])
# Code min-sum decoder, test it for correctness and profile it.
#import numpy as np
import os, sys
ldpcProjectDir = os.environ.get('LDPC')
sys.path.insert(1, ldpcProjectDir)
from concurrent.futures import ProcessPoolExecutor, ALL_COMPLETED

import concurrent
#filehandler is a file from the ldpc project
import fileHandler 
import numpy as np

class bitNode():
    """
    A bit node is a user of eta and lamda, and produces:
    new value for lamda
    """
    def __init__(self, bitNodeNumber): #chekNodes, received, checkNodeNumbers, checkNodes, bitNodeNumber):
        self.checkNodes = {}#{b : cn for (b,cn) in zip(checkNodeNumbers, checkNodes)}
        self.lamda = 1#{c:0 for c in checkNodes}
        self.eta = {}#{c:0 for c in checkNodes}
        self.received = 1#{b : r for (b,r) in zip(bitNumbers, received)}
        #self.received = received
        self.bitNodeNumber = bitNodeNumber
        return
    
    def addCheckNode(self, checkNodeNumber, checkNode):
        self.checkNodes[checkNodeNumber] = checkNode
        #self.lamda[checkNodeNumber] = 0 #{c:0 for c in checkNodes}
        self.eta[checkNodeNumber] = 0 #{c:0 for c in checkNodes}
        return
    
    def setReceived(self, received):
        self.received = received
        self.lamda = received
        return

    def pushLamda(self):
        # Not used !
        for k in list(self.checkNodes.keys()):
            self.checkNodes[k].lamda[self.bitNodeNumber] = self.lamda
        return
    
    def step(self):
        self.lamda = self.received + sum(self.eta.values())
        return
    
    def slice(self):
        if self.lamda > 0:
            return 1
        else:
            return 0
        
class checkNode(): 
    """
    Each check node is a consumer of eta, lamda, and produces:
    A new value for eta
    """
    def __init__(self, checkNumber):
        self.checkNodeNumber = checkNumber
        self.eta = {}#{b: 0 for b in bitNumbers}
        self.etaNew = {} #{b: 0 for b in bitNumbers}
        self.bitNodes = {}
        return
    
    def addBitNode(self, bitNodeNumber, bitNode):
        self.eta[bitNodeNumber] = 0#{b: 0 for b in bitNodes}
        self.etaNew[bitNodeNumber] = 0# {b: 0 for b in bitNodes}
        self.bitNodes[bitNodeNumber] = bitNode
        return
    
    def pushEta(self):
        for k in list(self.bitNodes.keys()):
            self.bitNodes[k].eta[self.checkNodeNumber] = self.etaNew[k]
        return
        
    def step(self):
        for key in self.bitNodes.keys():
            signs = np.where(np.array([self.bitNodes[otherKey].lamda - self.eta[otherKey] for otherKey in list(self.bitNodes.keys()) if otherKey != key]) > 0, 1, -1)
            #signs = np.where(np.array([self.bitNodes[otherKey].lamda for otherKey in list(self.bitNodes.keys()) if otherKey != key]) > 0, 1, -1)
            extendedSignProduct = np.prod( signs )
            self.etaNew[key] = min([ np.abs(self.bitNodes[otherKey].lamda - self.eta[otherKey]) 
                                  for otherKey in list(self.bitNodes.keys()) if otherKey != key]) * extendedSignProduct
            #self.etaNew[key] = min([ np.abs(self.bitNodes[otherKey].lamda) 
            #                      for otherKey in list(self.bitNodes.keys()) if otherKey != key]) * extendedSignProduct
        # Update eta to newEta
        for b in self.bitNodes.keys():
            self.eta[b] = self.etaNew[b]
        # Communicate new eta values to the bit nodes
        self.pushEta()
        return

def tannerGraphGenerator(parityMatrix):
    bitNodesList = [bitNode(i) for i in range(parityMatrix.shape[0])]
    checkNodesList = [checkNode(j) for j in range(parityMatrix.shape[1])]
    for i in range(parityMatrix.shape[0]):
        for j in range(parityMatrix.shape[0]):
            if parityMatrix[i,j] == 1:
                bitNodesList[j].addCheckNode(i, checkNodesList[i])
                checkNodesList[i].addBitNode(j, bitNodesList[j])
    return bitNodesList, checkNodesList
    
def decode(bitNodeList, checkNodeList, numberOfIterations, parityMatrix, checkEvery):
    """
    Arguments:

        parityMatrix: a binary parity matrix (0s and 1s) used to check if a codeword is obtained
        checkEvery: an integer to determine how often to check for codeword convergence
    
    Returns:
        None. Information is stored in the bit nodes

    """
    isCodeword = False
    for i in range(1, numberOfIterations + 1,1):
        sliced = np.array([b.slice() for b in bitNodeList])
        print(f"On iteration {i} the message is: {sliced}")
        print(i)
        # Check node calculation using step
        with ProcessPoolExecutor() as executor:
            results = [executor.submit(c.step()) for c in checkNodeList]
        # Wait for all check nodes to finish their update
        concurrent.futures.wait(results, timeout = None, return_when = ALL_COMPLETED)
        # Bit node calculation using step
        with ProcessPoolExecutor() as executor:
            results = [executor.submit(b.step()) for b in bitNodeList]
        # Wait for all bit nodes to finish their update
        concurrent.futures.wait(results, timeout = None, return_when = ALL_COMPLETED)
        
        if i % checkEvery == 0:
            sliced = np.array([b.slice() for b in bitNodeList])
            isCodeword = np.all(parityMatrix.T.dot(sliced) % 2 == 0)
        if isCodeword:
            print(f"Codeword found !!!")
            break
    # note that this function doesn't return a codeword or a vector, the information is in the bit nodes
    return
        
def exampleNearEarth():
    # Load parity matrix
    H = fileHandler.readMatrixFromFile(str(ldpcProjectDir) + '/codeMatrices/nearEarthParity.txt', 1022, 8176, 511, True, False, False)
    # Generate tanner graph
    bitNodesList, checkNodesList = tannerGraphGenerator(H)
    #
    length = H.shape[1]
    SNRdb = 2.5
    ## Now use the definition: SNR = signal^2 / sigma^2
    sigma = np.sqrt(0.5 / (10 ** (SNRdb/10)))
    noise = np.float32(np.random.normal(0, sigma, length))
    received = noise
    
    for i,b in zip(range(length), bitNodesList):
        b.setReceived(received[i])
    print(received)
    #print(np.where(received > 0, 1, 0))   
    a = checkNodesList[0]
   
    #a.step()
    decode(bitNodesList, checkNodesList, 10, H, 1)
    #b0 = bitNodesList[0]
    #b176 = bitNodesList[176]
    #b523 = bitNodesList[523]
    #b0.step()
    #b176.step()
    #b523.step()
    sliced = np.array([b.slice() for b in bitNodesList])
    print(sliced)


if __name__ == "__main__":
    exampleNearEarth()