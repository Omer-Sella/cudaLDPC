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
from concurrent.futures import ProcessPoolExecutor
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
        self.lamda = 0#{c:0 for c in checkNodes}
        self.eta = {}#{c:0 for c in checkNodes}
        self._received = 0#{b : r for (b,r) in zip(bitNumbers, received)}
        #self.received = received
        self.bitNodeNumber = bitNodeNumber
        return
    
    def addCheckNode(self, checkNodeNumber, checkNode):
        self.checkNodes[checkNodeNumber] = checkNode
        #self.lamda[checkNodeNumber] = 0 #{c:0 for c in checkNodes}
        self.eta[checkNodeNumber] = 0 #{c:0 for c in checkNodes}
        return
    def setReceived(self, received):
        self.lamda = received
        return

    def pushLamda(self):
        for c in self.checkNodes:
            c.lamda[self.bitNodeNumber] = self.lamda
        return
    
    def step(self):
        sumOfEta = sum(self.eta.values())
        self.lamda = self.received + sumOfEta
        self.pushLamda()
        return
        
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
        for b in self.bitNodes:
            b.eta[self.checkNodeNumber] = self.eta[b]
        return
        
    def step(self):
        for key in list(self.bitNodes.keys()):
            signs = np.where(np.array([self.bitNodes[key].lamda - self.eta[otherKey] for otherKey in list(self.bitNodes.keys()) if otherKey != key]) > 0, 1, -1)
            extendedSign = np.prod( signs )
            self.etaNew[key] = min([ np.abs(self.eta[otherKey] - self.lamda[otherKey]) 
                                  for otherKey in list(self.bitNodes.keys()) if otherKey != key]) * extendedSign
        for b in self.bitNumbers:
            self.eta[b] = self.etaNew[b]
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
    for i in range(numberOfIterations):
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
            isCodeword = np.all(parityMatrix.dot(sliced) % 2 == 0)
        if isCodeword:
            break
    # note that this function doesn't return a codeword or a vector, the information is in the bit nodes
    return
        
def exampleNearEarth():
    # Load parity matrix
    H = fileHandler.readMatrixFromFile(str(ldpcProjectDir) + '/codeMatrices/nearEarthParity.txt', 1022, 8176, 511, True, False, False)
    # Generate tanner graph
    bitNodesList, checkNodesList = tannerGraphGenerator(H)
    #
    received = np.random.normal(0,1,H.shape[1])
    #print(H.shape)
    decode(bitNodesList, checkNodesList, 10, H, 100)


if __name__ == "__main__":
    exampleNearEarth()