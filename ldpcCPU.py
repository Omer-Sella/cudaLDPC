# The min sum algorithm steps are defined as follows:
# Check node:
# eta[i,j] = min_{k \neq j}|lamda[i] - eta^{previous}[i,k]|\cdot \product_{k \neq j}sign(lamda[i] - eta^{previous}[i,k])
# Bit node:
# lamda[j] = r[j] + \sum_{c_i \in M_j} (eta[i,j])
# Code min-sum decoder, test it for correctness and profile it.
import numpy as np
class bitNode():
    """
    A bit node is a user of eta, lamda, and received, and produces:
    new value for lamda
    """
    def __init__(self, chekNodes, received, checkNodeNumbers, checkNodes, bitNodeNumber):
        self.checkNodes = {b : cn for (b,cn) in zip(checkNodeNumbers, checkNodes)}
        self.lamda = {c:0 for c in checkNodes}
        self.eta = {c:0 for c in checkNodes}
        #self.received = {b : r for (b,r) in zip(bitNumbers, received)}
        self.received = received
        return self
    
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
        self.bitNodes = []
        return self
    def setBitNodes(self, bitNodes):
        self.eta = {}{b: 0 for b in bitNodes}
        self.etaNew = {} {b: 0 for b in bitNodes}
        self.bitNodes = bitNodes
        return
    
    def pushEta(self):
        for b in self.bitNodes:
            b.eta[self.checkNodeNumber] = self.eta[b]
        return
        
    def step(self):
        for b in self.bits:
            extendedSign = np.prod(np.where([self.lamda[bn] - self.eta[bn]] for bn in self.bits if bn != b] > 0, 1, -1) 
            self.etaNew[c] = min([ np.abs(self.eta[bn] - self.lamda[bn]) 
                                  for bn in self.bits if bn != b]) * extendedSign
        [self.eta[b] = self.etaNew[b] for c in self.bitNumbers]
        self.pushEta()
        return
    

            
        

