import numpy as np
from sklearn.covariance import LedoitWolf

class CovarianceMatrixVolume(object):

    def __init__(self, 
                 returnsMatrix,
                 volumeMatrix):
        """Covariance Matrix Maker

        Args:
            returnsMatrix (numpy.ndarray): N x K array of asset returns
            volumeMatrix (numpy.ndarray): N x K array of asset trading volumes
        """
        self.returnsMatrix = np.array(returnsMatrix)
        self.volumeMatrix = np.array(volumeMatrix)

    def historicalCovariance(self):
        corH = np.corrcoef(self.returnsMatrix.T, ddof = 1)
        covH = np.cov(self.returnsMatrix.T, ddof = 1)
        return corH, covH
    
    def ledoitWolf(self):
        LWInst = LedoitWolf(assume_centered=False)
        LWFit = LWInst.fit(self.returnsMatrix)
        covLW = LWFit.covariance_
        D = np.linalg.inv(np.diag(np.sqrt(np.diag(covLW))))
        corLW = np.dot(D.T, np.dot(covLW, D))
        return corLW, covLW
    
    def identity(self):
        """Returns an NxN identity matrix

        Returns:
            list: list containing a nmupy.ndarray correlation matrix and a nmupy.ndarray covariance matrix
        """
        N = np.shape(self.returnsMatrix)[1]
        covIDN = np.diag(np.ones(N))
        corIDN = covIDN
        return corIDN, covIDN

    def GerberMatrix1(self, 
                      thresholdVectorReturns: np.ndarray, 
                      thresholdVectorVolume: np.ndarray,
                      method = 'quick'):
        """Gerber Matrix 1

        Args:
            thresholdVectorReturns (numpy.ndarray): K-dimensional vector of thresholds H_k for the returns Gerber statistic
            thresholdVectorVolume (numpy.ndarray): K-dimensional vector of thresholds H_k for trading volume
            method (str): of either 'quick' or 'slow'

        Returns:
            numpy.ndarray: K x K Matrix of correlation and covariance matrices (as arrays) of the first Gerber statistic
        """
        self.thresholdVectorReturns = thresholdVectorReturns
        self.thresholdVectorVolume = thresholdVectorVolume
        T, K = self.returnsMatrix.shape
        var = self.returnsMatrix.std(axis=0, ddof = 1)
        if method == 'slow':
            nUU, nUD, nDU, nDD, _ = self.upperNMatrices(T, K)
        elif method == 'quick':
            nUU, nUD, nDU, nDD, _ = self.upperNMatricesQuick(T, K)
        H = nUU + nDD - (nUD + nDU)
        h = np.sqrt(np.diag(H)).reshape(K,1)
        corG = H / np.dot(h, h.T) # correlation matrix
        covG = np.array([[corG[i,j]*var[i]*var[j] for i in range(K)] for j in range(K)]) # covariance matrix
        return corG, covG

    def GerberMatrix2(self,
                      thresholdVectorReturns: np.ndarray, 
                      thresholdVectorVolume: np.ndarray,
                      method: str = 'quick'):
        """Gerber Matrix 2

        Args:
            thresholdVectorReturns (numpy.ndarray): K-dimensional vector of thresholds H_k for the returns Gerber statistic
            thresholdVectorVolume (numpy.ndarray): K-dimensional vector of thresholds H_k for trading volume
            method (str): of either 'quick' or 'slow'

        Returns:
            numpy.ndarray: K x K Matrix of correlation and covariance matrices (as arrays) of the second Gerber statistic
        """
        self.thresholdVectorReturns = thresholdVectorReturns
        self.thresholdVectorVolume = thresholdVectorVolume
        T, K = self.returnsMatrix.shape
        var = self.returnsMatrix.std(axis=0, ddof = 1)
        if method == 'slow':
            nUU, nUD, nDU, nDD, nNN = self.upperNMatrices(T, K)
        elif method == 'quick':
            nUU, nUD, nDU, nDD, nNN = self.upperNMatricesQuick(T, K)
        H = nUU + nDD - (nUD + nDU)
        corG = H/(T - nNN) # correlation matrix
        covG = np.array([[corG[i,j]*var[i]*var[j] for i in range(K)] for j in range(K)]) # covariance matrix
        return corG, covG    
    
    def upperNMatricesQuick(self, T, K):
        """Gerber Helper function

        Args:
            T (int): number of rows (time periods)
            K (int): number of columns (assets)

        Returns:
            list: nUU, nUD, nDU, nDD and nNN matrices (list of numpy.ndarray)
        """
        U, D, N = self.matrixCountQuick(T, K)
        nUU = np.dot(U.T, U)
        nUD = np.dot(U.T, D)
        nDU = np.dot(D.T, U)
        nDD = np.dot(D.T, D)
        nNN = np.dot(N.T, N)
        return nUU, nUD, nDU, nDD, nNN
    
    def upperNMatrices(self, T, K):
        """Gerber Helper function

        Args:
            T (int): number of rows (time periods)
            K (int): number of columns (assets)

        Returns:
            list: nUU, nUD, nDU, nDD and nNN matrices (list of numpy.ndarray)
        """
        U = self.uMatrixVolume(T, K)
        D = self.dMatrixVolume(T, K)
        N = self.nMatrixVolume(T, K)
        nUU = np.dot(U.T, U)
        nUD = np.dot(U.T, D)
        nDU = np.dot(D.T, U)
        nDD = np.dot(D.T, D)
        nNN = np.dot(N.T, N)
        return nUU, nUD, nDU, nDD, nNN
    
    def matrixCountQuick(self, T, K):
        """Gerber Helper function

        Args:
            T (int): number of rows (time periods)
            K (int): number of columns (assets)

        Returns:
            list: U, D, N matrices (list of numpy.ndarray)
        """
        U = np.zeros(shape = (T,K))
        D = np.zeros(shape = (T,K))
        N = np.zeros(shape = (T,K))
        for t in range(T):
            for j in range(K):
                if (self.returnsMatrix[t,j] >= self.thresholdVectorReturns[j] and self.volumeMatrix[t,j] >= self.thresholdVectorVolume[j]):
                    U[t,j] = 1
                elif (self.returnsMatrix[t,j] <= -1*self.thresholdVectorReturns[j] and self.volumeMatrix[t,j] >= self.thresholdVectorVolume[j]):
                    D[t,j] = 1
                else:
                    N[t,j] = 1
        return U, D, N

    def uMatrixVolume(self, T, K):
        """Gerber Helper function

        Args:
            T (int): number of rows (time periods)
            K (int): number of columns (assets)

        Returns:
            numpy.ndarray: U matrix (numpy.ndarray)
        """
        U = np.zeros(shape = (T,K))
        for t in range(T):
            for j in range(K):
                U[t,j] = 1 if (self.returnsMatrix[t,j] >= self.thresholdVectorReturns[j] and self.volumeMatrix[t,j] >= self.thresholdVectorVolume[j]) else 0
        return U

    def dMatrixVolume(self, T, K):
        """Gerber Helper function

        Args:
            T (int): number of rows (time periods)
            K (int): number of columns (assets)

        Returns:
            numpy.ndarray: D matrix (numpy.ndarray)
        """
        D = np.zeros(shape = (T,K))
        for t in range(T):
            for j in range(K):
                D[t,j] = 1 if (self.returnsMatrix[t,j] <= -1*self.thresholdVectorReturns[j] and self.volumeMatrix[t,j] >= self.thresholdVectorVolume[j]) else 0
        return D

    def nMatrixVolume(self, T, K):
        """Gerber Helper function

        Args:
            T (int): number of rows (time periods)
            K (int): number of columns (assets)

        Returns:
            numpy.ndarray: N matrix (numpy.ndarray)
        """
        N = np.zeros(shape = (T,K))
        for t in range(T):
            for j in range(K):
                if not (self.returnsMatrix[t,j] >= self.thresholdVectorReturns[j] and self.volumeMatrix[t,j] >= self.thresholdVectorVolume[j]):
                    if not (self.returnsMatrix[t,j] <= -1*self.thresholdVectorReturns[j] and self.volumeMatrix[t,j] >= self.thresholdVectorVolume[j]):
                        N[t,j] = 1
        return N