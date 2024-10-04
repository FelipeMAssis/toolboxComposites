import numpy as np

def tensor2vec(A):
    if len(A) == 2:
        Avec = np.zeros([3,1])
        Avec[0] = A[0,0]
        Avec[1] = A[1,1]
        Avec[2] = A[0,1]
        return Avec
    if len(A) == 3:
        Avec = np.zeros([6,1])
        Avec[0] = A[0,0]
        Avec[1] = A[1,1]
        Avec[2] = A[2,2]
        Avec[3] = A[1,2]
        Avec[4] = A[0,2]
        Avec[5] = A[0,1]
        return Avec

def vec2tensor(Avec):
    if Avec.shape[0] == 3:
        A = np.zeros([2,2])
        A[0,0] = Avec[0]
        A[1,1] = Avec[1]
        A[0,1] = Avec[2]
        A[1,0] = Avec[2]
        return A
    if Avec.shape[0] == 5:
        A = np.zeros([3,3])
        A[0,0] = Avec[0]
        A[1,1] = Avec[1]
        A[2,2] = Avec[2]
        A[1,2] = Avec[3]
        A[2,1] = Avec[3]
        A[0,2] = Avec[4]
        A[2,0] = Avec[4]
        A[0,1] = Avec[5]
        A[1,0] = Avec[5]
        return A
    
def rotate(theta, A):
        flag = False
        if A.shape[1]==1:
            flag = True
            A = vec2tensor(A)
        theta = np.pi*theta/180
        c = np.cos(theta)
        s = np.sin(theta)
        T = np.array(
            [[c, -s],
            [s, c]]
        )
        Ap = T.transpose() @ A @ T
        if flag:
             return tensor2vec(Ap)
        return Ap