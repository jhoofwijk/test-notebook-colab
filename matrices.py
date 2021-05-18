import scipy
import scipy.sparse
import numpy as np
import scipy.sparse.linalg

def poisson1d(n):
    A = scipy.sparse.spdiags([[-1] * n, [2] * n, [-1]*n], [-1,0,1], n,n).tocsr()
    
    b = np.zeros(A.shape[1])
    b[:] = 0.005
    b[0] = 1
    b[-1] = 2
    return A, b

def poisson2d(n):
    I = scipy.sparse.identity(n)
    B = scipy.sparse.spdiags([[-1] * n, [2] * n, [-1]*n], [-1,0,1], n,n).tocsr()
    A = scipy.sparse.kron(B, I) + scipy.sparse.kron(I, B) 
    
    b = np.zeros(A.shape[1])
    b[0:n] = 1
    b[-n:] = 2
    return A, b

def poisson1d_plateau(n, kbig=10):
    A = scipy.sparse.lil_matrix((n,n))

    lx = 0.5
    rx = 0.8
    for i in range(n - 1):
        x = (i + 0.5) / n 
        k = 1
        if lx < x < rx:
            k = kbig

        A[i, i] += k
        A[i+1, i+1] += k

        A[i,i+1] -= k
        A[i+1,i] -= k
    A[0,0] += 1
    A[-1,-1] += 1
    A = A.tocsr()
    
    b = np.zeros(A.shape[1])
    b[0] = 1
    b[-1] = 2
    return A, b


def poisson2d_plateau(n, kbig=10):
    A = scipy.sparse.lil_matrix((n**2,n**2))
    
    lx = 0.5
    rx = 0.8
    ly = 0.5
    ry = 0.9
    # horizontal direction
    for i in range(n - 1):
        for j in range(n):
            x = (i + 0.5) / n 
            y = j / n
            k = 1
            if lx < x < rx and ly < y < ry:
                k = kbig
            
            ind = j*n + i
            A[ind, ind] += k
            A[ind+1, ind+1] += k
            
            A[ind+1, ind] -= k
            A[ind, ind+1] -= k
    
    # vertical direction
    for i in range(n):
        for j in range(n - 1):
            x = i / n 
            y = (j + 0.5) / n
            k = 1
            if lx < x < rx and ly < y < ry:
                k = kbig
            
            ind = j*n + i
            A[ind, ind] += k
            A[ind+n, ind+n] += k
            
            A[ind+n, ind] -= k
            A[ind, ind+n] -= k
    
    # something something for the border:
    for i in range(n):
        A[i,i] = 4 # top border
        A[i*n,i*n] = 4 # left border
        A[n*(n-1) + i,n*(n-1) + i] = 4 # bottom border
        A[(n-1) + i*n,(n-1) + i*n] = 4 # right border

    A = A.tocsr()
    
    b = np.zeros(A.shape[1])
    for i in range(n):
        b[i] = 1 # top border
        # b[i*n] = 4 # left border
        b[n*(n-1) + i] = 2 # bottom border
        # b[(n-1) + i*n] = 4 # right border

    return A, b


default_plateaus = (
    (.5,.9,.5,.8),
    (.1,.3,.1,.3),
    (.1,.3,.6,.9),
)

def poisson2d_plateaus(n, kbig=100, plateaus=default_plateaus):
    A = scipy.sparse.lil_matrix((n**2,n**2))
    
    # horizontal direction
    for i in range(n - 1):
        for j in range(n):
            x = (i + 0.5) / n 
            y = j / n
            k = 1
            
            for lx,rx,ly,ry in plateaus:
                if lx < x < rx and ly < y < ry:
                    k = kbig
            
            ind = j*n + i
            A[ind, ind] += k
            A[ind+1, ind+1] += k
            
            A[ind+1, ind] -= k
            A[ind, ind+1] -= k
    
    # vertical direction
    for i in range(n):
        for j in range(n - 1):
            x = i / n 
            y = (j + 0.5) / n
            k = 1
            
            for lx,rx,ly,ry in plateaus:
                if lx < x < rx and ly < y < ry:
                    k = kbig
            
            ind = j*n + i
            A[ind, ind] += k
            A[ind+n, ind+n] += k
            
            A[ind+n, ind] -= k
            A[ind, ind+n] -= k
    
    # something something for the border:
    for i in range(n):
        A[i,i] = 4 # top border
        A[i*n,i*n] = 4 # left border
        A[n*(n-1) + i,n*(n-1) + i] = 4 # bottom border
        A[(n-1) + i*n,(n-1) + i*n] = 4 # right border

    A = A.tocsr()
    
    b = np.zeros(A.shape[1])
    for i in range(n):
        b[i] = 1 # top border
        # b[i*n] = 4 # left border
        b[n*(n-1) + i] = 2 # bottom border
        # b[(n-1) + i*n] = 4 # right border

    return A, b
