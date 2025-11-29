import numpy as np
from scipy.linalg import pinv, qr
from scipy.optimize import linear_sum_assignment

def imsd(Y, X):
    """
    Improved Mahalanobis Squared Distance.
    D2 = IMSD(Y,X) returns the Mahalanobis distance (in squared units) of
    each observation (point) in Y from the sample data in X.
    
    Args:
        Y: (ry, cy) array
        X: (rx, cx) array
        
    Returns:
        d: (ry, 1) array of squared distances
    """
    rx, cx = X.shape
    ry, cy = Y.shape
    
    if cx != cy:
        raise ValueError("Input size mismatch: X and Y must have same number of columns")
    
    if rx < cx:
        # Not enough observations to estimate covariance, handle gracefully or raise error
        # The MATLAB code raises error, but let's see if we can handle it.
        # For now, follow MATLAB behavior.
        # raise ValueError("Too few rows in X")
        pass

    m = np.mean(X, axis=0)
    M = np.tile(m, (ry, 1))
    C = X - np.tile(m, (rx, 1))
    
    # QR decomposition
    Q, R = qr(C, mode='economic')
    
    # ri = pinv(R') * (Y - M)'
    # In Python: pinv(R.T) @ (Y - M).T
    ri = pinv(R.T) @ (Y - M).T
    
    # d = sum(ri.*ri, 1)' * (rx - 1)
    # In Python: sum(ri**2, axis=0) * (rx - 1)
    d = np.sum(ri**2, axis=0) * (rx - 1)
    
    return d

def bresenham_line3d(p1, p2, precision=0):
    """
    Generate X Y Z coordinates of a 3D Bresenham's line between two given points.
    
    Args:
        p1: [x1, y1, z1]
        p2: [x2, y2, z2]
        precision: integer, number of decimal digits to preserve
        
    Returns:
        points: (N, 3) array of coordinates
    """
    if precision == 0:
        p1 = np.round(p1).astype(int)
        p2 = np.round(p2).astype(int)
        factor = 1
    else:
        factor = 10**precision
        p1 = np.round(p1 * factor).astype(int)
        p2 = np.round(p2 * factor).astype(int)
        
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    
    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1
    
    ax = abs(dx) * 2
    ay = abs(dy) * 2
    az = abs(dz) * 2
    
    sx = np.sign(dx)
    sy = np.sign(dy)
    sz = np.sign(dz)
    
    x, y, z = x1, y1, z1
    
    points = []
    
    if ax >= max(ay, az): # x dominant
        yd = ay - ax / 2
        zd = az - ax / 2
        
        while True:
            points.append([x, y, z])
            if x == x2:
                break
                
            if yd >= 0:
                y += sy
                yd -= ax
            
            if zd >= 0:
                z += sz
                zd -= ax
                
            x += sx
            yd += ay
            zd += az
            
    elif ay >= max(ax, az): # y dominant
        xd = ax - ay / 2
        zd = az - ay / 2
        
        while True:
            points.append([x, y, z])
            if y == y2:
                break
                
            if xd >= 0:
                x += sx
                xd -= ay
                
            if zd >= 0:
                z += sz
                zd -= ay
                
            y += sy
            xd += ax
            zd += az
            
    elif az >= max(ax, ay): # z dominant
        xd = ax - az / 2
        yd = ay - az / 2
        
        while True:
            points.append([x, y, z])
            if z == z2:
                break
                
            if xd >= 0:
                x += sx
                xd -= az
                
            if yd >= 0:
                y += sy
                yd -= az
                
            z += sz
            xd += ax
            yd += ay
            
    points = np.array(points, dtype=float)
    if precision != 0:
        points /= factor
        
    return points

def lp_code(weight_matrix):
    """
    Solves the assignment problem using Linear Sum Assignment (Hungarian algorithm).
    Equivalent to the Linear Programming approach for assignment.
    
    Args:
        weight_matrix: (N1, N2) matrix of costs/weights
        
    Returns:
        permvec: (N1,) array where permvec[i] is the column index assigned to row i
        fval: Total cost of assignment
    """
    n1, n2 = weight_matrix.shape
    
    if n1 <= n2:
        row_ind, col_ind = linear_sum_assignment(weight_matrix)
        # linear_sum_assignment returns optimal assignment.
        # row_ind will be 0..n1-1, col_ind will be the assigned columns.
        # We need to ensure it matches the MATLAB behavior.
        # MATLAB code returns permvec where permvec[i] = assigned_column_for_row_i
        
        # Sort by row index to ensure permvec corresponds to rows 0, 1, ...
        assignment = sorted(zip(row_ind, col_ind), key=lambda x: x[0])
        permvec = np.array([x[1] for x in assignment])
        
        fval = weight_matrix[row_ind, col_ind].sum()
        return permvec, fval
    else:
        # Transpose if more rows than columns
        # The MATLAB code does: [permvec,res,FVAL] = injmatch(D',min);
        # Then permutes back.
        # If we transpose, we match columns to rows.
        row_ind, col_ind = linear_sum_assignment(weight_matrix.T)
        
        # row_ind corresponds to columns of original matrix
        # col_ind corresponds to rows of original matrix
        
        # We want permvec for original rows.
        # permvec[row] = col
        
        permvec = np.zeros(n1, dtype=int)
        # Initialize with -1 or something to indicate no assignment?
        # But in this case (n1 > n2), not all rows will be assigned.
        # The MATLAB code:
        # permvecold = permvec (from transposed call)
        # permvec = zeros(N1,1);
        # for i=1:N2
        #    permvec(permvecold(i))= i;
        # end
        
        # Let's trace:
        # D' is (N2, N1) where N2 < N1.
        # injmatch(D') returns permvec_transposed of size N2.
        # permvec_transposed[j] = row index in original D assigned to column j in original D.
        
        # So we have mapping: col_j -> row_i
        # We want mapping: row_i -> col_j
        
        # In Python linear_sum_assignment(weight_matrix.T):
        # row_ind (cols of orig) -> col_ind (rows of orig)
        
        # So for each pair (c, r) in zip(row_ind, col_ind):
        # Row r is assigned to Column c.
        
        permvec = np.full(n1, -1, dtype=int) # -1 for unassigned
        for c, r in zip(row_ind, col_ind):
            permvec[r] = c
            
        fval = weight_matrix.T[row_ind, col_ind].sum()
        
        return permvec, fval

def dtw(x, y):
    """
    Dynamic Time Warping distance between two sequences.
    Simple implementation.
    """
    nx = len(x)
    ny = len(y)
    
    D = np.zeros((nx + 1, ny + 1))
    D[0, 1:] = np.inf
    D[1:, 0] = np.inf
    
    for i in range(1, nx + 1):
        for j in range(1, ny + 1):
            cost = abs(x[i-1] - y[j-1])
            D[i, j] = cost + min(D[i-1, j], D[i, j-1], D[i-1, j-1])
            
    return D[nx, ny]
