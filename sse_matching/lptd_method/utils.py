import numpy as np
from scipy.linalg import pinv, qr
from scipy.optimize import linear_sum_assignment
try:
    import pulp
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False
    print("Warning: PuLP not available. Using Hungarian algorithm instead of GLPK.")

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
    # In MATLAB: sum along dimension 1 (columns), transpose to column, multiply by scalar
    # In Python: sum(ri**2, axis=0) gives (ry,) array, reshape to (ry, 1) and multiply
    d = (np.sum(ri**2, axis=0) * (rx - 1)).reshape(-1, 1)
    
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
    def matlab_round(arr):
        return np.where(arr >= 0, np.floor(arr + 0.5), np.ceil(arr - 0.5)).astype(int)

    if precision == 0:
        p1 = matlab_round(p1)
        p2 = matlab_round(p2)
        factor = 1
    else:
        factor = 10**precision
        p1 = matlab_round(p1 * factor)
        p2 = matlab_round(p2 * factor)
        
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

def lp_code(weight_matrix, use_glpk=False):
    """
    Solves the assignment problem using GLPK LP solver (like MATLAB) or Hungarian algorithm.
    
    This function replicates MATLAB's Lp_code.m which uses glpk to solve the assignment
    problem as a full linear program with inequality and equality constraints.
    
    Args:
        weight_matrix: (N1, N2) matrix of costs/weights
        use_glpk: If True and PuLP is available, use GLPK. Otherwise use Hungarian algorithm.
        
    Returns:
        permvec: (N1,) array where permvec[i] is the column index assigned to row i (0-indexed)
        fval: Total cost of assignment
    """
    n1, n2 = weight_matrix.shape
    
    if use_glpk and PULP_AVAILABLE:
        # Use GLPK through PuLP to match MATLAB exactly
        if n1 <= n2:
            return _lp_code_glpk_pulp(weight_matrix)
        else:
            # Transpose case: solve for transposed, then remap
            permvec_t, fval = _lp_code_glpk_pulp(weight_matrix.T)
            # Remap: permvec_t[j] tells which row of original was assigned to column j
            # We want: permvec[i] tells which column was assigned to row i
            permvec = np.full(n1, -1, dtype=int)  # Initialize with -1 for unassigned
            for j in range(n2):
                i = permvec_t[j]
                permvec[i] = j
            return permvec, fval
    else:
        # Fallback to Hungarian algorithm
        return _lp_code_hungarian(weight_matrix)


def _lp_code_glpk_pulp(D):
    """
    Solve assignment problem using GLPK via PuLP (matches MATLAB's injmatch function).
    
    Args:
        D: (N1, N2) cost matrix
        
    Returns:
        permvec: (N1,) array, permvec[i] = column assigned to row i (0-indexed)
        fval: Total cost
    """
    n1, n2 = D.shape
    
    # Handle transpose case: if more rows than columns, transpose and solve
    if n1 > n2:
        # Solve transposed problem
        D_T = D.T
        n1_T, n2_T = D_T.shape  # n1_T = n2, n2_T = n1
        
        # Create LP problem for transposed matrix
        prob = pulp.LpProblem("Assignment_Transposed", pulp.LpMinimize)
        
        # Decision variables: x[i,j] = 1 if column i assigned to row j (in transposed space)
        x = {}
        for i in range(n1_T):
            for j in range(n2_T):
                x[i, j] = pulp.LpVariable(f"x_{i}_{j}", cat='Binary')
        
        # Objective: minimize sum of costs
        prob += pulp.lpSum([D_T[i, j] * x[i, j] for i in range(n1_T) for j in range(n2_T)])
        
        # Equality constraints: each row (column in original) must be assigned to exactly one column (row in original)
        for i in range(n1_T):
            prob += pulp.lpSum([x[i, j] for j in range(n2_T)]) == 1
        
        # Inequality constraints: each column (row in original) can be assigned to at most one row (column in original)
        for j in range(n2_T):
            prob += pulp.lpSum([x[i, j] for i in range(n1_T)]) <= 1
        
        # Solve with GLPK
        solver = pulp.GLPK_CMD(msg=False)
        prob.solve(solver)
        
        # Extract solution: map back to original space
        permvec = np.full(n1, -1, dtype=int)  # n1 is original number of rows
        for i in range(n1_T):  # For each column in original (row in transposed)
            for j in range(n2_T):  # For each row in original (column in transposed)
                if pulp.value(x[i, j]) > 0.5:
                    permvec[j] = i  # Original row j assigned to original column i
                    break
        
        fval = pulp.value(prob.objective)
        return permvec, fval
        fval = pulp.value(prob.objective)
        return permvec, fval
    
    # Standard case: N1 <= N2
    # Create LP problem
    prob = pulp.LpProblem("Assignment", pulp.LpMinimize)
    
    # Decision variables: x[i,j] = 1 if row i assigned to column j
    x = {}
    for i in range(n1):
        for j in range(n2):
            x[i, j] = pulp.LpVariable(f"x_{i}_{j}", cat='Binary')
    
    # Objective: minimize sum of costs
    prob += pulp.lpSum([D[i, j] * x[i, j] for i in range(n1) for j in range(n2)])
    
    # Equality constraints: each row must be assigned to exactly one column
    for i in range(n1):
        prob += pulp.lpSum([x[i, j] for j in range(n2)]) == 1
    
    # Inequality constraints: each column can be assigned to at most one row
    for j in range(n2):
        prob += pulp.lpSum([x[i, j] for i in range(n1)]) <= 1
    
    # Solve with GLPK
    solver = pulp.GLPK_CMD(msg=False)
    prob.solve(solver)
    
    # Extract solution
    permvec = np.zeros(n1, dtype=int)
    for i in range(n1):
        for j in range(n2):
            if pulp.value(x[i, j]) > 0.5:
                permvec[i] = j
                break
    
    fval = pulp.value(prob.objective)
    
    return permvec, fval


def _lp_code_hungarian(weight_matrix):
    """
    Fallback: Solve using Hungarian algorithm (scipy).
    """
    n1, n2 = weight_matrix.shape
    
    if n1 <= n2:
        row_ind, col_ind = linear_sum_assignment(weight_matrix)
        assignment = sorted(zip(row_ind, col_ind), key=lambda x: x[0])
        permvec = np.array([x[1] for x in assignment])
        fval = weight_matrix[row_ind, col_ind].sum()
        return permvec, fval
    else:
        # Transpose case
        row_ind, col_ind = linear_sum_assignment(weight_matrix.T)
        permvec = np.full(n1, -1, dtype=int)
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
