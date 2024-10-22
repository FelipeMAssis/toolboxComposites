import numpy as np

def vector(A):
    """
    Convert a 2x2 or 3x3 tensor (matrix) to a vector format.
    
    Parameters:
    - A: 2D numpy array representing the tensor
    
    Returns:
    - Avec: Vector representation of the tensor
    """
    shape = A.shape
    if shape == (2, 2):
        return np.array([[A[0, 0]], [A[1, 1]], [A[0, 1]]])
    elif shape == (3, 3):
        return np.array([[A[0, 0]], [A[1, 1]], [A[2, 2]], [A[1, 2]], [A[0, 2]], [A[0, 1]]])
    elif shape == (2,1) or shape == (3,1):
        return A
    else:
        raise ValueError("Input tensor must be 2x2 or 3x3")

def tensor(Avec):
    """
    Convert a vector to a 2x2 or 3x3 tensor (matrix).
    
    Parameters:
    - Avec: Vector representing the tensor
    
    Returns:
    - A: Tensor (matrix) representation of the vector
    """
    shape = Avec.shape
    length = shape[0]
    
    if shape == (2, 2) or shape == (3,3):
        return Avec
    elif length == 3:  # Convert to 2x2 matrix
        A = np.zeros((2, 2))
        A[0, 0] = Avec[0]
        A[1, 1] = Avec[1]
        A[0, 1] = A[1, 0] = Avec[2]
        return A
    elif length == 6:  # Convert to 3x3 matrix
        A = np.zeros((3, 3))
        A[0, 0] = Avec[0]
        A[1, 1] = Avec[1]
        A[2, 2] = Avec[2]
        A[1, 2] = A[2, 1] = Avec[3]
        A[0, 2] = A[2, 0] = Avec[4]
        A[0, 1] = A[1, 0] = Avec[5]
        return A
    else:
        raise ValueError("Input vector must have a length of 3 (2x2 tensor) or 6 (3x3 tensor)")

def rotate(theta, A):
    """
    Rotate a tensor by an angle theta (in degrees).
    
    Parameters:
    - theta: Angle of rotation in degrees
    - A: Tensor to be rotated (either in matrix or vector form)
    
    Returns:
    - Rotated tensor in the same format as input (matrix or vector)
    """
    # Check if the input is a vector and convert to matrix if needed
    is_vector = A.shape[1] == 1
    A = tensor(A)
    
    # Convert theta from degrees to radians
    theta_rad = np.radians(theta)
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    
    # Rotation matrix
    T = np.array([[c, -s], [s, c]])
    
    # Perform the rotation
    A_rotated = T.T @ A @ T
    
    # Convert back to vector form if it was initially a vector
    if is_vector:
        return vector(A_rotated)
    
    return A_rotated
