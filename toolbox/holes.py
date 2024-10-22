
class SquareHole:
    """
    A class representing a square hole in a plate or material.

    Attributes:
    - x_bounds: Tuple representing the x-coordinate bounds of the square hole (e.g., (x_min, x_max)).
    - y_bounds: Tuple representing the y-coordinate bounds of the square hole (e.g., (y_min, y_max)).

    Methods:
    - __init__: Initializes the square hole with specified x and y bounds.
    """
    
    def __init__(self,x_bounds,y_bounds):
        """
        Initialize the SquareHole class.

        Parameters:
        - x_bounds: Tuple representing the x-coordinate bounds of the hole (x_min, x_max).
        - y_bounds: Tuple representing the y-coordinate bounds of the hole (y_min, y_max).
        """
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
