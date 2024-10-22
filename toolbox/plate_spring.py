
class Spring:
    """
    A class representing a spring attached to a plate, providing elastic support.

    Attributes:
    - stiffness: Stiffness value of the spring (force/displacement).
    - x_bounds: Tuple representing the x-coordinate bounds where the spring is attached (e.g., (x_min, x_max)).
    - y_bounds: Tuple representing the y-coordinate bounds where the spring is attached (e.g., (y_min, y_max)).
    - direction: Direction of the spring's force (e.g., 'x', 'y', or 'z').
    - type: Type of spring (default is 0 for linear spring, 1 for torsional spring).

    Methods:
    - __init__: Initializes the spring with stiffness, bounds, direction, and type.
    """

    def __init__(self,stiffness,x_bounds,y_bounds,direction,type=0):
        """
        Initialize the Spring class.

        Parameters:
        - stiffness: Stiffness of the spring (force per unit displacement).
        - x_bounds: Tuple representing the x-coordinate bounds where the spring is attached (x_min, x_max).
        - y_bounds: Tuple representing the y-coordinate bounds where the spring is attached (y_min, y_max).
        - direction: Direction in which the spring force acts (e.g., 'x', 'y', or 'z').
        - type: Integer representing the type of spring (default is 0 for a linear spring, 1 for torsional spring).
        """
        self.stiffness = stiffness
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.direction = direction
        self.type = type
