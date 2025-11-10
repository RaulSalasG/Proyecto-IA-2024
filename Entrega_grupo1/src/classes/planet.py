import numpy as np
from enum import Enum

class CellType(Enum):
    """Enumeration representing the different types of cells on a planet
    
    Cell Types:
        EMPTY: Cell containing no resources, accidents or obstacles
        WATER: Cell containing waters
        MINERAL: Cell containing minerals
        OBSTACLE: Untouchable cell
        ACCIDENT: Cell that disables rovers 
    """
    EMPTY = 0
    WATER = 1
    MINERAL = 2
    OBSTACLE = 3
    ACCIDENT = 4

class Planet:
    """Class representing a planet(toroid) with a grid of cells for rover exploration.
    """
    
    def __init__(
        self, 
        name: str, 
        width: int, 
        height: int, 
        water_prob: float, 
        mineral_prob: float,
        obstacle_prob: float,
        accident_prob: float,
        seed=None
    ):
        """Initialize a planet with randomized cell distribution.
        
        Args:
            name: Planet identifier
            width: Grid horizontal size in cells
            height: Grid vertical size in cells
            water_prob: Probability of water in a cell
            mineral_prob: Probability of minerals in a cell
            obstacle_prob: Probability of obstacles in a cell
            accident_prob: Probability of accidents in a cell
            seed: Random seed for reproducivity

        """
        self.name = name
        self.width = width
        self.height = height

        if seed is not None:
            np.random.seed(seed)
        
        # Validate probability distribution
        total_prob = water_prob + mineral_prob + obstacle_prob + accident_prob
        if total_prob > 1.0:
            raise ValueError("Total probability cannot exceed 1.0")
        
        # Generate planet grid with specified cell distribution
        self.grid = np.random.choice(
            list(CellType),
            size=(width, height),
            p=[
                1.0 - total_prob,  # Empty cell probability 
                water_prob,
                mineral_prob,
                obstacle_prob,
                accident_prob
            ]
        )
    
    def get_cell(self, pos: tuple[int, int]) -> CellType:
        """Retunrs cell content at specified coordinates. Implements toroidal coordinate wrapping
        
        Args:
            pos: (x,y) coordinates (wraps around planet edges)
            
        Returns:
            CellType: Type of the requested cell
        
        """
        x, y = pos[0] % self.width, pos[1] % self.height
        return self.grid[x, y]
    
    def empty_cell(self, pos: tuple[int, int]) -> bool:
        """Clear a water or mineral cell and mark it as empty. Implements toroidal coordinate wrapping
        
        Args:
            pos: (x,y) coordinates of target cell
            
        Returns:
            bool: True if cell was cleared, False if cell contained no resources
            
        """
        x, y = pos[0] % self.width, pos[1] % self.height
        cell = self.grid[x, y]
    
        if cell in (CellType.WATER, CellType.MINERAL):
            self.grid[x, y] = CellType.EMPTY
            return True
        return False