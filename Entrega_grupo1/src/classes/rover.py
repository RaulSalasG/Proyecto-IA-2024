import numpy as np
from enum import Enum
from .planet import CellType

class RoverState(Enum):
    """Enumeration representing the possible states of a rover
    
    States:
    EXPLORING: Rover is moving and searching for resources
    GATHERING: Rover is collecting resources from the cell it is in
    LOST: Rover has had an accident and it got lost
    """
    EXPLORING = 1
    GATHERING = 2
    LOST = 3

class Rover:
    """Class representing an exploration rover with movement and resource collection capabilities.
    """
    def __init__(self, rover_id: int, planet, width: int, height: int):
        """Initialize a rover with starting position and empty load.
        
        Args:
            rover_id: Unique identifier
            planet: Planet where the rover operates
            width: Initial x-coordinate position
            height: Initial y-coordinate position
        """
        initial_location = (width, height)
        self.id = rover_id
        self.planet = planet
        self.location = np.array(initial_location)
        self.state = RoverState.EXPLORING
        self.load = {"WATER": 0, "MINERAL": 0}
        self.findings_locations = {"WATER": [], "MINERAL": []}
    
    def move(self, direction: str) -> bool:
        """Move the rover in one of four cardinal directions with toroidal wrapping.
        
        Args:
            direction: Movement direction ('North', 'South', 'East', or 'West')
            
        Returns:
            bool: True if movement was successful, False if rover is blocked by obstacle
            
        """
        """Mueve el rover en una dirección (N/S/E/O). Devuelve True si el movimiento fue exitoso."""
        movements = {'N': (0, 1), 'S': (0, -1), 'E': (1, 0), 'W': (-1, 0)}
        dx, dy = movements[direction]
        new_location = (self.location + np.array([dx, dy])) % [self.planet.width, self.planet.height]
        
        if self.planet.get_cell(new_location) == CellType.OBSTACLE:
            return False
        
        self.location = new_location
        self._update_state()
        return True
    
    def gather(self):
        """Collect resources from current cell if available and update load.
        
        Functions:
            - Increments load
            - Marks cell as empty after gathering
            - Saves collection location
        """
        cell = self.planet.get_cell(self.location)
        if cell == CellType.WATER:
            self.load["WATER"] += 1
            self.planet.empty_cell(self.location)
            self.findings_locations["WATER"].append(self.location)
        elif cell == CellType.MINERAL:
            self.load["MINERAL"] += 1
            self.planet.empty_cell(self.location)
            self.findings_locations["MINERAL"].append(self.location)
    
    def _update_state(self):
        """Internal method to update rover state based on current cell content.

        - Sets GATHERING state when the rover is on resource cells
        - Set LOST state when the rover faced an accident
        - Default: Set EXPLORING state
        """
        cell = self.planet.get_cell(self.location)
        if cell == CellType.WATER or cell == CellType.MINERAL:
            self.state = RoverState.GATHERING
        elif cell == CellType.ACCIDENT:
            self.state = RoverState.LOST
        else:
            self.state = RoverState.EXPLORING
    