"""
    sim.py
    
    The main simulation runs in here. We generate a number of planets, fleets for each planet and
    number of rovers in each fleet, as well as the size of the planets and the steps that each rover 
    takes. We also generate a range of possible values that the probabilities for every cell of a planet
    to have minerals, water or accidents can take.
    A fleet is created and filled with a number of rovers, which are dropped in random but different
    positions of the planet. The behavior of the rovers is also defined in this class. 
    As a result, the program generates a csv file 'results.csv' that contains the results obtained by
    running the simulation. This file is used for an exhaustive analysis in '.ipynb'  
    
    CUNEF Universidad
    Doble Grado en Ingeniería Informática y Administración de Empresas
    Inteligencia Artificial
    Abril 2025

    Discrete Event Simulation: Simulation of Planetarian Exploration
    
    
    @author Pablo Ceballos 
    @author Nicolas Mangas
    @author Raúl Salas
"""


import itertools
import numpy as np
import pandas as pd
import simpy
from classes.planet import Planet, CellType
from classes.rover import Rover, RoverState


class Fleet:
    def __init__(self, env, planet, num_rovers, distance):
        """Constructor for the class Fleet that manages a group of rovers exploring a planet.

        Args:
            env (simpy.Environment): The simulation environment where the fleet operates
            planet (Planet): The planet being explored by the fleet
            num_rovers (int): Number of rovers in the fleet

        """
        self.env = env
        self.planet = planet
        self.rovers = []
        self.lost_rovers = 0
        self.water_total = 0
        self.mineral_total = 0
        self.quadrants = np.array_split(planet.grid, num_rovers)  # Divide the grid
        
        # Generates rovers for the fleet with random positions in the planet
        for i in range(num_rovers):
            x_pos, y_pos = ((i * planet.width) // num_rovers) + distance, (i * planet.height) // num_rovers
            rover = Rover(
                i+1, 
                planet,
                x_pos,
                y_pos
            )
            self.rovers.append(rover)
            env.process(self.rover_behavior(rover))

    def rover_behavior(self, rover: Rover):
        """Defines the behavior of a rover in each simulation step(a time of the simulation) 

        The rover will:
        1-> Gather resources if in a resource cell
        2-> Move in a random direction
        3-> Check if it has been lost (accident)
        4-> Wait for the next time step

        Args:
            rover(Rover): 

        Yields:
            simpy.Timeout: Controls the timing between simulation steps

        """
        while rover.state != RoverState.LOST:
            
            if rover.state == RoverState.GATHERING:
                cell_before = self.planet.get_cell(rover.location)
                rover.gather()
                cell_after = self.planet.get_cell(rover.location)
                
                # Verifies that the rover changes position
                if cell_before != cell_after: 
                    if cell_before == CellType.WATER:
                        self.water_total +=1
                    elif cell_before == CellType.MINERAL:
                        self.mineral_total +=1
                
            # Movement order for the rover
            direction = np.random.choice(['N', 'S', 'E', 'W'])
            rover.move(direction)
        
            # In case a rover gets lost by any reason, we eliminate it from the fleet
            if rover.state == RoverState.LOST:
                self.lost_rovers += 1

            # Wait until the next step
            yield self.env.timeout(1)
    


def sim(num_planets: int, num_fleets:int, num_rovers:int, sim_times:int, planet_size:int, water_probs: np.array,
         mineral_probs: np.array, accident_probs:np.array, seed:int) -> pd.DataFrame:
    
    """Simulation of planet exploration with Cartesian Product

    Args:
        num_planets (int): Number of Planets explored
        num_fleets (int): Number of fleets sent to a planet
        num_rovers (int): Number of rovers in a fleet
        sim_times (int): Number of times the simulation runs. Number of steps a rover takes
        planet_size (int): Number of cells in x and y axys for every planet
        water_probs (np.array): Probabilities of finding water in a cell of a planet
        mineral_probs (np.array): Probabilities of finding minerals in a cell of a planet
        accident_probs (np.array): Probabilities of being accidented in a cell of a planet
        seed (int): Seed for reproducivity

    Returns:
        pd.DataFrame: Results of the cartesian product
    """
    
    # DataFrame that will contain the result of the simulation
    df_results = pd.DataFrame()
    
    # Asigns random combinations of the probabilities(no repetition)
    np.random.seed(seed)
    combinations = list(itertools.product(
        water_probs, 
        mineral_probs, 
        accident_probs
    ))
    combinations_used = np.random.choice(len(combinations), size=num_planets, replace=False)
    
    # Runs all simulations for each planet
    for i in range(num_planets):
        planet_num = i + 1
        water_probs, mineral_probs, accident_probs = combinations[combinations_used[i]]
        
        # Print basic information of the planet
        print(f"\n===============================\n         PLANET {planet_num} \n===============================")
        print(f"Configuration of probabilities:\n-Water: {water_probs}\n-Mineral:"
              f" {mineral_probs}\n-Accidents: {accident_probs} \n===============================")
        
        # Create planet with the parametres given
        planet = Planet(
            name=f"Planet {planet_num}",
            width=planet_size,
            height=planet_size,
            water_prob=water_probs,
            mineral_prob=mineral_probs,
            obstacle_prob=0.05,  # Stays the same for every planet
            accident_prob=accident_probs,
            seed=planet_num  # Unique seed for every planet
        )
        
        # Runs all the different fleets for the same planet
        for exploration_num in range(1, num_fleets + 1):

            # Unique seed for every exploration
            np.random.seed(planet_num * 100 + exploration_num)
            
            # Run Simulation
            # Creates an environment and processes in it
            env = simpy.Environment()
            distance = int(planet.width/num_rovers)
            fleet = Fleet(env, planet, num_rovers, distance)
            env.run(until=sim_times)

            #Print the exploration results in the terminal
            print(f"--Exploration {exploration_num}...COMPLETED -> [Water:{fleet.water_total}"
                  f", Minerals:{fleet.mineral_total}, Lost Rovers:{fleet.lost_rovers}]")
            

            # Save results
            df_temp = pd.DataFrame.from_dict({
                'Planet': [planet_num],
                'Exploration': [exploration_num],
                'Watet_probs': [water_probs],
                'Mineral_probs': [mineral_probs],
                'Accident_probs': [accident_probs],
                'Water_gathered': [fleet.water_total],
                'Mineral_gathered': [fleet.mineral_total],
                'Total_gathered': [fleet.water_total + fleet.mineral_total],
                'Lost_rovers': [fleet.lost_rovers]
            })

            # Concat the DataFrame to all the results 
            df_results = pd.concat([df_results, df_temp])
    
    return df_results


if __name__ == "__main__":
    
    # Configuration data of the simulation
    num_planets = 27
    num_fleets = 20 
    num_rovers = 20
    sim_times = 20
    planet_size = 100
    water_probs = [0.025, 0.05, 0.1]  
    mineral_probs = [0.05, 0.1, 0.15]
    accident_probs = [0.005, 0.01, 0.015]
    seed = 0 
    
    
    # Run simulation and save results in the DataFrame
    df_results = sim(num_planets, num_fleets, num_rovers, sim_times, planet_size, water_probs, mineral_probs, accident_probs, seed)
    
    # Convert DataFrame to csv file for further analysis
    df_results.to_csv("results.csv", index=False)
    print("\nSimulation completed and saved in 'results.csv'")