#!/usr/bin/env python3
"""
Test script for the PD controller implementation.
This script demonstrates the complete closed-loop simulation and visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from uuv_mission.dynamic import Submarine, ClosedLoop, Mission
from uuv_mission.control import PDController

def test_pd_controller():
    """Test the PD controller with mission data and plot results."""
    
    # Create submarine and controller
    submarine = Submarine()
    controller = PDController(kp=0.15, kd=0.6)
    closed_loop = ClosedLoop(submarine, controller)
    
    # Load mission data
    mission = Mission.from_csv('data/mission.csv')
    print(f"Mission loaded: {len(mission.reference)} time steps")
    
    # Test with different disturbance levels
    disturbance_levels = [0.1, 0.5, 1.0]
    
    fig, axes = plt.subplots(len(disturbance_levels), 1, figsize=(12, 4*len(disturbance_levels)))
    if len(disturbance_levels) == 1:
        axes = [axes]
    
    for i, variance in enumerate(disturbance_levels):
        # Reset for new simulation
        submarine.reset_state()
        controller.reset()
        
        # Run simulation
        trajectory = closed_loop.simulate_with_random_disturbances(mission, variance=variance)
        
        # Plot results
        ax = axes[i]
        x_values = np.arange(len(mission.reference))
        min_depth = np.min(mission.cave_depth)
        max_height = np.max(mission.cave_height)
        
        # Plot cave boundaries
        ax.fill_between(x_values, mission.cave_height, mission.cave_depth, 
                       color='blue', alpha=0.3, label='Safe zone')
        ax.fill_between(x_values, mission.cave_depth, min_depth*np.ones(len(x_values)), 
                       color='saddlebrown', alpha=0.3, label='Bottom')
        ax.fill_between(x_values, max_height*np.ones(len(x_values)), mission.cave_height, 
                       color='saddlebrown', alpha=0.3, label='Top')
        
        # Plot trajectory and reference
        ax.plot(trajectory.position[:, 1], 'g-', linewidth=2, label='Actual trajectory')
        ax.plot(mission.reference, 'r--', linewidth=2, label='Reference')
        
        ax.set_title(f'PD Controller Performance (Disturbance variance: {variance})')
        ax.set_xlabel('Time step')
        ax.set_ylabel('Depth')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print performance metrics
    print("\nPerformance Analysis:")
    for variance in disturbance_levels:
        submarine.reset_state()
        controller.reset()
        trajectory = closed_loop.simulate_with_random_disturbances(mission, variance=variance)
        
        # Calculate tracking error
        tracking_error = np.mean(np.abs(trajectory.position[:, 1] - mission.reference))
        print(f"Disturbance variance {variance}: Mean absolute tracking error = {tracking_error:.3f}")

if __name__ == "__main__":
    test_pd_controller()
