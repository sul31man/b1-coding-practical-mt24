import numpy as np

class PDController:
    """
    Proportional-Derivative (PD) feedback controller.
    
    Control law: u[t] = Kp * e[t] + KD * (e[t] - e[t-1])
    where e[t] = r[t] - y[t] (error between reference and output)
    """
    
    def __init__(self, kp: float = 0.15, kd: float = 0.6):
        """
        Initialize PD controller with gains.
        
        Args:
            kp: Proportional gain (default: 0.15)
            kd: Derivative gain (default: 0.6)
        """
        self.kp = kp
        self.kd = kd
        self.previous_error = 0.0
        self.is_first_step = True
    
    def compute_control_action(self, reference: float, output: float) -> float:
        """
        Compute control action based on reference and current output.
        
        Args:
            reference: Desired reference value r[t]
            output: Current output value y[t]
            
        Returns:
            Control action u[t]
        """
        # Calculate current error
        error = reference - output
        
        if self.is_first_step:
            # For the first step, derivative term is zero
            derivative_term = 0.0
            self.is_first_step = False
        else:
            # Calculate derivative term
            derivative_term = error - self.previous_error
        
        # Compute control action
        control_action = self.kp * error + self.kd * derivative_term
        
        # Store current error for next iteration
        self.previous_error = error
        
        return control_action
    
    def reset(self):
        """Reset controller state for new simulation."""
        self.previous_error = 0.0
        self.is_first_step = True
