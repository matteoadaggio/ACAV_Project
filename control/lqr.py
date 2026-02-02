import numpy as np
import math

class LQRController:
    def __init__(self, wheelbase=2.84):
        self.L = wheelbase
        
        # --- LQR Parameters ---
        # Q: Penalties on errors
        # [Lateral Error, Heading Error]
        self.Q = np.eye(2)
        self.Q[0,0] = 5.0   # Line distance Error Penalty
        self.Q[1,1] = 10.0  # Heading Error Penalty
        
        # R: Penalty on control input
        self.R = np.eye(1) * 300.0 
        
        # Discretization time (delta t between controls)
        self.dt = 0.1 

    def solve_dare(self, A, B, Q, R):
        """
        Solves the Discrete-time Algebraic Riccati Equation (DARE)
        to find the optimal matrix P.
        """
        P = Q
        max_iter = 100
        eps = 0.01

        for i in range(max_iter):
            Pn = A.T @ P @ A - A.T @ P @ B @ np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A + Q
            if (np.abs(Pn - P)).max() < eps:
                break
            P = Pn
        return P

    def compute_steering(self, waypoints, velocity=10.0):
        """
        Calculates steering using LQR based on the predicted trajectory.
        
        Args:
            waypoints: np.array [[x,y], ...] predicted by the network (in the car's frame)
            velocity: current velocity (m/s)
        """
        # 1. Calculate Current Errors relative to the trajectory
        # Since we are in the car's frame, our position is (0,0) and angle 0.
        # We need to find the lateral and angular error relative to the FIRST useful waypoint.
        
        # We take a point a bit ahead (Lookahead) for stability, e.g. the 3rd point
        # If we take the first point (0), it's too close and the control oscillates.
        idx = 3 
        if len(waypoints) <= idx: idx = len(waypoints) - 1
            
        target_x, target_y = waypoints[idx]
        
        # Lateral Error (e): y distance of the point (approximation for small angles)
        lateral_error = 0.0 - target_y 
        
        # Heading Error (th_e): angle of the line connecting us to the point
        target_angle = np.arctan2(target_y, target_x)
        heading_error = 0.0 - target_angle
        
        # --- LINEARIZED KINEMATIC MODEL ---
        # State: [lateral_error, heading_error]
        # Input: [steering_angle]
        
        # Matrix A (State evolution)
        # e_next = e + v * th_e * dt
        # th_e_next = th_e
        A = np.array([
            [1.0, velocity * self.dt],
            [0.0, 1.0]
        ])
        
        # Matrix B (Effect of steering)
        # e_next += 0
        # th_e_next += (v / L) * delta * dt
        B = np.array([
            [0.0],
            [(velocity / self.L) * self.dt]
        ])
        
        R_running = self.R * 10.0 # Increase penalty for sharp steering
        
        # --- CALCULATE GAIN K ---
        P = self.solve_dare(A, B, self.Q, R_running)
        
        # K = (R + B^T P B)^-1 * (B^T P A)
        K = np.linalg.inv(R_running + B.T @ P @ B) @ (B.T @ P @ A)
        
        # --- CONTROL LAW ---
        # u = -K * x
        x_state = np.array([[lateral_error], [heading_error]])
        steering_input = - (K @ x_state)
        
        # Extract scalar and clamp (physical steering limit ~35 degrees)
        steer_rad = steering_input[0,0]
        max_steer = np.radians(35)
        steer_rad = np.clip(steer_rad, -max_steer, max_steer)
        
        # Feedforward (optional)
        
        return steer_rad

"""# --- TEST ---
if __name__ == "__main__":
    controller = LQRController()
    
    # Case: Left curve road (positive y)
    path = np.array([[2, 0.2], [4, 0.8], [6, 1.5], [10, 3.0]])
    steer = controller.compute_steering(path, velocity=5.0)
    print(f"Path left   -> LQR Steering: {np.degrees(steer):.2f}°")
    
    # Case: Right curve road (negative y)
    path_r = np.array([[2, -0.2], [4, -0.8], [6, -1.5], [10, -3.0]])
    steer_r = controller.compute_steering(path_r, velocity=5.0)
    print(f"Path right   -> LQR Steering: {np.degrees(steer_r):.2f}°")"""