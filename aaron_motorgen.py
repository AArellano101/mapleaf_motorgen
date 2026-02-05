"""
This program processes Wizgod motor's static fire test data from a rocket motor and generates
a MapleLeaf-compatible motor definition file.

Author: Aaron Arellano

"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Optional, Tuple, List

# ==============================================================================
# CONFIGURATION SECTION
# ==============================================================================

TEST = False  # Set to True to run unit tests instead of full processing

class MotorConfig:
    """Configuration parameters for the motor and processing options"""
    
    def __init__(self):
        # File paths
        self.input_file = r'C:\Users\aaron\soar_oct_8\StaticFireData.txt'
        self.output_file = r'C:\Users\aaron\soar_jan_10\aaron_motorgen\Aegis3P5kN_4S.txt'
        
        # Processing options
        self.extend_burn = True          # Extend the motor burn time
        self.extension_duration = 1.3      # Extension time in seconds
        self.use_estimation = True        # Scale thrust to target average
        self.estimated_avg_thrust = 3500   # Target average thrust (N)
        self.write_output = True          # Write output file
        self.zero_moi = True              # Set MOI to zero (for simplified models)
        
        # Dynamic flow rate calculation options
        self.use_actual_flow_rates = False     # Use measured flow rates from load cell
        self.flow_smoothing_window = 20       # Smoothing window for flow rate calculation
        

        # Motor geometry - Pressure vessel (oxidizer tank)
        self.pressure_vessel_diameter = 0.17  # meters
        self.pressure_vessel_length = 0.813 # meters
        
        # Motor geometry - Fuel grain
        self.fuel_grain_length = 0.290576         # meters
        self.fuel_grain_outer_diameter = 0.15     # meters
        self.fuel_grain_inner_diameter = 0.05715  # meters
        
        # Combustion parameters
        self.of_ratio = 6.58             
        
        # Motor burn detection threshold
        self.chamber_pressure_threshold = 15  # PSI - pressure to detect motor start

        # Initial/Final CGs
        self.burnedOxidizerInitialCG = -3.8
        self.burnedOxidizerLastCG = -3.4
        self.burnedFuelInitialCG = -3.75  
        self.burnedFuelLastCG = -3.75

        self.target_mass = False

        self.name = "Aaron Aegis3P5kN_4S"

        self.save_plots = True


# ==============================================================================
# MOMENT OF INERTIA CALCULATIONS
# ==============================================================================

class MomentOfInertia:
    """Calculate moments of inertia for various geometric shapes"""
    
    @staticmethod
    def solid_cylinder(mass, length, radius, zeroed=False):
        """
        Calculate MOI for a solid cylinder (oxidizer tank approximation)
        
        Args:
            mass: Mass of cylinder (kg)
            length: Length of cylinder (m)
            radius: Radius of cylinder (m)
            zeroed: If True, return zero MOI (for simplified models)
            
        Returns:
            [MOI_x, MOI_y, MOI_z] in kg*m^2
        """
        if zeroed:
            return [0, 0, 0]
        
        # Perpendicular to cylinder axis (x and y)
        moi_xy = (0.25 * mass * radius**2) + ((1/12) * mass * length**2)
        
        # Along cylinder axis (z)
        moi_z = 0.5 * mass * radius**2
        
        return [moi_xy, moi_xy, moi_z]
    
    @staticmethod
    def hollow_cylinder(mass, length, inner_radius, outer_radius, zeroed=False):
        """
        Calculate MOI for a hollow cylinder (fuel grain)
        
        Args:
            mass: Mass of cylinder (kg)
            length: Length of cylinder (m)
            inner_radius: Inner radius of cylinder (m)
            outer_radius: Outer radius of cylinder (m)
            zeroed: If True, return zero MOI (for simplified models)
            
        Returns:
            [MOI_x, MOI_y, MOI_z] in kg*m^2
        """
        if zeroed:
            return [0, 0, 0]
        
        # Perpendicular to cylinder axis (x and y)
        moi_xy = (mass/12) * (3 * (inner_radius**2 + outer_radius**2) + length**2)
        
        # Along cylinder axis (z)
        moi_z = (mass/2) * (inner_radius**2 + outer_radius**2)
        
        return [moi_xy, moi_xy, moi_z]
    

# ==============================================================================
#  DynamicFlowRateCalculator
# ==============================================================================

class DynamicFlowRateCalculator:
    """
    Calculate actual oxidizer and fuel flow rates from test data
    """
    
    @staticmethod
    def calculate_from_load_cell(times: List[float], 
                                 load_cell_mass: List[float],
                                 of_ratio: float,
                                 smoothing_window: int = 10) -> Tuple[List[float], List[float]]:
        """
        Calculate actual flow rates from load cell mass measurements
        
        Args:
            times: Array of time points (s)
            load_cell_mass: Oxidizer tank mass from load cell (kg)
            of_ratio: Oxidizer-to-fuel mass ratio
            smoothing_window: Window size for smoothing the derivative (points)
            
        Returns:
            (oxidizer_flow_rates, fuel_flow_rates) in kg/s
        """
        
        # Calculate oxidizer mass change rate (negative because mass decreases)
        # Use central difference for better accuracy
        ox_flow_rates = []
        
        for i in range(len(times)):
            if i == 0:
                # Forward difference for first point
                dt = times[i+1] - times[i]
                dm = load_cell_mass[i+1] - load_cell_mass[i]
                flow_rate = -dm / dt  # Negative because mass decreases
            elif i == len(times) - 1:
                # Backward difference for last point
                dt = times[i] - times[i-1]
                dm = load_cell_mass[i] - load_cell_mass[i-1]
                flow_rate = -dm / dt
            else:
                # Central difference for middle points
                dt = times[i+1] - times[i-1]
                dm = load_cell_mass[i+1] - load_cell_mass[i-1]
                flow_rate = -dm / dt
            
            ox_flow_rates.append(flow_rate)
        
        # Apply smoothing to reduce noise
        ox_flow_rates_smoothed = DynamicFlowRateCalculator.smooth(
            ox_flow_rates, smoothing_window
        )
        
        # Calculate fuel flow rate from O/F ratio
        # O/F = m_ox / m_fuel, so m_fuel = m_ox / (O/F)
        fuel_flow_rates = [ox_rate / of_ratio for ox_rate in ox_flow_rates_smoothed]
        
        return ox_flow_rates_smoothed, fuel_flow_rates
    
    @staticmethod
    def smooth(data: List[float], window: int) -> List[float]:
        """
        Apply moving average smoothing
        
        Args:
            data: Input data array
            window: Window size (odd number recommended)
            
        Returns:
            Smoothed data array
        """
        if window <= 1:
            return data
        
        # Ensure window is odd
        if window % 2 == 0:
            window += 1
        
        half_window = window // 2
        smoothed = []
        
        for i in range(len(data)):
            # Calculate window bounds
            start_idx = max(0, i - half_window)
            end_idx = min(len(data), i + half_window + 1)
            
            # Calculate mean in window
            window_data = data[start_idx:end_idx]
            smoothed.append(np.mean(window_data))
        
        return smoothed





# ==============================================================================
# DATA PROCESSING CLASS
# ==============================================================================

class MotorDataProcessor:
    """Process raw motor test data and generate MapleLeaf motor file"""
    
    def __init__(self, config):
        """
        Initialize processor with configuration
        
        Args:
            config: MotorConfig object with processing parameters
        """
        self.config = config
        self.raw_data = None
        self.motor_start_idx = 0
        self.motor_end_idx = 0
        
        # Processed data arrays
        self.times = []
        self.thrust = []
        self.chamber_pressure = []
        self.oxidizer_moi = []
        self.fuel_moi = []

        # Actual flow rate arrays
        self.oxidizer_flow_rates = []
        self.fuel_flow_rates = []
        self.load_cell_masses = []  
        
        # Store original data before modifications (for comparison)
        self.original_times = []
        self.original_thrust = []
        
        # Calculated parameters
        self.oxidizer_flowrate = 0
        self.fuel_flowrate = 0
        self.total_ox_mass = 0
        self.total_fuel_mass = 0
        self.initial_time = 0
        
    def load_data(self):
        """Load raw test data from CSV file"""
        print(f"Loading data from: {self.config.input_file}")
        self.raw_data = pd.read_csv(self.config.input_file, sep=',', index_col=False)
        print(f"Loaded {len(self.raw_data)} data points")
        
    def find_motor_burn_window(self):
        """
        Identify the start and end indices of motor burn
        
        Start: First point where chamber pressure exceeds threshold
        End: First 'Standby' comment after motor start
        """
        # Find motor start
        for i in range(len(self.raw_data)):
            if self.raw_data['Chamber Pressure (PSI)'][i] >= self.config.chamber_pressure_threshold:
                self.motor_start_idx = i
                break
        
        # Find motor end
        for j in range(len(self.raw_data)):
            if (self.raw_data['Comment'][j] == 'Standby' and 
                j > self.motor_start_idx and 
                j > 0):
                self.motor_end_idx = j
                break
        
        print(f"Motor burn detected:")
        print(f"  Start: Line {self.motor_start_idx + 2} (row index {self.motor_start_idx})")
        print(f"  End: Line {self.motor_end_idx + 2} (row index {self.motor_end_idx})")
        
        self.initial_time = self.raw_data['Time (s)'][self.motor_start_idx]
        
    def calculate_flow_rates(self):
        """
        Calculate oxidizer and fuel flow rates from test data
        
        Oxidizer flow: Calculated from load cell mass change over burn time
        Fuel flow: Derived from oxidizer flow using O/F ratio
        """
        # Calculate oxidizer flow rate from mass change
        ox_mass_start = self.raw_data['Ox Tank Load Cell (kg)'][self.motor_start_idx]
        ox_mass_end = self.raw_data['Ox Tank Load Cell (kg)'][self.motor_end_idx]
        burn_time = self.raw_data['Time (s)'][self.motor_end_idx] - self.initial_time
        
        # Total oxidizer consumed (absolute value since load cell shows mass decrease)
        self.total_ox_mass = abs(ox_mass_start - ox_mass_end)
        
        # Oxidizer mass flow rate (kg/s)
        self.oxidizer_flowrate = self.total_ox_mass / burn_time
        
        # Calculate fuel flow rate using O/F ratio
        self.fuel_flowrate = self.oxidizer_flowrate / self.config.of_ratio
        
        # Total fuel consumed
        self.total_fuel_mass = self.fuel_flowrate * burn_time
        
        print(f"\nFlow rate calculations:")
        print(f"  Burn time: {burn_time:.3f} s")
        print(f"  Total oxidizer mass: {self.total_ox_mass:.3f} kg")
        print(f"  Total fuel mass: {self.total_fuel_mass:.3f} kg")
        print(f"  Oxidizer flow rate: {self.oxidizer_flowrate:.4f} kg/s")
        print(f"  Fuel flow rate: {self.fuel_flowrate:.4f} kg/s")
        print(f"  O/F ratio: {self.config.of_ratio:.2f}")
        
    def extract_burn_data(self):
        """
        Extract and process data points during motor burn
        
        Converts units:
        - Time: Relative to motor ignition (s)
        - Thrust: lbf to N (subtract baseline, multiply by 4.4482216)
        - Pressure: Keep in PSI
        
        Calculates MOI for oxidizer (solid cylinder) and fuel (hollow cylinder)
        at each time step based on remaining mass
        """
        # Geometry calculations
        pv_radius = self.config.pressure_vessel_diameter / 2
        fg_outer_radius = self.config.fuel_grain_outer_diameter / 2
        fg_inner_radius = self.config.fuel_grain_inner_diameter / 2
        
        # Baseline thrust (for zeroing)
        baseline_thrust = self.raw_data['Thrust Load Cell (lbf)'][self.motor_start_idx]

        # Extract load cell mass data
        self.load_cell_masses = self.raw_data.iloc[self.motor_start_idx:self.motor_end_idx + 1]['Ox Tank Load Cell (kg)'].values.tolist()
        
        # Extract data for each time point during burn
        for idx in range(self.motor_start_idx, self.motor_end_idx + 1):
            # Time relative to ignition
            time = self.raw_data['Time (s)'][idx] - self.initial_time
            self.times.append(time)
            
            # Chamber pressure (PSI)
            self.chamber_pressure.append(self.raw_data['Chamber Pressure (PSI)'][idx])
            
            # Thrust (convert lbf to N, zero at motor start)
            thrust_lbf = self.raw_data['Thrust Load Cell (lbf)'][idx] - baseline_thrust
            thrust_n = thrust_lbf * 4.4482216
            self.thrust.append(thrust_n)
        
        if self.config.use_actual_flow_rates and len(self.load_cell_masses) > 0:
            print(f"\nCalculating flow rates from load cell data...")
            
            self.oxidizer_flow_rates, self.fuel_flow_rates = \
                DynamicFlowRateCalculator.calculate_from_load_cell(
                    self.times,
                    self.load_cell_masses,
                    self.config.of_ratio,
                    self.config.flow_smoothing_window
                )
            
            # Verify mass conservation
            integrated_ox_mass = np.trapz(self.oxidizer_flow_rates, self.times)
            integrated_fuel_mass = np.trapz(self.fuel_flow_rates, self.times)
            
            print(f"   Integrated oxidizer mass: {integrated_ox_mass:.4f} kg")
            print(f"   Integrated fuel mass: {integrated_fuel_mass:.4f} kg")
            print(f"   Mass conservation error: {abs(integrated_ox_mass - self.total_ox_mass)/self.total_ox_mass * 100:.2f}%")
            
            print(f"   Flow rate range:")
            print(f"     Oxidizer: {min(self.oxidizer_flow_rates):.4f} to {max(self.oxidizer_flow_rates):.4f} kg/s")
            print(f"     Fuel: {min(self.fuel_flow_rates):.4f} to {max(self.fuel_flow_rates):.4f} kg/s")
        else:
            #  Use constant flow rates (original behavior)
            print(f"\nUsing constant (average) flow rates")
            self.oxidizer_flow_rates = [self.oxidizer_flowrate] * len(self.times)
            self.fuel_flow_rates = [self.fuel_flowrate] * len(self.times)
        

        for i, time in enumerate(self.times):
            # Calculate consumed mass up to this point by integrating flow rates
            if i == 0:
                ox_mass_consumed = 0
                fuel_mass_consumed = 0
            else:
                time_slice = self.times[:i+1]
                ox_flow_slice = self.oxidizer_flow_rates[:i+1]
                fuel_flow_slice = self.fuel_flow_rates[:i+1]
                
                ox_mass_consumed = np.trapz(ox_flow_slice, time_slice)
                fuel_mass_consumed = np.trapz(fuel_flow_slice, time_slice)
            
            ox_mass_remaining = self.total_ox_mass - ox_mass_consumed
            fuel_mass_remaining = self.total_fuel_mass - fuel_mass_consumed
            
            # Calculate MOI for oxidizer (solid cylinder model)
            ox_moi = MomentOfInertia.solid_cylinder(
                ox_mass_remaining,
                self.config.pressure_vessel_length,
                pv_radius,
                self.config.zero_moi
            )
            self.oxidizer_moi.append(ox_moi)
            
            # Calculate MOI for fuel (hollow cylinder model)
            fuel_moi = MomentOfInertia.hollow_cylinder(
                fuel_mass_remaining,
                self.config.fuel_grain_length,
                fg_inner_radius,
                fg_outer_radius,
                self.config.zero_moi
            )
            self.fuel_moi.append(fuel_moi)
        
        
        print(f"\nExtracted {len(self.times)} data points")
        print(f"  Time range: {self.times[0]:.3f} to {self.times[-1]:.3f} s")
        print(f"  Thrust range: {min(self.thrust):.2f} to {max(self.thrust):.2f} N")
        print(f"  Average thrust: {np.mean(self.thrust):.2f} N")
        
        # Save original data for comparison (before extension or scaling)
        self.original_times = self.times.copy()
        self.original_thrust = self.thrust.copy()
        
    def extend_burn_time(self):
        """
        Extend motor burn time by duplicating stable thrust region
        
        This is useful for testing or simulation purposes where a longer
        burn time is desired. The function:
        1. Finds a stable thrust region (relatively constant thrust)
        2. Extends the burn by REPEATING the actual thrust pattern from that region
        3. Adjusts time values and recalculates MOI
        
        This preserves the natural variation in the thrust curve rather than
        creating a flat line.
        """
        if not self.config.extend_burn or self.config.extension_duration <= 0:
            return
        
        print(f"\nExtending burn time by {self.config.extension_duration} seconds...")
        
        # Find stable thrust region (middle portion of burn)
        # We'll use indices where thrust doesn't vary too much
        avg_thrust = np.mean(self.thrust)
        max_thrust = max(self.thrust)
        
        # Find the stable region by looking for where thrust is relatively constant
        stable_start_idx = None
        stable_end_idx = None
        window_size = 500  # Look at 500-point windows
        
        for i in range(len(self.thrust) - window_size):
            window = self.thrust[i:i+window_size]
            window_std = np.std(window)
            window_mean = np.mean(window)
            
            # Check if this window is in a stable region
            # (not too close to max, reasonable variation, after initial spike)
            if (window_std < 100 and  # Low variation
                window_mean > avg_thrust * 0.7 and  # Decent thrust
                window_mean < max_thrust * 0.95):  # Not at peak
                
                stable_start_idx = i
                stable_end_idx = i + window_size
                print(f"  Found stable thrust region: indices {stable_start_idx} to {stable_end_idx}")
                print(f"    Mean thrust: {window_mean:.1f} N, Std: {window_std:.1f} N")
                break
        
        if stable_start_idx is None:
            print("  Warning: Could not find stable thrust region, using middle section")
            # Fallback: use middle third of the burn
            total_points = len(self.thrust)
            stable_start_idx = total_points // 3
            stable_end_idx = 2 * total_points // 3
        
        # Extract the stable thrust pattern
        stable_thrust_pattern = self.thrust[stable_start_idx:stable_end_idx]
        stable_time_pattern = [self.times[i] - self.times[stable_start_idx] 
                               for i in range(stable_start_idx, stable_end_idx)]
        
        stable_ox_flow_pattern = self.oxidizer_flow_rates[stable_start_idx:stable_end_idx]
        stable_fuel_flow_pattern = self.fuel_flow_rates[stable_start_idx:stable_end_idx]
        
        
        print(f"  Using {len(stable_thrust_pattern)} points from stable region")
        
        # Calculate how many times we need to repeat the pattern
        timestep = self.times[1] - self.times[0]
        pattern_duration = stable_time_pattern[-1]
        num_repetitions = int(np.ceil(self.config.extension_duration / pattern_duration))
        
        # Find insertion point (middle of the burn)
        insert_idx = len(self.thrust) // 2
        insert_time = self.times[insert_idx]
        
        print(f"  Repeating pattern {num_repetitions} times to achieve {self.config.extension_duration}s extension")
        
        # Create extended thrust and time arrays
        extension_thrust = []
        extension_times = []

        extension_ox_flow = []
        extension_fuel_flow = []
        
        for rep in range(num_repetitions):
            for i, thrust_val in enumerate(stable_thrust_pattern):
                new_time = insert_time + (rep * pattern_duration) + stable_time_pattern[i]
                # Only add points that fit within our extension duration
                if rep * pattern_duration + stable_time_pattern[i] < self.config.extension_duration:
                    extension_thrust.append(thrust_val)
                    extension_times.append(new_time)

                    extension_ox_flow.append(stable_ox_flow_pattern[i])
                    extension_fuel_flow.append(stable_fuel_flow_pattern[i])
        
        print(f"  Created {len(extension_thrust)} extension points")
        
        # Insert extension data at the middle point
        for i in range(len(extension_thrust)):
            self.times.insert(insert_idx + i, extension_times[i])
            self.thrust.insert(insert_idx + i, extension_thrust[i])

            self.oxidizer_flow_rates.insert(insert_idx + i, extension_ox_flow[i])
            self.fuel_flow_rates.insert(insert_idx + i, extension_fuel_flow[i])
        
        # Adjust times after the insertion point
        time_shift = self.config.extension_duration
        for i in range(insert_idx + len(extension_thrust), len(self.times)):
            self.times[i] += time_shift
        
        # Recalculate MOI for all points
        self._recalculate_moi()
        
        print(f"  Extended burn to {self.times[-1]:.3f} seconds")
        print(f"  New total points: {len(self.times)}")
        print(f"  New average thrust: {np.mean(self.thrust):.1f} N")
        
    def apply_thrust_estimation(self):
        """
        Scale thrust curve to match target average thrust
        
        This is useful when you have theoretical calculations (e.g., from
        NASA CEA) and want to scale test data to match expected performance.
        
        The scaling preserves the shape of the thrust curve while adjusting
        the magnitude to match the target average.
        """
        if not self.config.use_estimation:
            return
        
        print(f"\nScaling thrust to target average: {self.config.estimated_avg_thrust} N")
        
        current_avg = np.mean(self.thrust)
        scale_factor = self.config.estimated_avg_thrust / current_avg
        
        print(f"  Current average: {current_avg:.2f} N")
        print(f"  Scale factor: {scale_factor:.4f}")
        
        # Apply scaling
        self.thrust = [t * scale_factor for t in self.thrust]
        
        print(f"  New average: {np.mean(self.thrust):.2f} N")
        print(f"  New range: {min(self.thrust):.2f} to {max(self.thrust):.2f} N")
        
    def _recalculate_moi(self):
        """Recalculate MOI for all time points (used after extension)"""
        pv_radius = self.config.pressure_vessel_diameter / 2
        fg_outer_radius = self.config.fuel_grain_outer_diameter / 2
        fg_inner_radius = self.config.fuel_grain_inner_diameter / 2
        
        self.oxidizer_moi = []
        self.fuel_moi = []
        
        for i, time in enumerate(self.times):
            if i == 0:
                ox_mass_consumed = 0
                fuel_mass_consumed = 0
            else:
                time_slice = self.times[:i+1]
                ox_flow_slice = self.oxidizer_flow_rates[:i+1]
                fuel_flow_slice = self.fuel_flow_rates[:i+1]
                
                ox_mass_consumed = np.trapz(ox_flow_slice, time_slice)
                fuel_mass_consumed = np.trapz(fuel_flow_slice, time_slice)
            
            ox_mass_remaining = self.total_ox_mass - ox_mass_consumed
            fuel_mass_remaining = self.total_fuel_mass - fuel_mass_consumed
            
            ox_moi = MomentOfInertia.solid_cylinder(
                ox_mass_remaining,
                self.config.pressure_vessel_length,
                pv_radius,
                self.config.zero_moi
            )
            self.oxidizer_moi.append(ox_moi)
            
            fuel_moi = MomentOfInertia.hollow_cylinder(
                fuel_mass_remaining,
                self.config.fuel_grain_length,
                fg_inner_radius,
                fg_outer_radius,
                self.config.zero_moi
            )
            self.fuel_moi.append(fuel_moi)
    
    def plot_results(self):
        print("printing")
        # Check if using actual flow rates for enhanced plotting
        if self.config.use_actual_flow_rates:
            fig, axes = plt.subplots(3, 1, figsize=(14, 16))
        else:
            fig = plt.figure(figsize=(12, 6))
            axes = [plt.gca(), None, None]
        
        # ========== PANEL 1: THRUST CURVE ==========
        ax1 = axes[0]
        
        # Check if data was modified (extended or scaled)
        data_modified = (len(self.times) != len(self.original_times) or 
                        not np.allclose(self.thrust, self.original_thrust))
        
        if data_modified:
            # Plot both original and modified curves for comparison
            ax1.plot(self.original_times, self.original_thrust, 'b-', 
                    linewidth=2, label='Original', alpha=0.7)
            ax1.plot(self.times, self.thrust, 'darkorange', 
                    linewidth=2, label='Modified')
            ax1.legend(loc='upper right', fontsize=10)
        else:
            # Just plot the thrust curve
            ax1.plot(self.times, self.thrust, 'b-', linewidth=2)
        
        ax1.set_xlabel('Time (s)', fontsize=12)
        ax1.set_ylabel('Thrust (N)', fontsize=12)
        ax1.set_title('Motor Thrust Curve', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add statistics to plot
        avg_thrust = np.mean(self.thrust)
        max_thrust = max(self.thrust)
        total_impulse = np.trapz(self.thrust, self.times)
        
        stats_text = f'Average Thrust: {avg_thrust:.1f} N\n'
        stats_text += f'Max Thrust: {max_thrust:.1f} N\n'
        stats_text += f'Total Impulse: {total_impulse:.1f} N·s'
        
        ax1.text(0.98, 0.97, stats_text,
                transform=ax1.transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=10)
        
        if self.config.use_actual_flow_rates and axes[1] is not None:
            ax2 = axes[1]
            
            ax2.plot(self.times, self.oxidizer_flow_rates, 'b-', 
                    linewidth=2, label='Oxidizer Flow')
            ax2.plot(self.times, self.fuel_flow_rates, 'r-', 
                    linewidth=2, label='Fuel Flow')
            
            # Add average reference lines
            ax2.axhline(y=self.oxidizer_flowrate, color='b', linestyle='--', 
                       linewidth=1, alpha=0.5, 
                       label=f'Ox Avg: {self.oxidizer_flowrate:.4f} kg/s')
            ax2.axhline(y=self.fuel_flowrate, color='r', linestyle='--', 
                       linewidth=1, alpha=0.5, 
                       label=f'Fuel Avg: {self.fuel_flowrate:.4f} kg/s')
            
            ax2.set_xlabel('Time (s)', fontsize=12)
            ax2.set_ylabel('Flow Rate (kg/s)', fontsize=12)
            ax2.set_title('Mass Flow Rates (From Load Cell Data)', 
                         fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.legend(fontsize=9, loc='best')
            
            # Add statistics box
            flow_stats = (
                f'Oxidizer:\n'
                f'  Min: {min(self.oxidizer_flow_rates):.4f} kg/s\n'
                f'  Max: {max(self.oxidizer_flow_rates):.4f} kg/s\n'
                f'  Avg: {self.oxidizer_flowrate:.4f} kg/s\n'
                f'Fuel:\n'
                f'  Min: {min(self.fuel_flow_rates):.4f} kg/s\n'
                f'  Max: {max(self.fuel_flow_rates):.4f} kg/s\n'
                f'  Avg: {self.fuel_flowrate:.4f} kg/s'
            )
            ax2.text(0.02, 0.98, flow_stats, transform=ax2.transAxes,
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                    fontsize=9, verticalalignment='top', fontfamily='monospace')
        
        if self.config.use_actual_flow_rates and axes[2] is not None and len(self.load_cell_masses) > 0:
            ax3 = axes[2]
            
            # Plot load cell data (only original burn, not extended)
            original_length = len(self.load_cell_masses)
            ax3.plot(self.times[:original_length], self.load_cell_masses, 'g-', 
                    linewidth=2, label='Measured Load Cell Mass')
            
            ax3.set_xlabel('Time (s)', fontsize=12)
            ax3.set_ylabel('Oxidizer Tank Mass (kg)', fontsize=12)
            ax3.set_title('Oxidizer Tank Load Cell Measurement', 
                         fontsize=14, fontweight='bold')
            ax3.grid(True, alpha=0.3)
            ax3.legend(fontsize=10)
            
            # Add mass consumed annotation
            mass_consumed = self.load_cell_masses[0] - self.load_cell_masses[-1]
            ax3.text(0.02, 0.02, 
                    f'Mass consumed: {abs(mass_consumed):.4f} kg\n'
                    f'Initial mass: {self.load_cell_masses[0]:.4f} kg\n'
                    f'Final mass: {self.load_cell_masses[-1]:.4f} kg',
                    transform=ax3.transAxes,
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
                    fontsize=10, verticalalignment='bottom')
        
        plt.tight_layout()
        
        # Save with appropriate filename
        if self.config.use_actual_flow_rates and self.config.save_plots:
            plt.savefig('motor_analysis_with_actual_flow.png', dpi=150, bbox_inches='tight')
            print("\nSaved enhanced plot: motor_analysis_with_actual_flow.png")
        else:
            plt.savefig('motor_analysis.png', dpi=150, bbox_inches='tight')
            print("\nSaved plot: motor_analysis.png")
        
        plt.show()
        
    def write_motor_file(self):
        """
        Write MapleLeaf-compatible motor definition file
        
        Format includes:
        - Header with CG locations
        - Tabulated data: Time, Thrust, Flow rates, MOI values
        """
        if not self.config.write_output:
            print("\nSkipping file output (write_output = False)")
            return
        
        print(f"\nWriting motor file to: {self.config.output_file}")
        
        # Generate header
        header = self._generate_header()
        
        # Write file
        with open(self.config.output_file, 'w') as f:
            f.write(header)
            
            for i in range(len(self.times)):
                line = (f"{self.times[i]}\t"
                       f"{self.thrust[i]}\t"
                       f"{self.oxidizer_flow_rates[i]}\t"
                       f"{self.fuel_flow_rates[i]}\t"
                       f"({self.oxidizer_moi[i][0]} {self.oxidizer_moi[i][1]} {self.oxidizer_moi[i][2]})\t"
                       f"({self.fuel_moi[i][0]} {self.fuel_moi[i][1]} {self.fuel_moi[i][2]})\n")
                f.write(line)
        
        print(f"  Wrote {len(self.times)} data points")
        print("  Output file created successfully")
        
    def _generate_header(self):
        """Generate MapleLeaf motor file header"""
        header = f"""# Motor Definition File: {self.config.name}
# Written by Aaron Arellano

# Thrust curve will be linearly interpolated between the given values
# Time(s) should be relative to motor ignition

# Unburned fuel/oxidizer not currently accounted for, could be added in as a fixed mass?
# Engine/Tank structure mass should be accounted for separately - the motor only accounts for propellant masses
# MOI = Moment of Inertia
    # It is assumed that oxidizer and fuel MOIs are always defined about the current CG of the oxidizer and fuel, respectively
# To represent a solid rocket motor, simply put all of the MOI/CG/Flowrate info in either the fuel or oxidizer columns and set the other one to zero

# Meters, all Z-coordinate, assumed centered
BurnedOxidizerInitialCG     {self.config.burnedOxidizerInitialCG}
BurnedOxidizerLastCG        {self.config.burnedOxidizerLastCG}
BurnedFuelInitialCG         {self.config.burnedFuelInitialCG}
BurnedFuelLastCG            {self.config.burnedFuelLastCG}

#Time(s)   Thrust(N)   OxidizerFlowRate(kg/s)  FuelBurnRate(kg/s) OxMOI(kg*m^2) FuelMOI(kg*m^2) 
"""
        return header
    
    def process(self):
        """
        Main processing pipeline
        
        Execute all processing steps in order:
        1. Load raw data
        2. Find motor burn window
        3. Calculate flow rates
        4. Extract burn data
        5. Apply modifications (extension, scaling)
        6. Plot results
        7. Write output file
        """
        print("="*70)
        print("ROCKET MOTOR DATA PROCESSOR")
        print("="*70)
        
        self.load_data()
        self.find_motor_burn_window()
        self.calculate_flow_rates()
        self.extract_burn_data()
        self.extend_burn_time()
        self.apply_thrust_estimation()
        self.plot_results()
        self.write_motor_file()
        
        print("\n" + "="*70)
        print("PROCESSING COMPLETE")
        print("="*70)

def configure_motor_from_masses(
    target_oxidizer_mass: float,
    target_fuel_mass: float,
    avg_thrust: float,
    burn_time: float = None,
    of_ratio: float = 6.58,
    isp: float = 230.0
) -> dict:
    """
    Calculate motor parameters needed to achieve specific fuel and oxidizer masses.
    
    This function works backwards from desired propellant masses to determine
    the required thrust profile, burn time, and flow rates.
    
    Args:
        target_oxidizer_mass: Desired oxidizer mass (kg)
        target_fuel_mass: Desired fuel mass (kg)
        avg_thrust: Target average thrust (N)
        burn_time: Desired burn time (s). If None, calculated from Isp
        of_ratio: Oxidizer-to-fuel ratio
        isp: Specific impulse (s) - used if burn_time not specified
        
    Returns:
        Dictionary with motor parameters:
        {
            'burn_time': float,           # Burn duration (s)
            'avg_thrust': float,          # Average thrust (N)
            'oxidizer_flowrate': float,   # Oxidizer mass flow (kg/s)
            'fuel_flowrate': float,       # Fuel mass flow (kg/s)
            'total_mass_flow': float,     # Total propellant flow (kg/s)
            'total_impulse': float,       # Total impulse (N·s)
            'of_ratio': float,            # O/F ratio
            'isp_effective': float        # Effective Isp (s)
        }
    
    Example:
        >>> params = configure_motor_from_masses(
        ...     target_oxidizer_mass=5.0,  # 5 kg oxidizer
        ...     target_fuel_mass=0.76,     # 0.76 kg fuel
        ...     avg_thrust=536,            # 536 N average
        ...     burn_time=3.0              # 3 second burn
        ... )
        >>> print(f"Oxidizer flow: {params['oxidizer_flowrate']:.3f} kg/s")
        >>> print(f"Fuel flow: {params['fuel_flowrate']:.3f} kg/s")
    """
    total_propellant_mass = target_oxidizer_mass + target_fuel_mass
    
    # Calculate burn time if not provided
    if burn_time is None:
        # Use rocket equation: F = (dm/dt) * Isp * g0
        g0 = 9.80665  # Standard gravity (m/s^2)
        total_mass_flow = avg_thrust / (isp * g0)
        burn_time = total_propellant_mass / total_mass_flow
    else:
        total_mass_flow = total_propellant_mass / burn_time
    
    # Calculate flow rates
    oxidizer_flowrate = target_oxidizer_mass / burn_time
    fuel_flowrate = target_fuel_mass / burn_time
    
    # Calculate actual O/F ratio
    actual_of_ratio = oxidizer_flowrate / fuel_flowrate if fuel_flowrate > 0 else of_ratio
    
    # Calculate total impulse
    total_impulse = avg_thrust * burn_time
    
    # Calculate effective Isp
    g0 = 9.80665
    isp_effective = avg_thrust / (total_mass_flow * g0)
    
    results = {
        'burn_time': burn_time,
        'avg_thrust': avg_thrust,
        'oxidizer_flowrate': oxidizer_flowrate,
        'fuel_flowrate': fuel_flowrate,
        'total_mass_flow': total_mass_flow,
        'total_impulse': total_impulse,
        'of_ratio': actual_of_ratio,
        'isp_effective': isp_effective,
        'target_oxidizer_mass': target_oxidizer_mass,
        'target_fuel_mass': target_fuel_mass,
        'total_propellant_mass': total_propellant_mass
    }
    
    return results


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    """Main execution function"""
    
    # Create configuration
    config = MotorConfig()
    config.use_actual_flow_rates = True
    config.flow_smoothing_window = 20
    
    # Create processor and run
    processor = MotorDataProcessor(config)
    processor.process()


import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile


class TestMomentOfInertia(unittest.TestCase):
    """Test MOI calculations for solid and hollow cylinders"""
    
    def test_solid_cylinder_moi(self):
        """Test solid cylinder MOI formulas against hand calculations"""
        mass, length, radius = 1.0, 0.5, 0.1
        moi = MomentOfInertia.solid_cylinder(mass, length, radius, zeroed=False)
        
        expected_moi_xy = (0.25 * mass * radius**2) + ((1/12) * mass * length**2)

        #expected_moi_xy = 0
        expected_moi_z = 0.5 * mass * radius**2
        
        self.assertAlmostEqual(moi[0], expected_moi_xy, places=6)
        self.assertAlmostEqual(moi[1], expected_moi_xy, places=6)
        self.assertAlmostEqual(moi[2], expected_moi_z, places=6)
    
    def test_hollow_cylinder_moi(self):
        """Test hollow cylinder MOI formulas"""
        mass, length, inner_r, outer_r = 0.5, 0.3, 0.05, 0.075
        moi = MomentOfInertia.hollow_cylinder(mass, length, inner_r, outer_r, zeroed=False)
        
        expected_moi_xy = (mass/12) * (3 * (inner_r**2 + outer_r**2) + length**2)
        expected_moi_z = (mass/2) * (inner_r**2 + outer_r**2)
        
        self.assertAlmostEqual(moi[0], expected_moi_xy, places=6)
        self.assertAlmostEqual(moi[2], expected_moi_z, places=6)
    
    def test_zeroed_mode(self):
        """Test that zeroed mode returns [0, 0, 0]"""
        moi1 = MomentOfInertia.solid_cylinder(1.0, 0.5, 0.1, zeroed=True)
        moi2 = MomentOfInertia.hollow_cylinder(0.5, 0.3, 0.05, 0.075, zeroed=True)
        self.assertEqual(moi1, [0, 0, 0])
        self.assertEqual(moi2, [0, 0, 0])


class TestDynamicFlowRateCalculator(unittest.TestCase):
    """Test actual flow rate calculation from load cell data"""
    
    def test_flow_rate_from_decreasing_mass(self):
        """Test flow rate calculation from linearly decreasing mass"""
        times = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        masses = [10.0, 9.0, 8.0, 7.0, 6.0, 5.0]  # Linear decrease of 1 kg/s
        of_ratio = 6.0
        
        ox_flow, fuel_flow = DynamicFlowRateCalculator.calculate_from_load_cell(
            times, masses, of_ratio, smoothing_window=1
        )
        
        for rate in ox_flow:
            self.assertAlmostEqual(rate, 1.0, delta=0.1)
        for rate in fuel_flow:
            self.assertAlmostEqual(rate, 1.0/6.0, delta=0.02)
    
    def test_mass_conservation(self):
        """Test that integrated flow rates equal total mass consumed"""
        times = np.linspace(0, 3.0, 100).tolist()
        initial_mass, final_mass = 10.0, 7.5
        masses = [initial_mass - (initial_mass - final_mass) * (t/3.0) for t in times]
        
        ox_flow, _ = DynamicFlowRateCalculator.calculate_from_load_cell(
            times, masses, 6.58, smoothing_window=5
        )
        
        integrated_ox_mass = np.trapz(ox_flow, times)
        expected_ox_mass = initial_mass - final_mass
        error_percent = abs(integrated_ox_mass - expected_ox_mass) / expected_ox_mass * 100
        
        self.assertLess(error_percent, 5.0, 
                       f"Mass conservation error {error_percent:.2f}% exceeds 5%")
    
    def test_smoothing_reduces_noise(self):
        """Test that smoothing reduces variation in flow rates"""
        times = np.linspace(0, 2.0, 50).tolist()
        base_masses = [10.0 - 2.5*t for t in times]
        noisy_masses = [m + 0.1*np.random.randn() for m in base_masses]
        
        ox_flow_noisy, _ = DynamicFlowRateCalculator.calculate_from_load_cell(
            times, noisy_masses, 6.0, smoothing_window=1
        )
        ox_flow_smooth, _ = DynamicFlowRateCalculator.calculate_from_load_cell(
            times, noisy_masses, 6.0, smoothing_window=11
        )
        
        self.assertLess(np.std(ox_flow_smooth), np.std(ox_flow_noisy))
    
    def test_fuel_flow_from_of_ratio(self):
        """Test that fuel flow is correctly calculated from O/F ratio"""
        times = [0.0, 1.0, 2.0]
        masses = [10.0, 9.0, 8.0]
        of_ratio = 5.0
        
        ox_flow, fuel_flow = DynamicFlowRateCalculator.calculate_from_load_cell(
            times, masses, of_ratio, smoothing_window=1
        )
        
        for ox, fuel in zip(ox_flow, fuel_flow):
            self.assertAlmostEqual(fuel, ox / of_ratio, places=6)


class TestMotorConfig(unittest.TestCase):
    """Test motor configuration class"""
    
    def test_default_initialization(self):
        """Test that default configuration values are set correctly"""
        config = MotorConfig()
        
        self.assertTrue(config.extend_burn)
        self.assertEqual(config.extension_duration, 1.3)
        self.assertTrue(config.use_actual_flow_rates)
        self.assertEqual(config.flow_smoothing_window, 20)
        self.assertEqual(config.of_ratio, 6.58)
    
    def test_configuration_modification(self):
        """Test that configuration can be modified"""
        config = MotorConfig()
        config.use_actual_flow_rates = False
        config.flow_smoothing_window = 30
        
        self.assertFalse(config.use_actual_flow_rates)
        self.assertEqual(config.flow_smoothing_window, 30)


class TestMotorDataProcessor(unittest.TestCase):
    """Test motor data processor functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = MotorConfig()
        self.config.write_output = False
        self.processor = MotorDataProcessor(self.config)
    
    def test_processor_initialization(self):
        """Test that processor initializes correctly"""
        self.assertIsNotNone(self.processor.config)
        self.assertEqual(len(self.processor.times), 0)
        self.assertEqual(len(self.processor.oxidizer_flow_rates), 0)
        self.assertEqual(self.processor.oxidizer_flowrate, 0)
    
    def test_burn_window_detection(self):
        """Test motor burn window detection from pressure data"""
        mock_data = pd.DataFrame({
            'Chamber Pressure (PSI)': [5, 5, 20, 25, 30, 25, 20, 5, 5],
            'Comment': ['', '', '', '', '', '', '', 'Standby', ''],
            'Time (s)': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            'Ox Tank Load Cell (kg)': [10]*9,
            'Thrust Load Cell (lbf)': [30]*9
        })
        
        self.processor.raw_data = mock_data
        self.processor.find_motor_burn_window()
        
        self.assertEqual(self.processor.motor_start_idx, 2)
        self.assertEqual(self.processor.motor_end_idx, 7)
    
    def test_flow_rate_calculation(self):
        """Test average flow rate calculation"""
        mock_data = pd.DataFrame({
            'Time (s)': [0.0, 1.0, 2.0],
            'Ox Tank Load Cell (kg)': [10.0, 8.0, 6.0],
            'Chamber Pressure (PSI)': [20, 20, 20],
            'Comment': ['', '', 'Standby']
        })
        
        self.processor.raw_data = mock_data
        self.processor.motor_start_idx = 0
        self.processor.motor_end_idx = 2
        self.processor.initial_time = 0.0
        
        self.processor.calculate_flow_rates()
        
        self.assertAlmostEqual(self.processor.oxidizer_flowrate, 2.0, places=1)
    
    def test_thrust_scaling(self):
        """Test thrust scaling to target average"""
        self.processor.thrust = [1000, 1500, 2000, 1500, 1000]
        self.processor.config.use_estimation = True
        self.processor.config.estimated_avg_thrust = 3000
        
        self.processor.apply_thrust_estimation()
        new_avg = np.mean(self.processor.thrust)
        
        self.assertAlmostEqual(new_avg, 3000, places=0)
    
    def test_moi_recalculation(self):
        """Test MOI recalculation with flow rates"""
        self.processor.times = [0.0, 1.0, 2.0]
        self.processor.oxidizer_flow_rates = [1.0, 1.0, 1.0]
        self.processor.fuel_flow_rates = [0.15, 0.15, 0.15]
        self.processor.total_ox_mass = 2.0
        self.processor.total_fuel_mass = 0.3
        
        self.processor._recalculate_moi()
        
        self.assertEqual(len(self.processor.oxidizer_moi), 3)
        self.assertEqual(len(self.processor.fuel_moi), 3)


class TestConfigureMotorFromMasses(unittest.TestCase):
    """Test the configure_motor_from_masses utility function"""
    
    def test_basic_calculation(self):
        """Test basic motor configuration from masses"""
        result = configure_motor_from_masses(
            target_oxidizer_mass=5.0,
            target_fuel_mass=0.76,
            avg_thrust=536,
            burn_time=3.0
        )
        
        self.assertAlmostEqual(result['burn_time'], 3.0)
        self.assertAlmostEqual(result['oxidizer_flowrate'], 5.0/3.0, places=2)
        self.assertAlmostEqual(result['total_impulse'], 536 * 3.0)


def run_tests():
    """Run all unit tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestMomentOfInertia))
    suite.addTests(loader.loadTestsFromTestCase(TestDynamicFlowRateCalculator))
    suite.addTests(loader.loadTestsFromTestCase(TestMotorConfig))
    suite.addTests(loader.loadTestsFromTestCase(TestMotorDataProcessor))
    suite.addTests(loader.loadTestsFromTestCase(TestConfigureMotorFromMasses))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*70)
    
    return result.wasSuccessful()


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    import sys
    
    # Check if running in test mode
    if TEST:
        print("="*70)
        print("RUNNING UNIT TESTS")
        print("="*70)
        success = run_tests()
        sys.exit(0 if success else 1)
    else:
        # Normal execution
        main()