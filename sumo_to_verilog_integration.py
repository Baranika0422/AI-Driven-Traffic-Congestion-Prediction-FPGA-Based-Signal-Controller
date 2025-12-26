"""
DIRECTION-AWARE SUMO TO VERILOG INTEGRATION - FIXED
Emergency vehicle detection now properly initialized
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Add SUMO tools to path
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

import traci
import sumolib

class EmergencyVehicleDetector:
    """Detects emergency vehicles from SUMO"""
    
    def __init__(self, junction_edges):  # FIXED: Now properly using double underscores
        self.junction_edges = junction_edges
        self.emergency_types = ['ambulance', 'emergency', 'police', 'fire', 'firebrigade']
        print("  ✓ EmergencyVehicleDetector initialized")
        
    def get_edge_direction(self, edge_id):
        """Map edge ID to direction (N/S/E/W)"""
        for direction, edges in self.junction_edges.items():
            if any(e in edge_id.lower() for e in edges):
                return direction
        return None
        
    def encode_direction(self, direction):
        """Encode direction to 2-bit value"""
        direction_map = {
            'north': 0b00,
            'south': 0b01, 
            'east': 0b10,
            'west': 0b11
        }
        return direction_map.get(direction.lower(), 0b00)
    
    def detect_emergency_vehicles(self):
        """Scan for emergency vehicles and return info"""
        emergency_info = {
            'detected': 0,
            'direction': 0,
            'direction_name': 'none',
            'priority': 0,
            'distance': 999,
            'vehicle_id': '',
            'type': ''
        }
        
        try:
            all_vehicles = traci.vehicle.getIDList()
            
            for vid in all_vehicles:
                vtype = traci.vehicle.getTypeID(vid).lower()
                
                # Check if emergency vehicle
                is_emergency = any(keyword in vtype for keyword in self.emergency_types)
                
                if is_emergency:
                    road_id = traci.vehicle.getRoadID(vid)
                    direction = self.get_edge_direction(road_id)
                    
                    if direction is None:
                        continue
                    
                    try:
                        distance = traci.vehicle.getLanePosition(vid)
                        
                        # If this is closer than previous emergency
                        if distance < emergency_info['distance']:
                            # Determine priority
                            priority = 1 if 'ambulance' in vtype else 0
                            
                            emergency_info = {
                                'detected': 1,
                                'direction': self.encode_direction(direction),
                                'direction_name': direction,
                                'priority': priority,
                                'distance': distance,
                                'vehicle_id': vid,
                                'type': vtype
                            }
                    except:
                        continue
                        
        except Exception as e:
            print(f"Error detecting emergency: {e}")
        
        return emergency_info


class DirectionalSUMOToVerilog:
    """Extract SUMO data with directional predictions"""
    
    def __init__(self, sumo_config_file):  # FIXED: Now properly using double underscores
        self.sumo_config = sumo_config_file
        self.traffic_data = []
        self.junction_id = "J1"
        
        # Define junction edges - CUSTOMIZE FOR YOUR NETWORK
        self.junction_edges = {
            'north': ['n', 'north', '_n'],
            'south': ['s', 'south', '_s'],
            'east': ['e', 'east', '_e'],
            'west': ['w', 'west', '_w']
        }
        
        # Initialize emergency detector
        print("\n[INITIALIZATION] Setting up components...")
        self.emergency_detector = EmergencyVehicleDetector(self.junction_edges)
        print("  ✓ DirectionalSUMOToVerilog initialized")
    
    def run_sumo_simulation(self, gui=False, steps=1000):
        """Run SUMO simulation and collect directional traffic data"""
        
        print("\n" + "="*80)
        print("DIRECTIONAL SUMO SIMULATION - TRAFFIC DATA COLLECTION")
        print("="*80)
        print(f"Config file: {self.sumo_config}")
        print(f"Simulation steps: {steps}")
        print(f"GUI mode: {gui}")
        print("="*80 + "\n")
        
        # Start SUMO
        if gui:
            sumo_binary = sumolib.checkBinary('sumo-gui')
        else:
            sumo_binary = sumolib.checkBinary('sumo')
        
        sumo_cmd = [sumo_binary, "-c", self.sumo_config, "--no-warnings", "true"]
        
        traci.start(sumo_cmd)
        
        step = 0
        emergency_count = 0
        
        try:
            while step < steps and traci.simulation.getMinExpectedNumber() > 0:
                traci.simulationStep()
                
                # Collect data every step
                data = self.collect_directional_metrics(step)
                if data:
                    self.traffic_data.append(data)
                    
                    # Log emergency detections
                    if data['emergency_detected']:
                        emergency_count += 1
                        if emergency_count <= 10:
                            print(f"  🚨 Emergency at step {step}: "
                                  f"{data['emergency_type']} from {data['emergency_direction_name']}")
                
                # Progress indicator
                if step % 100 == 0:
                    print(f"  Step: {step}/{steps} ({step*100//steps}%) | Emergencies: {emergency_count}")
                
                step += 1
                
        except traci.exceptions.FatalTraCIError as e:
            print(f"\n⚠ SUMO simulation ended: {e}")
        
        finally:
            traci.close()
        
        print(f"\n✓ Simulation complete!")
        print(f"✓ Collected {len(self.traffic_data)} data points")
        print(f"✓ Detected {emergency_count} emergency vehicle instances")
        
        return self.traffic_data
    
    def collect_directional_metrics(self, step):
        """Collect traffic metrics organized by direction"""
        
        try:
            # Get all vehicles
            vehicle_ids = traci.vehicle.getIDList()
            
            if len(vehicle_ids) == 0:
                return None
            
            # Initialize directional metrics
            metrics = {
                'step': step,
                'time_seconds': step,
                'total_vehicles': len(vehicle_ids),
                
                # Per-direction queues
                'north_queue': 0,
                'south_queue': 0,
                'east_queue': 0,
                'west_queue': 0,
                
                # Per-direction speeds
                'north_speed': 0,
                'south_speed': 0,
                'east_speed': 0,
                'west_speed': 0,
                
                # Per-direction waiting times
                'north_wait': 0,
                'south_wait': 0,
                'east_wait': 0,
                'west_wait': 0,
                
                # Per-direction vehicle counts
                'north_vehicles': 0,
                'south_vehicles': 0,
                'east_vehicles': 0,
                'west_vehicles': 0,
                
                # Overall metrics
                'avg_speed_kmh': 0,
                'avg_waiting_time': 0,
                'max_waiting_time': 0,
                'halting_count': 0,
                
                # Vehicle types
                'cars': 0,
                'trucks': 0,
                'buses': 0,
                'motorcycles': 0,
                'emergency_vehicles': 0,
                
                # Emergency fields
                'emergency_detected': 0,
                'emergency_direction': 0,
                'emergency_direction_name': 'none',
                'emergency_priority': 0,
                'emergency_distance': 999,
                'emergency_vehicle_id': '',
                'emergency_type': ''
            }
            
            # Directional data collectors
            dir_speeds = {'north': [], 'south': [], 'east': [], 'west': []}
            dir_waits = {'north': [], 'south': [], 'east': [], 'west': []}
            
            speeds = []
            waiting_times = []
            
            # Collect vehicle data
            for vid in vehicle_ids:
                vtype = traci.vehicle.getTypeID(vid).lower()
                
                # Vehicle type classification
                if 'car' in vtype or 'passenger' in vtype:
                    metrics['cars'] += 1
                elif 'truck' in vtype:
                    metrics['trucks'] += 1
                elif 'bus' in vtype:
                    metrics['buses'] += 1
                elif 'motorcycle' in vtype or 'bike' in vtype:
                    metrics['motorcycles'] += 1
                elif any(e in vtype for e in ['emergency', 'ambulance', 'police', 'fire']):
                    metrics['emergency_vehicles'] += 1
                else:
                    metrics['cars'] += 1
                
                # Speed
                speed = traci.vehicle.getSpeed(vid)
                speed_kmh = speed * 3.6
                speeds.append(speed_kmh)
                
                # Waiting time
                wait_time = traci.vehicle.getWaitingTime(vid)
                waiting_times.append(wait_time)
                
                # Halting
                if speed < 0.1:
                    metrics['halting_count'] += 1
                
                # Determine direction and update directional metrics
                lane_id = traci.vehicle.getLaneID(vid)
                direction = None
                
                for dir_name in ['north', 'south', 'east', 'west']:
                    if any(e in lane_id.lower() for e in self.junction_edges[dir_name]):
                        direction = dir_name
                        break
                
                if direction:
                    metrics[f'{direction}_queue'] += 1
                    metrics[f'{direction}_vehicles'] += 1
                    dir_speeds[direction].append(speed_kmh)
                    dir_waits[direction].append(wait_time)
            
            # Calculate overall averages
            if speeds:
                metrics['avg_speed_kmh'] = np.mean(speeds)
            
            if waiting_times:
                metrics['avg_waiting_time'] = np.mean(waiting_times)
                metrics['max_waiting_time'] = np.max(waiting_times)
            
            # Calculate directional averages
            for direction in ['north', 'south', 'east', 'west']:
                if dir_speeds[direction]:
                    metrics[f'{direction}_speed'] = np.mean(dir_speeds[direction])
                if dir_waits[direction]:
                    metrics[f'{direction}_wait'] = np.mean(dir_waits[direction])
            
            # Detect emergency vehicles - NOW THIS WILL WORK!
            emergency_info = self.emergency_detector.detect_emergency_vehicles()
            metrics['emergency_detected'] = emergency_info['detected']
            metrics['emergency_direction'] = emergency_info['direction']
            metrics['emergency_direction_name'] = emergency_info['direction_name']
            metrics['emergency_priority'] = emergency_info['priority']
            metrics['emergency_distance'] = emergency_info['distance']
            metrics['emergency_vehicle_id'] = emergency_info['vehicle_id']
            metrics['emergency_type'] = emergency_info['type']
            
            return metrics
            
        except Exception as e:
            print(f"Error collecting metrics at step {step}: {e}")
            return None
    
    def calculate_directional_congestion(self, df):
        """Calculate congestion per direction using rules"""
        print("\n[DIRECTIONAL PREDICTION] Calculating congestion per direction...")
        print("✓ Using rule-based congestion calculation")
        
        directions = ['north', 'south', 'east', 'west']
        
        for direction in directions:
            congestion_levels = []
            
            for _, row in df.iterrows():
                speed = row[f'{direction}_speed'] if row[f'{direction}_speed'] > 0 else 30
                congestion = self.rule_based_congestion(
                    row[f'{direction}_queue'],
                    speed,
                    row[f'{direction}_wait']
                )
                congestion_levels.append(congestion)
            
            df[f'{direction}_congestion'] = congestion_levels
        
        return df
    
    def rule_based_congestion(self, queue, speed, wait_time):
        """Calculate congestion using rules"""
        speed_factor = max(0, (60 - speed) / 60 * 40)
        waiting_factor = min(wait_time / 120 * 30, 30)
        density_factor = min(queue / 50 * 30, 30)
        
        congestion = speed_factor + waiting_factor + density_factor
        return int(min(100, max(0, congestion)))
    
    def congestion_to_level(self, congestion):
        """Convert congestion (0-100) to level (0-3)"""
        if congestion < 30:
            return 0  # LOW
        elif congestion < 60:
            return 1  # MEDIUM
        elif congestion < 80:
            return 2  # HIGH
        else:
            return 3  # CRITICAL
    
    def save_to_csv(self):
        """Save directional traffic data to CSV"""
        
        print("\n[SAVING] Creating directional CSV files...")
        
        # Convert to DataFrame
        df = pd.DataFrame(self.traffic_data)
        
        # Calculate directional congestion
        df = self.calculate_directional_congestion(df)
        
        # Save FULL dataset
        full_csv = 'directional_traffic_data.csv'
        df.to_csv(full_csv, index=False)
        print(f"✓ Saved full dataset: {full_csv} ({len(df)} rows)")
        
        # Save VERILOG testbench CSV
        verilog_df = pd.DataFrame({
            'index': range(len(df)),
            'north_congestion': df['north_congestion'].astype(int),
            'south_congestion': df['south_congestion'].astype(int),
            'east_congestion': df['east_congestion'].astype(int),
            'west_congestion': df['west_congestion'].astype(int),
            'north_level': df['north_congestion'].apply(self.congestion_to_level).astype(int),
            'south_level': df['south_congestion'].apply(self.congestion_to_level).astype(int),
            'east_level': df['east_congestion'].apply(self.congestion_to_level).astype(int),
            'west_level': df['west_congestion'].apply(self.congestion_to_level).astype(int),
            'emergency_detected': df['emergency_detected'].astype(int),
            'emergency_direction': df['emergency_direction'].astype(int),
            'north_queue': df['north_queue'].astype(int),
            'south_queue': df['south_queue'].astype(int),
            'east_queue': df['east_queue'].astype(int),
            'west_queue': df['west_queue'].astype(int)
        })
        
        verilog_csv = 'directional_verilog_testbench.csv'
        verilog_df.to_csv(verilog_csv, index=False)
        print(f"✓ Saved Verilog CSV: {verilog_csv} ({len(verilog_df)} rows)")
        
        # Statistics
        print("\n[STATISTICS]")
        print(f"  Total samples: {len(df)}")
        print(f"  Time range: {df['time_seconds'].min():.1f}s - {df['time_seconds'].max():.1f}s")
        print(f"  Emergency detections: {df['emergency_detected'].sum()}")
        
        print("\n  Average Congestion by Direction:")
        for direction in ['north', 'south', 'east', 'west']:
            avg_cong = df[f'{direction}_congestion'].mean()
            print(f"    {direction.capitalize():6s}: {avg_cong:.1f}")
        
        # Emergency statistics
        if df['emergency_detected'].sum() > 0:
            print("\n  Emergency Distribution:")
            for direction in ['north', 'south', 'east', 'west']:
                count = df[df['emergency_direction_name'] == direction].shape[0]
                if count > 0:
                    print(f"    {direction.capitalize():6s}: {count} detections")
        
        return df


def main():
    """Main execution"""
    
    print("\n" + "="*80)
    print("DIRECTION-AWARE SUMO TO VERILOG INTEGRATION - FIXED")
    print("="*80)
    
    # Configuration
    SUMO_CONFIG = "simulation.sumocfg"
    GUI_MODE = False
    SIMULATION_STEPS = 1000
    
    # Check if config exists
    if not os.path.exists(SUMO_CONFIG):
        print(f"\n✗ ERROR: SUMO config file not found: {SUMO_CONFIG}")
        print("\nPlease update SUMO_CONFIG with your actual config file path.")
        return
    
    # Create integrator
    integrator = DirectionalSUMOToVerilog(SUMO_CONFIG)
    
    # Run SUMO simulation
    integrator.run_sumo_simulation(gui=GUI_MODE, steps=SIMULATION_STEPS)
    
    # Save data
    df = integrator.save_to_csv()
    
    print("\n" + "="*80)
    print("✓ DIRECTIONAL INTEGRATION COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  1. directional_traffic_data.csv    - Full directional data with emergencies")
    print("  2. directional_verilog_testbench.csv - For Verilog testbench")
    print("\nNext steps:")
    print("  1. Run directional Verilog testbench")
    print("  2. Emergency vehicles will trigger override")
    print("  3. View waveforms with emergency handling")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()