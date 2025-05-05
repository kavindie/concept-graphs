import numpy as np
import math
from pyproj import Transformer
import os

def latlon_to_mercator(lat, lon, scale=1.0):
    """
    Convert latitude/longitude to mercator coordinates.
    """
    er = 6378137.0  # earth radius in meters
    mx = scale * lon * np.pi * er / 180.0
    my = scale * er * np.log(np.tan((90.0 + lat) * np.pi / 360.0))
    return mx, my

def rotx(t):
    """Rotation about X-axis"""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1, 0, 0], 
                    [0, c, -s], 
                    [0, s, c]])

def roty(t):
    """Rotation about Y-axis"""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s], 
                    [0, 1, 0], 
                    [-s, 0, c]])

def rotz(t):
    """Rotation about Z-axis"""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s, 0], 
                    [s, c, 0], 
                    [0, 0, 1]])

def convert_gps_to_transform_matrices(gps_data, output_file):
    """
    Convert GPS data to 4x4 transformation matrices.
    
    Args:
        gps_data: List of dictionaries containing GPS data for each timestamp
        output_file: Path to save the transformation matrices
    """
    # Initialize origin with the first GPS reading
    if not gps_data:
        raise ValueError("GPS data is empty")
    
    # Use the first point as reference (local origin)
    lat0 = gps_data[0]['lat']
    lon0 = gps_data[0]['lon']
    alt0 = gps_data[0]['alt']
    
    # Create transformer for more accurate conversion
    transformer = Transformer.from_crs(
        f"+proj=latlong +datum=WGS84", 
        f"+proj=tmerc +datum=WGS84 +lat_0={lat0} +lon_0={lon0}",
        always_xy=True
    )
    
    transforms = []
    
    for entry in gps_data:
        # Extract data
        lat = entry['lat']
        lon = entry['lon']
        alt = entry['alt']
        roll = entry['roll']
        pitch = entry['pitch']
        yaw = entry['yaw']
        
        # Convert to local ENU (East-North-Up) coordinate system
        east, north = transformer.transform(lon, lat)
        up = alt - alt0
        
        # Create rotation matrix from roll, pitch, yaw
        # Note: The order of rotations depends on your coordinate system convention
        # For ENU system: first yaw around Up, then pitch around East, then roll around North
        R = np.dot(rotz(yaw), np.dot(roty(pitch), rotx(roll)))
        
        # Create 4x4 transformation matrix
        transform = np.eye(4)
        transform[:3, :3] = R
        transform[0, 3] = east
        transform[1, 3] = north
        transform[2, 3] = up
        
        transforms.append(transform)
    
    # Save the transformation matrices to a file
    with open(output_file, 'w') as f:
        for transform in transforms:
            # Write all 16 elements in a single line
            line = ' '.join(map(str, transform.flatten()))
            f.write(line + '\n')
    
    return transforms

def parse_gps_file(input_folder):
    """
    Parse a GPS data folder into a list of dictionaries.
    
    Expected format: Each line contains all the GPS parameters for one timestamp.
    """
    files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith(('.txt'))])
    gps_data = []
    
    for file in files:
        file_path = os.path.join(input_folder, file)
        with open(file_path, 'r') as f:
            for line in f:
                values = line.strip().split()
                if len(values) < 26:  # Ensure we have all required fields
                    continue
                    
                entry = {
                    'lat': float(values[0]),    # latitude
                    'lon': float(values[1]),    # longitude
                    'alt': float(values[2]),    # altitude
                    'roll': float(values[3]),   # roll angle
                    'pitch': float(values[4]),  # pitch angle
                    'yaw': float(values[5]),    # yaw angle
                    'vn': float(values[6]),     # velocity north
                    've': float(values[7]),     # velocity east
                    'vf': float(values[8]),     # forward velocity
                    'vl': float(values[9]),     # leftward velocity
                    'vu': float(values[10]),    # upward velocity
                    'ax': float(values[11]),    # acceleration x
                    'ay': float(values[12]),    # acceleration y
                    'az': float(values[13]),    # acceleration z
                    'af': float(values[14]),    # forward acceleration
                    'al': float(values[15]),    # leftward acceleration
                    'au': float(values[16]),    # upward acceleration
                    'wx': float(values[17]),    # angular rate x
                    'wy': float(values[18]),    # angular rate y
                    'wz': float(values[19]),    # angular rate z
                    'wf': float(values[20]),    # angular rate forward
                    'wl': float(values[21]),    # angular rate leftward
                    'wu': float(values[22]),    # angular rate upward
                    'pos_accuracy': float(values[23]),  # position accuracy
                    'vel_accuracy': float(values[24]),  # velocity accuracy
                    'navstat': int(values[25]),         # navigation status
                }
                
                # Add additional fields if available
                if len(values) > 26:
                    entry['numsats'] = int(values[26])  # number of satellites
                if len(values) > 27:
                    entry['posmode'] = int(values[27])  # position mode
                if len(values) > 28:
                    entry['velmode'] = int(values[28])  # velocity mode
                if len(values) > 29:
                    entry['orimode'] = int(values[29])  # orientation mode
                    
                gps_data.append(entry)
    
    return gps_data

def main():
    """
    Main function to process GPS data and generate transformation matrices.
    """
    # Configure input/output paths
    input_folder = "/scratch3/kat049/concept-graphs/my_local_data/KITTI/2011_09_26/2011_09_26_drive_0019_sync/oxts/data"
    output_file = "/scratch3/kat049/concept-graphs/my_local_data/KITTI/2011_09_26/2011_09_26_drive_0019_sync/oxts/traj.txt"
    # input_file = "oxts/data.txt"  # Update with your GPS data file path
    # output_file = "poses/transform_matrices.txt"
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Parse GPS data
    print(f"Parsing GPS data from {input_folder}...")
    gps_data = parse_gps_file(input_folder)
    print(f"Found {len(gps_data)} GPS entries")
    
    # Convert to transformation matrices
    print("Converting GPS data to transformation matrices...")
    transforms = convert_gps_to_transform_matrices(gps_data, output_file)
    print(f"Saved {len(transforms)} transformation matrices to {output_file}")
    
    # Print example of first transformation matrix
    if transforms:
        print("\nExample of first transformation matrix:")
        print(transforms[0])

if __name__ == "__main__":
    main()