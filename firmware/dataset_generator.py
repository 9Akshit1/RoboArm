import mujoco
import mujoco.viewer
import numpy as np
import random
import xml.etree.ElementTree as ET
import os
import json
import cv2
import math
from datetime import datetime
import argparse

class GaussNewtonIK:
    """Gauss-Newton IK solver - EXACTLY matching the working  implementation"""
    
    def __init__(self, model, step_size=0.5, tol=0.01, max_iter=1000):
        self.model = model
        self.step_size = step_size
        self.tol = tol
        self.max_iter = max_iter
        self.iterations = 0
        self.converged = False
        
        # Pre-allocate jacobians for speed - EXACTLY like 
        self.jacp = np.zeros((3, model.nv))
        self.jacr = np.zeros((3, model.nv))
        
    def check_joint_limits(self, q, joint_indices):
        """Check and clamp joint limits for specific joints - EXACTLY like """
        q_limited = q.copy()
        for i, joint_idx in enumerate(joint_indices):
            if joint_idx < self.model.njnt and joint_idx < len(self.model.jnt_range):
                joint_range = self.model.jnt_range[joint_idx]
                # Check if joint has finite limits
                if not (np.isinf(joint_range[0]) or np.isinf(joint_range[1])):
                    q_limited[i] = np.clip(q_limited[i], joint_range[0], joint_range[1])
                    # Debug: Print when limits are applied
                    if q[i] != q_limited[i]:
                        print(f"Joint {joint_idx} clamped from {q[i]:.3f} to {q_limited[i]:.3f} (range: {joint_range})")
        return q_limited
    
    def calculate_error(self, data, goal, body_id):
        """Calculate position error - EXACTLY like """
        current_pos = data.body(body_id).xpos
        return goal - current_pos
    
    def solve(self, data, goal, init_q, body_id, arm_joint_indices, arm_actuator_indices):
        """Solve using Gauss-Newton method - EXACTLY like """
        # Ensure init_q matches arm DOF count
        if len(init_q) != len(arm_joint_indices):
            init_q = np.zeros(len(arm_joint_indices))

        # Set arm joint positions
        for i, joint_idx in enumerate(arm_joint_indices):
            data.qpos[joint_idx] = init_q[i]
        mujoco.mj_forward(self.model, data)

        self.iterations = 0
        self.converged = False

        for i in range(self.max_iter):
            error = self.calculate_error(data, goal, body_id)
            error_norm = np.linalg.norm(error)

            if error_norm < self.tol:
                self.converged = True
                break

            # CRITICAL FIX: Calculate jacobian at current position, not goal
            current_pos = data.body(body_id).xpos
            mujoco.mj_jac(self.model, data, self.jacp, self.jacr, current_pos, body_id)

            # Extract jacobian columns for arm joints only
            arm_jacp = self.jacp[:, arm_joint_indices]

            # Check for singularity
            if np.linalg.norm(arm_jacp) < 1e-8:
                print(f"Warning: Near-singular jacobian at iteration {i}")
                break

            # Gauss-Newton update with better regularization
            JTJ = arm_jacp.T @ arm_jacp
            
            # Adaptive regularization
            condition_num = np.linalg.cond(JTJ)
            if condition_num > 1e6:
                reg_factor = 1e-3
            else:
                reg_factor = 1e-4
            
            reg = reg_factor * np.eye(JTJ.shape[0])

            try:
                if np.linalg.det(JTJ + reg) > 1e-8:
                    j_inv = np.linalg.inv(JTJ + reg) @ arm_jacp.T
                else:
                    j_inv = np.linalg.pinv(arm_jacp, rcond=1e-4)
            except np.linalg.LinAlgError:
                j_inv = np.linalg.pinv(arm_jacp, rcond=1e-4)

            delta_q = j_inv @ error

            # CRITICAL FIX: Limit step size
            max_step = 0.3
            delta_q_norm = np.linalg.norm(delta_q)
            if delta_q_norm > max_step:
                delta_q = delta_q * (max_step / delta_q_norm)

            # Update arm joint positions
            current_q = np.array([data.qpos[idx] for idx in arm_joint_indices])
            new_q = current_q + self.step_size * delta_q

            # Apply joint limits
            new_q = self.check_joint_limits(new_q, arm_joint_indices)

            # Set new joint positions
            for j, joint_idx in enumerate(arm_joint_indices):
                data.qpos[joint_idx] = new_q[j]

            # Forward kinematics
            mujoco.mj_forward(self.model, data)
            self.iterations = i + 1

        return np.array([data.qpos[idx] for idx in arm_joint_indices])

class DatasetGenerator:
    def __init__(self, output_dir="firmware/dataset", assets_dir="firmware/dataset/scenes/assets"):
        self.output_dir = output_dir
        self.assets_dir = assets_dir
        self.scene_counter = 0
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "scenes"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "annotations"), exist_ok=True)

        # Create geometric meshes if they don't exist
        self.create_geometric_meshes()
        
        # Available objects in assets folder
        self.available_objects = [
            "apple.stl", 
            "banana.stl", 
            "bottle.stl", 
            "bowl.stl", 
            "computer_mouse.stl",
            "cup.stl", 
            "minion.stl",
            "robot.stl", 
            "teddy_bear.stl", 
            "sphere.stl",
            "cylinder.stl",
            "cube.stl"
        ]
        
        # Object scaling factors
        self.object_scales = {
            "apple.stl": 0.12, 
            "banana.stl": 0.001, 
            "book.stl": 0.0065,
            "bottle.stl": 0.002, 
            "bowl.stl": 0.0007, 
            "computer_mouse.stl": 0.0011,
            "cup.stl": 0.001, 
            "dinner_plate.stl": 0.007, 
            "minion.stl": 0.0012,
            "robot.stl": 0.0004, 
            "teddy_bear.stl": 0.003, 
            "vase.stl": 0.0018,
            "sphere.stl": 0.02,
            "cylinder.stl": 0.02,
            "cube.stl": 0.02,
            "default": 0.002
        }
                
        # Enhanced object colors with textures
        self.object_colors = [
            ("object_red", [0.8, 0.2, 0.2, 1]),
            ("object_blue", [0.2, 0.2, 0.8, 1]),
            ("object_green", [0.2, 0.8, 0.2, 1]),
            ("object_yellow", [0.8, 0.8, 0.2, 1]),
            ("object_purple", [0.8, 0.2, 0.8, 1]),
            ("object_orange", [0.8, 0.5, 0.2, 1]),
            ("object_cyan", [0.2, 0.8, 0.8, 1]),
            ("object_pink", [0.8, 0.4, 0.6, 1]),
            ("object_brown", [0.6, 0.4, 0.2, 1]),
            ("object_gray", [0.5, 0.5, 0.5, 1]),
            ("object_white", [0.9, 0.9, 0.9, 1]),
            ("object_black", [0.2, 0.2, 0.2, 1]),
            ("object_stripe_red", [0.8, 0.2, 0.2, 1]),
            ("object_stripe_blue", [0.2, 0.2, 0.8, 1]),
            ("object_dots_green", [0.2, 0.8, 0.2, 1])
        ]
        
        # Enhanced materials
        self.ground_materials = [
            "groundplane1", "groundplane2", "groundplane3", 
            "groundplane4", "groundplane5", "groundplane6"
        ]
        self.table_materials = [
            "wood_mat1", "wood_mat2", "wood_mat3", "wood_mat4", "wood_mat5",
            "marble_mat1", "marble_mat2", "metal_mat1"
        ]
        
        # Lighting configurations
        self.lighting_configs = [
            {"main": [0.5, 0, 2.0], "aux1": [1.0, 0.8, 1.5], "aux2": [-0.3, 0.8, 1.2]},
            {"main": [0.8, 0.3, 1.8], "aux1": [0.5, -0.5, 1.3], "aux2": [0.2, 1.0, 1.0]},
            {"main": [0.2, 0, 2.2], "aux1": [1.2, 0.2, 1.4], "aux2": [-0.5, 0.3, 1.6]},
            {"main": [0.6, -0.2, 1.9], "aux1": [0.8, 1.0, 1.2], "aux2": [-0.2, -0.5, 1.4]},
            {"main": [0.3, 0.5, 2.1], "aux1": [1.1, -0.3, 1.6], "aux2": [0.1, 0.9, 1.1]}
        ]

        # Based on arm base at [0.2, 0, 0.45] from gen3.xml
        self.arm_base_position = [0.2, 0, 0.45]
        self.arm_reach = 0.9  # Gen3 arm reach is approximately 90cm

        # Workspace constraints for target generation
        self.workspace_limits = {
            'min_radius': 0.5,   # INCREASED: Minimum distance from arm base
            'max_radius': 0.85,  # Keep same
            'min_height': 0.9,   # INCREASED: Higher minimum height
            'max_height': 1.6,   # INCREASED: Much higher maximum
            'min_angle': -60,    # Keep same
            'max_angle': 120,    # Keep same
            'preferred_height_range': [1.0, 1.4]  # INCREASED: Much higher preferred range
        }
                                        
        # Table position variations - matching gen3_scene.xml
        self.table_positions = [
            [0.6, 0, 0.43],      # Default from gen3_scene.xml
            [0.65, 0.05, 0.43],  # Slightly right
            [0.55, -0.05, 0.43], # Slightly left
            [0.6, 0.08, 0.43],   # Forward
            [0.6, -0.08, 0.43]   # Backward
        ]

    def create_geometric_meshes(self):
        """Create geometric primitive meshes programmatically"""
        if not os.path.exists(self.assets_dir):
            os.makedirs(self.assets_dir, exist_ok=True)
        
        # Create sphere STL
        sphere_path = os.path.join(self.assets_dir, "sphere.stl")
        if not os.path.exists(sphere_path):
            self.create_sphere_stl(sphere_path, radius=1.0, resolution=20)
        
        # Create cylinder STL  
        cylinder_path = os.path.join(self.assets_dir, "cylinder.stl")
        if not os.path.exists(cylinder_path):
            self.create_cylinder_stl(cylinder_path, radius=1.0, height=2.0, resolution=20)
        
        # Create cube STL
        cube_path = os.path.join(self.assets_dir, "cube.stl")
        if not os.path.exists(cube_path):
            self.create_cube_stl(cube_path, size=2.0)

    def create_sphere_stl(self, filepath, radius=1.0, resolution=20):
        """Create a complete sphere STL file"""
        import numpy as np
        
        vertices = []
        faces = []
        
        # Generate sphere vertices using spherical coordinates
        for i in range(resolution + 1):
            lat = np.pi * (-0.5 + float(i) / resolution)  # Latitude from -π/2 to π/2
            for j in range(resolution):
                lng = 2 * np.pi * float(j) / resolution  # Longitude from 0 to 2π
                x = radius * np.cos(lat) * np.cos(lng)
                y = radius * np.cos(lat) * np.sin(lng)
                z = radius * np.sin(lat)
                vertices.append([x, y, z])
        
        # Generate faces (triangles) connecting the vertices
        for i in range(resolution):
            for j in range(resolution):
                # Current vertex indices
                v1 = i * resolution + j
                v2 = i * resolution + (j + 1) % resolution
                v3 = (i + 1) * resolution + j
                v4 = (i + 1) * resolution + (j + 1) % resolution
                
                # Skip the poles to avoid degenerate triangles
                if i == 0:  # Top cap
                    faces.append([v3, v1, v4])
                elif i == resolution - 1:  # Bottom cap
                    faces.append([v1, v3, v2])
                else:  # Middle sections - create two triangles per quad
                    faces.append([v1, v3, v2])
                    faces.append([v2, v3, v4])
        
        self.write_stl_file(filepath, vertices, faces)

    def create_cylinder_stl(self, filepath, radius=1.0, height=2.0, resolution=20):
        """Create a complete cylinder STL file"""
        import numpy as np
        
        vertices = []
        faces = []
        
        # Create vertices for bottom circle (z = -height/2)
        bottom_center_idx = 0
        vertices.append([0, 0, -height/2])  # Bottom center
        
        for i in range(resolution):
            angle = 2 * np.pi * i / resolution
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            vertices.append([x, y, -height/2])
        
        # Create vertices for top circle (z = height/2)
        top_center_idx = resolution + 1
        vertices.append([0, 0, height/2])  # Top center
        
        for i in range(resolution):
            angle = 2 * np.pi * i / resolution
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            vertices.append([x, y, height/2])
        
        # Generate bottom face triangles (looking up from below)
        for i in range(resolution):
            next_i = (i + 1) % resolution
            bottom_v1 = 1 + i
            bottom_v2 = 1 + next_i
            faces.append([bottom_center_idx, bottom_v2, bottom_v1])  # Counter-clockwise when viewed from below
        
        # Generate top face triangles (looking down from above)
        for i in range(resolution):
            next_i = (i + 1) % resolution
            top_v1 = top_center_idx + 1 + i
            top_v2 = top_center_idx + 1 + next_i
            faces.append([top_center_idx, top_v1, top_v2])  # Counter-clockwise when viewed from above
        
        # Generate side face triangles
        for i in range(resolution):
            next_i = (i + 1) % resolution
            
            # Bottom vertices
            bottom_v1 = 1 + i
            bottom_v2 = 1 + next_i
            
            # Top vertices
            top_v1 = top_center_idx + 1 + i
            top_v2 = top_center_idx + 1 + next_i
            
            # Two triangles per side face
            faces.append([bottom_v1, top_v1, bottom_v2])  # First triangle
            faces.append([bottom_v2, top_v1, top_v2])     # Second triangle
        
        self.write_stl_file(filepath, vertices, faces)

    def create_cube_stl(self, filepath, size=2.0):
        """Create a complete cube STL file"""
        s = size / 2  # Half size for center-based coordinates
        
        # Define 8 vertices of the cube
        vertices = [
            # Bottom face (z = -s)
            [-s, -s, -s],  # 0: bottom-left-back
            [s, -s, -s],   # 1: bottom-right-back
            [s, s, -s],    # 2: bottom-right-front
            [-s, s, -s],   # 3: bottom-left-front
            
            # Top face (z = s)
            [-s, -s, s],   # 4: top-left-back
            [s, -s, s],    # 5: top-right-back
            [s, s, s],     # 6: top-right-front
            [-s, s, s]     # 7: top-left-front
        ]
        
        # Define 12 triangular faces (2 triangles per cube face)
        faces = [
            # Bottom face (z = -s) - normal pointing down (0, 0, -1)
            [0, 2, 1],  # Triangle 1
            [0, 3, 2],  # Triangle 2
            
            # Top face (z = s) - normal pointing up (0, 0, 1)
            [4, 5, 6],  # Triangle 1
            [4, 6, 7],  # Triangle 2
            
            # Front face (y = s) - normal pointing forward (0, 1, 0)
            [3, 6, 2],  # Triangle 1
            [3, 7, 6],  # Triangle 2
            
            # Back face (y = -s) - normal pointing backward (0, -1, 0)
            [0, 1, 5],  # Triangle 1
            [0, 5, 4],  # Triangle 2
            
            # Right face (x = s) - normal pointing right (1, 0, 0)
            [1, 2, 6],  # Triangle 1
            [1, 6, 5],  # Triangle 2
            
            # Left face (x = -s) - normal pointing left (-1, 0, 0)
            [0, 4, 7],  # Triangle 1
            [0, 7, 3],  # Triangle 2
        ]
        
        self.write_stl_file(filepath, vertices, faces)

    def write_stl_file(self, filepath, vertices, faces):
        """Write vertices and faces to STL file with proper normals"""
        import struct
        
        with open(filepath, 'wb') as f:
            # Write 80-byte header
            header = b'Binary STL file created by MuJoCo dataset generator' + b'\x00' * (80 - 47)
            f.write(header)
            
            # Write number of triangles
            f.write(struct.pack('<I', len(faces)))
            
            # Write triangles
            for face in faces:
                if len(face) != 3:
                    continue  # Skip invalid faces
                    
                try:
                    v1, v2, v3 = [vertices[i] for i in face]
                    
                    # Calculate normal vector using cross product
                    u = [v2[j] - v1[j] for j in range(3)]  # Edge 1
                    v = [v3[j] - v1[j] for j in range(3)]  # Edge 2
                    
                    # Cross product u × v
                    normal = [
                        u[1] * v[2] - u[2] * v[1],  # nx
                        u[2] * v[0] - u[0] * v[2],  # ny
                        u[0] * v[1] - u[1] * v[0]   # nz
                    ]
                    
                    # Normalize the normal vector
                    normal_length = (normal[0]**2 + normal[1]**2 + normal[2]**2)**0.5
                    if normal_length > 1e-10:  # Avoid division by zero
                        normal = [n / normal_length for n in normal]
                    else:
                        normal = [0, 0, 1]  # Default normal if calculation fails
                    
                    # Write normal vector (3 floats)
                    for coord in normal:
                        f.write(struct.pack('<f', float(coord)))
                    
                    # Write vertices (9 floats total)
                    for vertex in [v1, v2, v3]:
                        for coord in vertex:
                            f.write(struct.pack('<f', float(coord)))
                    
                    # Write attribute byte count (2 bytes, usually 0)
                    f.write(struct.pack('<H', 0))
                    
                except (IndexError, ValueError) as e:
                    print(f"Warning: Skipping invalid face {face}: {e}")
                    continue

    def find_arm_indices(self, model):
        """Find arm joint and actuator indices - EXACTLY matching """
        arm_joint_indices = []
        arm_actuator_indices = []
        
        print("Finding arm joints and actuators for gen3...")
        
        # Based on gen3.xml, the joints are named joint_1 through joint_7
        expected_joint_names = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'joint_7']
        
        # Find joints by exact name match
        for expected_name in expected_joint_names:
            try:
                joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, expected_name)
                if joint_id >= 0:
                    arm_joint_indices.append(joint_id)
                    print(f"Found joint: {expected_name} (index {joint_id})")
            except:
                print(f"Warning: Could not find joint {expected_name}")
        
        # Find actuators by exact name match (same names as joints in gen3.xml)
        for expected_name in expected_joint_names:
            try:
                actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, expected_name)
                if actuator_id >= 0:
                    arm_actuator_indices.append(actuator_id)
                    print(f"Found actuator: {expected_name} (index {actuator_id})")
            except:
                print(f"Warning: Could not find actuator {expected_name}")
        
        # Fallback: if exact names don't work, use pattern matching
        if not arm_joint_indices:
            print("Fallback: Using pattern matching for joints...")
            for i in range(model.njnt):
                try:
                    joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
                    if joint_name and 'joint_' in joint_name:
                        arm_joint_indices.append(i)
                        print(f"Found joint (fallback): {joint_name} (index {i})")
                except:
                    continue
        
        if not arm_actuator_indices:
            print("Fallback: Using pattern matching for actuators...")
            for i in range(model.nu):
                try:
                    actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                    if actuator_name and 'joint_' in actuator_name:
                        arm_actuator_indices.append(i)
                        print(f"Found actuator (fallback): {actuator_name} (index {i})")
                except:
                    continue
        
        print(f"Found {len(arm_joint_indices)} arm joints: {arm_joint_indices}")
        print(f"Found {len(arm_actuator_indices)} arm actuators: {arm_actuator_indices}")
        
        return arm_joint_indices, arm_actuator_indices

    def get_end_effector_body_id(self, model):
        """Find the end effector body ID - EXACTLY matching """
        # Based on gen3.xml, the end effector is 'bracelet_link'
        end_effector_names = [
            'bracelet_link',           # Exact name from gen3.xml
            'bracelet_with_vision_link', # Alternative
            'spherical_wrist_2_link',  # Second best option
            'spherical_wrist_1_link',  # Third option
            'ee_link',                 # Generic fallback
            'end_effector',
            'gripper',
            'tool0',
            'ee'
        ]
        
        print("Searching for end effector body...")
        
        for name in end_effector_names:
            try:
                body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                if body_id >= 0:  # Valid body ID
                    print(f"Found end effector body: '{name}' (ID: {body_id})")
                    return body_id
            except:
                continue
        
        # Final fallback: find a body with "bracelet" or "wrist" in the name
        print("Searching for bodies with 'bracelet' or 'wrist' in name...")
        for i in range(model.nbody):
            try:
                body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
                if body_name and ('bracelet' in body_name.lower() or 'wrist' in body_name.lower()):
                    print(f"Found end effector body (pattern match): '{body_name}' (ID: {i})")
                    return i
            except:
                continue
        
        # Last resort: use the last body
        if model.nbody > 0:
            body_id = model.nbody - 1
            try:
                body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
                print(f"Using last body as end effector: '{body_name}' (ID: {body_id})")
            except:
                print(f"Using last body as end effector (ID: {body_id})")
            return body_id
        else:
            raise ValueError("No bodies found in model!")

    def point_camera_at_target(self, model, data, target_pos, ee_body_id, arm_joint_indices, ik_solver):
        """Point the end effector camera at the target position"""
        # Get current end effector position
        current_ee_pos = data.body(ee_body_id).xpos.copy()
        
        # Calculate direction vector from end effector to target
        direction = np.array(target_pos) - current_ee_pos
        direction = direction / np.linalg.norm(direction)  # Normalize
        
        # Calculate desired orientation (simplified - point Z-axis toward target)
        # This is a simplified approach - you might need more sophisticated orientation control
        
        # For now, just move the end effector closer to the target while maintaining height
        # Adjust the target position to be slightly above and closer to objects
        adjusted_target = current_ee_pos + direction * 0.1  # Move 10cm toward objects
        adjusted_target[2] = max(adjusted_target[2], target_pos[2] + 0.15)  # Maintain some height
        
        # Use IK to reach this adjusted position
        result_q, final_pos, error = self.move_arm_to_target_ik(
            model, data, adjusted_target, arm_joint_indices, ee_body_id, ik_solver
        )
        
        return result_q, final_pos, error

    def generate_camera_viewpoints(self, objects_center, base_target_pos):
        """Generate multiple camera viewpoints around the objects center"""
        viewpoints = []
        
        # Base viewpoint (original target)
        viewpoints.append({
            'name': 'center',
            'position': base_target_pos,
            'description': 'Center view'
        })
        
        # Generate offset viewpoints around the objects
        offsets = [
            ([0.1, 0.1, 0.05], 'top_right'),
            ([0.1, -0.1, 0.05], 'top_left'), 
            ([-0.1, 0.1, 0.05], 'bottom_right'),
            ([-0.1, -0.1, 0.05], 'bottom_left'),
            ([0.0, 0.0, 0.1], 'overhead'),
            ([0.15, 0.0, 0.0], 'side_right'),
            ([-0.15, 0.0, 0.0], 'side_left')
        ]
        
        for offset, name in offsets:
            viewpoint_pos = [
                base_target_pos[0] + offset[0],
                base_target_pos[1] + offset[1], 
                base_target_pos[2] + offset[2]
            ]
            viewpoints.append({
                'name': name,
                'position': viewpoint_pos,
                'description': f'{name.replace("_", " ").title()} view'
            })
        
        return viewpoints
    
    def add_debug_visuals(self, model, data, ee_body_id, objects_center):
        """Add visual debugging elements to see camera direction and objects center"""
        
        # Find the wrist camera
        try:
            camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist")
            if camera_id >= 0:
                # Get actual camera position and orientation
                camera_pos = data.cam_xpos[camera_id].copy()
                camera_mat = data.cam_xmat[camera_id].reshape(3, 3)
                
                # Camera looks down its -Z axis in MuJoCo
                camera_direction = -camera_mat[:, 2]
            else:
                print("Warning: wrist camera not found, using end effector")
                camera_pos = data.body(ee_body_id).xpos.copy()
                ee_rotation_matrix = data.body(ee_body_id).xmat.reshape(3, 3)
                camera_direction = -ee_rotation_matrix[:, 2]
        except:
            print("Warning: using end effector fallback")
            camera_pos = data.body(ee_body_id).xpos.copy()
            ee_rotation_matrix = data.body(ee_body_id).xmat.reshape(3, 3)
            camera_direction = -ee_rotation_matrix[:, 2]
        
        # Create a line from camera to where it's pointing (extend 0.5m in that direction)
        camera_target_point = camera_pos + camera_direction * 0.5
        
        print("=== DEBUG INFO ===")
        print(f"Camera position: {camera_pos}")
        print(f"Objects center: {objects_center}")
        print(f"Camera direction vector: {camera_direction}")
        print(f"Camera pointing to: {camera_target_point}")
        print(f"Distance to objects center: {np.linalg.norm(np.array(objects_center) - camera_pos):.3f}")
        
        # Calculate if camera is pointing toward objects
        to_objects = np.array(objects_center) - camera_pos
        to_objects_normalized = to_objects / (np.linalg.norm(to_objects) + 1e-8)
        alignment = np.dot(camera_direction, to_objects_normalized)
        print(f"Camera-to-objects alignment: {alignment:.3f} (1.0 = perfect, 0.0 = perpendicular, -1.0 = opposite)")
        print("==================")
        
        return camera_pos, camera_direction, camera_target_point
    
    def calculate_objects_center(self, scene_info):
        """Calculate the 3D center point of all objects on the table"""
        if 'objects' not in scene_info or not scene_info['objects']:
            # Fallback to table center if no objects
            table_pos = scene_info['config']['table_position']
            return [table_pos[0], table_pos[1], table_pos[2] + 0.05]
        
        total_x, total_y, total_z = 0, 0, 0
        count = 0
        
        for obj in scene_info['objects']:
            pos = obj['position']
            total_x += pos[0]
            total_y += pos[1] 
            total_z += pos[2]
            count += 1
        
        if count == 0:
            table_pos = scene_info['config']['table_position']
            return [table_pos[0], table_pos[1], table_pos[2] + 0.05]
        
        center = [total_x/count, total_y/count, total_z/count]
        print(f"Calculated objects center: {center}")
        return center
    
    def generate_valid_target_position(self, table_position, objects_center=None):
        """
        Generate a valid target position that ALWAYS points toward the table/objects.
        Ensures camera never goes behind the arm base.
        """
        import math
        
        arm_base = np.array(self.arm_base_position)
        table_pos = np.array(table_position)
        
        # If objects center is provided, use it; otherwise use table center
        if objects_center is not None:
            focus_point = np.array(objects_center)
        else:
            focus_point = table_pos
        
        # CRITICAL: Only allow positions that are FORWARD of the arm base
        # and point TOWARD the table/objects
        
        # Calculate direction from arm base to focus point
        to_focus = focus_point - arm_base
        to_focus_normalized = to_focus / np.linalg.norm(to_focus)
        
        # Generate position in a cone pointing toward the focus point
        # This ensures we never go "behind" the arm
        
        best_position = None
        best_score = -1
        
        for attempt in range(30):
            # CHANGE THESE VALUES FOR HIGHER CAMERA POSITIONS:
            lateral_offset = random.uniform(-0.15, 0.15)  # Small lateral movement
            vertical_offset = random.uniform(0.4, 0.8)    # INCREASED: Much higher above focus point
            distance_offset = random.uniform(-0.3, -0.1)  # INCREASED: Further back from focus for wider view
            
            # Calculate position
            # Start from focus point and move back toward arm slightly for good viewing
            base_to_focus = focus_point - arm_base
            distance_to_focus = np.linalg.norm(base_to_focus)
            
            # Position along the line from arm to focus, but pulled back for viewing
            viewing_distance = distance_to_focus + distance_offset
            
            # CHANGE THESE LIMITS FOR WIDER VIEW:
            viewing_distance = np.clip(viewing_distance, 
                                    0.5,   # INCREASED: Minimum distance for wider view
                                    self.workspace_limits['max_radius'])
            
            # Calculate target position
            direction = to_focus_normalized
            target_pos = arm_base + direction * viewing_distance
            
            # Add small lateral offset
            # Create perpendicular vector for lateral movement
            up_vector = np.array([0, 0, 1])
            right_vector = np.cross(direction, up_vector)
            right_vector = right_vector / (np.linalg.norm(right_vector) + 1e-8)
            
            target_pos += right_vector * lateral_offset
            target_pos[2] += vertical_offset  # Always lift up for better view
            
            # CHANGE HEIGHT CONSTRAINTS FOR OVERVIEW:
            target_pos[2] = np.clip(target_pos[2], 
                                0.9,   # INCREASED: Higher minimum height
                                1.6)   # INCREASED: Allow much higher positions
            
            # CRITICAL CHECK: Ensure position is FORWARD of arm base (positive X direction from arm)
            if target_pos[0] <= arm_base[0]:
                continue  # Skip positions behind or at arm base
                
            # Check if position points toward focus
            camera_to_focus = focus_point - target_pos
            if np.linalg.norm(camera_to_focus) < 0.3:  # INCREASED: Minimum distance for overview
                continue  # Too close
                
            # Score based on viewing angle and distance
            score = self.score_target_position(target_pos.tolist(), arm_base, focus_point, table_pos)
            
            if score > best_score:
                best_score = score
                best_position = target_pos.tolist()
        
        if best_position is None:
            # CHANGE FALLBACK POSITION FOR OVERVIEW:
            safe_pos = [
                arm_base[0] + 0.4,  # Always forward of arm
                arm_base[1] + 0.1,  # Slightly to the right
                arm_base[2] + 0.6   # INCREASED: Much higher above arm base
            ]
            best_position = safe_pos
            print("Using guaranteed safe fallback position")
        
        print(f"Generated target position: {best_position} (score: {best_score:.3f})")
        print(f"Position relative to arm base: {np.array(best_position) - arm_base}")
        return best_position

    def score_target_position(self, position, arm_base, focus_point, table_pos):
        """
        Score a target position based on multiple criteria:
        - Distance from arm base (prefer middle range)
        - Height above table (prefer elevated view)
        - Angle to focus point (prefer good viewing angle)
        - Workspace constraints compliance
        """
        pos = np.array(position)
        base = np.array(arm_base)
        focus = np.array(focus_point)
        table = np.array(table_pos)
        
        score = 0
        
        # 1. Distance from arm base (prefer middle range for stability)
        distance_to_base = np.linalg.norm(pos - base)
        # CHANGE OPTIMAL DISTANCE FOR WIDER VIEW:
        optimal_distance = 0.7  # INCREASED: Prefer further distances
        distance_score = 1.0 - abs(distance_to_base - optimal_distance) / optimal_distance
        score += distance_score * 0.3
        
        # 2. Height above table (prefer elevated positions for overview)
        height_above_table = pos[2] - table[2]
        # CHANGE HEIGHT PREFERENCES FOR OVERVIEW:
        if 0.5 <= height_above_table <= 1.2:  # INCREASED: Much higher range for table overview
            height_score = 1.0
        elif height_above_table > 1.2:
            height_score = max(0, 1.0 - (height_above_table - 1.2) / 0.5)
        else:
            height_score = max(0, height_above_table / 0.5)
        score += height_score * 0.3
        
        # 3. Viewing angle to focus point (prefer positions that can look down at table)
        to_focus = focus - pos
        distance_to_focus = np.linalg.norm(to_focus)
        
        # CHANGE MINIMUM DISTANCE FOR OVERVIEW:
        if distance_to_focus > 0.3:  # INCREASED: Allow further distances
            # Calculate angle from horizontal
            horizontal_dist = np.linalg.norm(to_focus[:2])
            vertical_dist = -to_focus[2]  # Negative because we want to look down
            
            if horizontal_dist > 0:
                look_down_angle = math.atan2(vertical_dist, horizontal_dist)
                # CHANGE PREFERRED ANGLES FOR WIDER VIEW:
                if 0.3 <= look_down_angle <= 1.0:  # INCREASED: 17-57 degrees looking down
                    angle_score = 1.0
                else:
                    angle_score = max(0, 1.0 - abs(look_down_angle - 0.65) / 0.65)
            else:
                angle_score = 0.5
        else:
            angle_score = 0
        
        score += angle_score * 0.25
        
        # 4. Prefer positions that are not too far to the right
        lateral_distance = abs(pos[1] - base[1])
        if lateral_distance < 0.4:
            lateral_score = 1.0
        else:
            lateral_score = max(0, 1.0 - (lateral_distance - 0.4) / 0.3)
        score += lateral_score * 0.15
        
        return score

    def add_debug_markers_to_scene(self, scene_root, objects_center, ee_pos, camera_direction):
        """Add visual debug markers to the scene XML"""
        worldbody = scene_root.find("worldbody")
        
        # Add marker at objects center (red sphere)
        center_body = ET.SubElement(worldbody, "body")
        center_body.set("name", "objects_center_marker")
        center_body.set("pos", f"{objects_center[0]} {objects_center[1]} {objects_center[2]}")
        center_geom = ET.SubElement(center_body, "geom")
        center_geom.set("name", "center_marker")
        center_geom.set("type", "sphere")
        center_geom.set("size", "0.02")
        center_geom.set("rgba", "1 0 0 1")  # Red
        center_geom.set("contype", "0")
        center_geom.set("conaffinity", "0")
        
        # Add marker at camera position (blue sphere)
        camera_body = ET.SubElement(worldbody, "body")
        camera_body.set("name", "camera_marker")
        camera_body.set("pos", f"{ee_pos[0]} {ee_pos[1]} {ee_pos[2]}")
        camera_geom = ET.SubElement(camera_body, "geom")
        camera_geom.set("name", "camera_marker")
        camera_geom.set("type", "sphere")
        camera_geom.set("size", "0.015")
        camera_geom.set("rgba", "0 0 1 1")  # Blue
        camera_geom.set("contype", "0")
        camera_geom.set("conaffinity", "0")

    def calculate_look_at_orientation(self, ee_pos, target_pos):
        """Calculate quaternion to make end effector look at target"""
        # Direction from end effector to target
        direction = np.array(target_pos) - np.array(ee_pos)
        direction = direction / np.linalg.norm(direction)
        
        # We want the camera (which points along -Z axis in end effector frame) to point at target
        # So we want the -Z axis of end effector to align with direction vector
        
        # Default forward direction (-Z in camera frame)
        forward = np.array([0, 0, -1])
        
        # Calculate rotation axis and angle
        cross = np.cross(forward, direction)
        dot = np.dot(forward, direction)
        
        if np.linalg.norm(cross) < 1e-6:  # Vectors are parallel
            if dot > 0:
                return [0, 0, 0, 1]  # No rotation needed
            else:
                return [1, 0, 0, 0]  # 180 degree rotation
        
        # Normalize cross product
        cross = cross / np.linalg.norm(cross)
        
        # Calculate angle
        angle = np.arccos(np.clip(dot, -1, 1))
        
        # Convert to quaternion
        half_angle = angle / 2
        quat = [
            cross[0] * np.sin(half_angle),
            cross[1] * np.sin(half_angle),
            cross[2] * np.sin(half_angle),
            np.cos(half_angle)
        ]
        
        return quat

    def solve_ik_with_orientation(self, model, data, target_pos, target_quat, arm_joint_indices, ee_body_id, ik_solver):
        """Solve IK with both position and orientation constraints"""
        # This is a simplified version - just solve for position first
        # Then try to adjust orientation (this is complex, so we'll use a simpler approach)
        
        # First solve for position
        init_q = np.zeros(len(arm_joint_indices))
        result_q = ik_solver.solve(data, np.array(target_pos), init_q, ee_body_id, arm_joint_indices, arm_joint_indices)
        
        # For now, we'll just use position-based IK
        # Full 6DOF IK (position + orientation) is much more complex
        return result_q

    def move_arm_to_target_ik(self, model, data, target_position, arm_joint_indices, ee_body_id, ik_solver):
        """Move arm to target position using IK - EXACTLY matching  test_ik_methods"""
        print(f"Moving arm to target position: {target_position}")
        
        # EXACTLY like  - multiple initialization attempts for better convergence
        init_positions = [
            np.zeros(len(arm_joint_indices)),  # Original home position
            np.array([0.0, 0.5, 0.0, 1.5, 0.0, 0.0, 0.0])[:len(arm_joint_indices)],  # Bent position
            np.array([0.0, -0.5, 0.0, 1.0, 0.0, 0.5, 0.0])[:len(arm_joint_indices)]   # Alternative position
        ]
        
        best_result = None
        best_error = float('inf')
        
        for attempt, init_q_attempt in enumerate(init_positions):
            if len(init_q_attempt) != len(arm_joint_indices):
                init_q_attempt = np.zeros(len(arm_joint_indices))
            
            # Reset to this initialization
            for i, joint_idx in enumerate(arm_joint_indices):
                data.qpos[joint_idx] = init_q_attempt[i]
            mujoco.mj_forward(model, data)
            
            try:
                result_q = ik_solver.solve(data, np.array(target_position), init_q_attempt, ee_body_id, 
                                      arm_joint_indices, arm_joint_indices)
                
                # Get final end effector position
                final_pos = data.body(ee_body_id).xpos.copy()
                error = np.linalg.norm(np.array(target_position) - final_pos)
                
                # Keep best result
                if error < best_error:
                    best_error = error
                    best_result = {
                        'joint_angles': result_q,
                        'end_effector_pos': final_pos,
                        'error': error,
                        'iterations': ik_solver.iterations,
                        'converged': ik_solver.converged,
                        'attempt': attempt
                    }
                
                # If we get a good solution, stop trying
                if error < 0.05 and ik_solver.converged:
                    break
                    
            except Exception as e:
                print(f"  Attempt {attempt} failed: {e}")
                continue
        
        if best_result is None:
            # Fallback result
            final_pos = data.body(ee_body_id).xpos.copy()
            error = np.linalg.norm(np.array(target_position) - final_pos)
            best_result = {
                'joint_angles': np.array([data.qpos[idx] for idx in arm_joint_indices]),
                'end_effector_pos': final_pos,
                'error': error,
                'iterations': 0,
                'converged': False,
                'attempt': -1
            }
        
        # Set to best configuration
        for i, joint_idx in enumerate(arm_joint_indices):
            data.qpos[joint_idx] = best_result['joint_angles'][i]
        mujoco.mj_forward(model, data)
        
        print(f"IK Result: target={target_position}, achieved={best_result['end_effector_pos']}, error={best_result['error']:.4f}")
        print(f"Converged: {best_result['converged']}, Iterations: {best_result['iterations']}, Best attempt: {best_result['attempt']}")
        
        return best_result['joint_angles'], best_result['end_effector_pos'], best_result['error']

    def check_objects_exist(self):
        """Check which objects actually exist in the assets folder"""
        if not os.path.exists(self.assets_dir):
            print(f"Warning: {self.assets_dir} folder not found. Using placeholder objects.")
            return []
        
        existing_objects = []
        for obj in self.available_objects:
            if os.path.exists(os.path.join(self.assets_dir, obj)):
                existing_objects.append(obj)
        
        if existing_objects:
            print(f"Found {len(existing_objects)} objects: {existing_objects}")
        else:
            print("No STL files found, will use geometric primitives")
        
        return existing_objects

    def create_base_scene_xml(self):
        """Create the base scene XML structure WITHOUT keyframes to avoid DOF mismatch"""
        xml_content = '''<?xml version="1.0" ?>
    <mujoco model="gen3 scene">

    <!-- Inline robot definition to avoid keyframe conflicts -->
    <compiler angle="radian" meshdir="assets"/>

    <default>
        <joint damping="0.1" armature="0.01"/>
        <default class="visual">
        <geom type="mesh" contype="0" conaffinity="0" group="2" rgba="0.75294 0.75294 0.75294 1"/>
        </default>
        <default class="collision">
        <geom type="mesh" group="3"/>
        </default>
        <default class="large_actuator">
        <general gaintype="fixed" biastype="affine" gainprm="1000" biasprm="0 -1000 -200" forcerange="-39 39"/>
        </default>
        <default class="small_actuator">
        <general gaintype="fixed" biastype="affine" gainprm="1000" biasprm="0 -1000 -200" forcerange="-9 9"/>
        </default>
        <site size="0.001" rgba="0.5 0.5 0.5 0.3" group="4"/>
        <general gaintype="fixed" biastype="affine" gainprm="1000" biasprm="0 -1000 -200" forcerange="-50 50"/>
    </default>

    <option integrator="implicitfast" gravity="0 0 -9.81"/>

    <statistic center="0.4 0 0.5" extent="1.2" meansize="0.05"/>

    <visual>
        <headlight diffuse="0.6 0.6 0.6" ambient="0.1 0.1 0.1" specular="0 0 0"/>
        <rgba haze="0.15 0.25 0.35 1"/>
        <global azimuth="120" elevation="-20" fovy="57"/> <!-- Matches D410 vertical FOV -->
    </visual>

    <asset>
        <!-- Robot meshes -->
        <mesh name="base_link" file="base_link.stl"/>
        <mesh name="shoulder_link" file="shoulder_link.stl"/>
        <mesh name="half_arm_1_link" file="half_arm_1_link.stl"/>
        <mesh name="half_arm_2_link" file="half_arm_2_link.stl"/>
        <mesh name="forearm_link" file="forearm_link.stl"/>
        <mesh name="spherical_wrist_1_link" file="spherical_wrist_1_link.stl"/>
        <mesh name="spherical_wrist_2_link" file="spherical_wrist_2_link.stl"/>
        <mesh name="bracelet_with_vision_link" file="bracelet_with_vision_link.stl"/>
        
        <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
        
        <!-- Enhanced ground textures with more variety -->
        <texture type="2d" name="groundplane1" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" 
        markrgb="0.8 0.8 0.8" width="300" height="300"/>
        <texture type="2d" name="groundplane2" builtin="flat" rgb1="0.15 0.15 0.2" width="300" height="300"/>
        <texture type="2d" name="groundplane3" builtin="checker" mark="cross" rgb1="0.25 0.25 0.25" rgb2="0.35 0.35 0.35" 
        markrgb="0.6 0.6 0.6" width="300" height="300"/>
        <texture type="2d" name="groundplane4" builtin="gradient" rgb1="0.4 0.4 0.5" rgb2="0.2 0.2 0.3" width="300" height="300"/>
        <texture type="2d" name="groundplane5" builtin="checker" mark="none" rgb1="0.3 0.2 0.2" rgb2="0.4 0.3 0.3" width="400" height="400"/>
        <texture type="2d" name="groundplane6" builtin="flat" rgb1="0.1 0.3 0.1" width="300" height="300"/>
        
        <material name="groundplane1" texture="groundplane1" texuniform="true" texrepeat="8 8" reflectance="0.2"/>
        <material name="groundplane2" texture="groundplane2" texuniform="true" reflectance="0.1"/>
        <material name="groundplane3" texture="groundplane3" texuniform="true" texrepeat="6 6" reflectance="0.3"/>
        <material name="groundplane4" texture="groundplane4" texuniform="true" texrepeat="4 4" reflectance="0.15"/>
        <material name="groundplane5" texture="groundplane5" texuniform="true" texrepeat="10 10" reflectance="0.25"/>
        <material name="groundplane6" texture="groundplane6" texuniform="true" reflectance="0.05"/>

        <!-- Enhanced table textures with more wood varieties -->
        <texture type="2d" name="wood_tex1" builtin="checker" mark="edge" rgb1="0.6 0.4 0.2" rgb2="0.5 0.3 0.1" 
        markrgb="0.7 0.5 0.3" width="100" height="100"/>
        <texture type="2d" name="wood_tex2" builtin="flat" rgb1="0.4 0.3 0.2" width="100" height="100"/>
        <texture type="2d" name="wood_tex3" builtin="checker" mark="none" rgb1="0.7 0.5 0.3" rgb2="0.6 0.4 0.2" 
        width="100" height="100"/>
        <texture type="2d" name="wood_tex4" builtin="gradient" rgb1="0.8 0.6 0.4" rgb2="0.6 0.4 0.2" width="150" height="150"/>
        <texture type="2d" name="wood_tex5" builtin="checker" mark="cross" rgb1="0.5 0.35 0.2" rgb2="0.45 0.3 0.15" 
        markrgb="0.6 0.45 0.3" width="120" height="120"/>
        <texture type="2d" name="marble_tex1" builtin="flat" rgb1="0.9 0.9 0.85" width="100" height="100"/>
        <texture type="2d" name="marble_tex2" builtin="gradient" rgb1="0.95 0.95 0.9" rgb2="0.85 0.85 0.8" width="100" height="100"/>
        <texture type="2d" name="metal_tex1" builtin="flat" rgb1="0.7 0.7 0.8" width="100" height="100"/>
        
        <material name="wood_mat1" texture="wood_tex1" texuniform="true" texrepeat="4 4" reflectance="0.1"/>
        <material name="wood_mat2" texture="wood_tex2" texuniform="true" reflectance="0.05"/>
        <material name="wood_mat3" texture="wood_tex3" texuniform="true" texrepeat="5 5" reflectance="0.15"/>
        <material name="wood_mat4" texture="wood_tex4" texuniform="true" texrepeat="3 3" reflectance="0.12"/>
        <material name="wood_mat5" texture="wood_tex5" texuniform="true" texrepeat="6 6" reflectance="0.18"/>
        <material name="marble_mat1" texture="marble_tex1" texuniform="true" texrepeat="2 2" reflectance="0.4"/>
        <material name="marble_mat2" texture="marble_tex2" texuniform="true" texrepeat="2 2" reflectance="0.35"/>
        <material name="metal_mat1" texture="metal_tex1" texuniform="true" texrepeat="8 8" reflectance="0.6"/>
        
        <!-- Enhanced object materials with textures and more colors -->
        <texture type="2d" name="red_tex" builtin="flat" rgb1="0.8 0.2 0.2" width="50" height="50"/>
        <texture type="2d" name="blue_tex" builtin="flat" rgb1="0.2 0.2 0.8" width="50" height="50"/>
        <texture type="2d" name="green_tex" builtin="flat" rgb1="0.2 0.8 0.2" width="50" height="50"/>
        <texture type="2d" name="yellow_tex" builtin="flat" rgb1="0.8 0.8 0.2" width="50" height="50"/>
        <texture type="2d" name="purple_tex" builtin="flat" rgb1="0.8 0.2 0.8" width="50" height="50"/>
        <texture type="2d" name="orange_tex" builtin="flat" rgb1="0.8 0.5 0.2" width="50" height="50"/>
        <texture type="2d" name="cyan_tex" builtin="flat" rgb1="0.2 0.8 0.8" width="50" height="50"/>
        <texture type="2d" name="pink_tex" builtin="flat" rgb1="0.8 0.4 0.6" width="50" height="50"/>
        <texture type="2d" name="brown_tex" builtin="flat" rgb1="0.6 0.4 0.2" width="50" height="50"/>
        <texture type="2d" name="gray_tex" builtin="flat" rgb1="0.5 0.5 0.5" width="50" height="50"/>
        <texture type="2d" name="white_tex" builtin="flat" rgb1="0.9 0.9 0.9" width="50" height="50"/>
        <texture type="2d" name="black_tex" builtin="flat" rgb1="0.2 0.2 0.2" width="50" height="50"/>
        
        <!-- Textured object materials -->
        <texture type="2d" name="stripe_tex1" builtin="checker" mark="edge" rgb1="0.8 0.2 0.2" rgb2="0.9 0.3 0.3" 
        markrgb="0.7 0.1 0.1" width="50" height="50"/>
        <texture type="2d" name="stripe_tex2" builtin="checker" mark="edge" rgb1="0.2 0.2 0.8" rgb2="0.3 0.3 0.9" 
        markrgb="0.1 0.1 0.7" width="50" height="50"/>
        <texture type="2d" name="dots_tex1" builtin="checker" mark="cross" rgb1="0.2 0.8 0.2" rgb2="0.3 0.9 0.3" 
        markrgb="0.1 0.7 0.1" width="50" height="50"/>
        
        <material name="object_red" texture="red_tex" texuniform="true" reflectance="0.1"/>
        <material name="object_blue" texture="blue_tex" texuniform="true" reflectance="0.1"/>
        <material name="object_green" texture="green_tex" texuniform="true" reflectance="0.1"/>
        <material name="object_yellow" texture="yellow_tex" texuniform="true" reflectance="0.15"/>
        <material name="object_purple" texture="purple_tex" texuniform="true" reflectance="0.1"/>
        <material name="object_orange" texture="orange_tex" texuniform="true" reflectance="0.12"/>
        <material name="object_cyan" texture="cyan_tex" texuniform="true" reflectance="0.2"/>
        <material name="object_pink" texture="pink_tex" texuniform="true" reflectance="0.15"/>
        <material name="object_brown" texture="brown_tex" texuniform="true" reflectance="0.05"/>
        <material name="object_gray" texture="gray_tex" texuniform="true" reflectance="0.3"/>
        <material name="object_white" texture="white_tex" texuniform="true" reflectance="0.4"/>
        <material name="object_black" texture="black_tex" texuniform="true" reflectance="0.02"/>
        <material name="object_stripe_red" texture="stripe_tex1" texuniform="true" texrepeat="4 4" reflectance="0.1"/>
        <material name="object_stripe_blue" texture="stripe_tex2" texuniform="true" texrepeat="4 4" reflectance="0.1"/>
        <material name="object_dots_green" texture="dots_tex1" texuniform="true" texrepeat="6 6" reflectance="0.1"/>
    </asset>

    <worldbody>
        <!-- Randomizable lighting setup -->
        <light pos="0.5 0 2.0" directional="true" diffuse="0.8 0.8 0.8"/>
        <light pos="1.0 0.8 1.5" diffuse="0.4 0.4 0.4"/>
        <light pos="-0.3 0.8 1.2" diffuse="0.3 0.3 0.3"/>

        <!-- Ground plane -->
        <geom name="floor" size="0 0 0.05" type="plane" material="groundplane1"/>

        <!-- Larger table - positioned near arm's base, accessible from edge -->
        <geom name="table" type="box" size="0.5 0.4 0.02" pos="0.6 0 0.43" material="wood_mat1"/>

        <!-- Table legs for the larger table -->
        <geom name="table_leg1" type="cylinder" size="0.02 0.21" pos="0.2 0.3 0.21" rgba="0.4 0.2 0.1 1"/>
        <geom name="table_leg2" type="cylinder" size="0.02 0.21" pos="1.0 0.3 0.21" rgba="0.4 0.2 0.1 1"/>
        <geom name="table_leg3" type="cylinder" size="0.02 0.21" pos="0.2 -0.3 0.21" rgba="0.4 0.2 0.1 1"/>
        <geom name="table_leg4" type="cylinder" size="0.02 0.21" pos="1.0 -0.3 0.21" rgba="0.4 0.2 0.1 1"/>
        
        <!-- Robot arm positioned on table edge -->
        <body name="base_link" pos="0.2 0 0.45">
        <inertial pos="-0.000648 -0.000166 0.084487" quat="0.999294 0.00139618 -0.0118387 0.035636" mass="1.697"
            diaginertia="0.00462407 0.00449437 0.00207755"/>
        <geom class="visual" mesh="base_link"/>
        <geom class="collision" mesh="base_link"/>
        <body name="shoulder_link" pos="0 0 0.15643" quat="0 1 0 0">
            <inertial pos="-2.3e-05 -0.010364 -0.07336" quat="0.707051 0.0451246 -0.0453544 0.704263" mass="1.3773"
            diaginertia="0.00488868 0.00457 0.00135132"/>
            <joint name="joint_1" range="-6.2832 6.2832" limited="true" armature="0.1"/>
            <geom class="visual" mesh="shoulder_link"/>
            <geom class="collision" mesh="shoulder_link"/>
            <body name="half_arm_1_link" pos="0 0.005375 -0.12838" quat="1 1 0 0">
            <inertial pos="-4.4e-05 -0.09958 -0.013278" quat="0.482348 0.516286 -0.516862 0.483366" mass="1.1636"
                diaginertia="0.0113017 0.011088 0.00102532"/>
            <joint name="joint_2" range="-2.2 2.2" limited="true" armature="0.1"/>
            <geom class="visual" mesh="half_arm_1_link"/>
            <geom class="collision" mesh="half_arm_1_link"/>
            <body name="half_arm_2_link" pos="0 -0.21038 -0.006375" quat="1 -1 0 0">
                <inertial pos="-4.4e-05 -0.006641 -0.117892" quat="0.706144 0.0213722 -0.0209128 0.707437" mass="1.1636"
                diaginertia="0.0111633 0.010932 0.00100671"/>
                <joint name="joint_3" range="-6.2832 6.2832" limited="true" armature="0.1"/>
                <geom class="visual" mesh="half_arm_2_link"/>
                <geom class="collision" mesh="half_arm_2_link"/>
                <body name="forearm_link" pos="0 0.006375 -0.21038" quat="1 1 0 0">
                <inertial pos="-1.8e-05 -0.075478 -0.015006" quat="0.483678 0.515961 -0.515859 0.483455" mass="0.9302"
                    diaginertia="0.00834839 0.008147 0.000598606"/>
                <joint name="joint_4" range="-2.5656 2.5656" limited="true" armature="0.1"/>
                <geom class="visual" mesh="forearm_link"/>
                <geom class="collision" mesh="forearm_link"/>
                <body name="spherical_wrist_1_link" pos="0 -0.20843 -0.006375" quat="1 -1 0 0">
                    <inertial pos="1e-06 -0.009432 -0.063883" quat="0.703558 0.0707492 -0.0707492 0.703558" mass="0.6781"
                    diaginertia="0.00165901 0.001596 0.000346988"/>
                    <joint name="joint_5" range="-6.2832 6.2832" limited="true" armature="0.1"/>
                    <geom class="visual" mesh="spherical_wrist_1_link"/>
                    <geom class="collision" mesh="spherical_wrist_1_link"/>
                    <body name="spherical_wrist_2_link" pos="0 0.00017505 -0.10593" quat="1 1 0 0">
                    <inertial pos="1e-06 -0.045483 -0.00965" quat="0.44426 0.550121 -0.550121 0.44426" mass="0.6781"
                        diaginertia="0.00170087 0.001641 0.00035013"/>
                    <joint name="joint_6" range="-2.05 2.05" limited="true" armature="0.1"/>
                    <geom class="visual" mesh="spherical_wrist_2_link"/>
                    <geom class="collision" mesh="spherical_wrist_2_link"/>
                    <body name="bracelet_link" pos="0 -0.10593 -0.00017505" quat="1 -1 0 0">
                        <inertial pos="0.000281 0.011402 -0.029798" quat="0.394358 0.596779 -0.577293 0.393789" mass="0.5"
                        diaginertia="0.000657336 0.000587019 0.000320645"/>
                        <joint name="joint_7" range="-6.2832 6.2832" limited="true" armature="0.1"/>
                        <geom class="visual" mesh="bracelet_with_vision_link"/>
                        <geom class="collision" mesh="bracelet_with_vision_link"/>
                        
                        <!-- Camera positioned on the bracelet, facing upward/outward from the bracelet -->
                        <!--Camera FOC is usually 47 to 60 -->
                        <camera name="wrist" pos="0 -0.05639 -0.058475" quat="0 0 0 1" fovy="60" resolution="1280 720"/>  
                        
                        <!-- End effector site pointing in the same direction as camera -->
                        <site name="pinch_site" pos="0 0 -0.061525" quat="0 1 0 0"/>
                    </body>
                    </body>
                </body>
                </body>
            </body>
            </body>
        </body>
        </body>
    </worldbody>

    <actuator>
        <general class="large_actuator" name="joint_1" joint="joint_1" ctrlrange="-6.2832 6.2832"/>
        <general class="large_actuator" name="joint_2" joint="joint_2" ctrlrange="-2.2 2.2"/>
        <general class="large_actuator" name="joint_3" joint="joint_3" ctrlrange="-6.2832 6.2832"/>
        <general class="large_actuator" name="joint_4" joint="joint_4" ctrlrange="-2.5656 2.5656"/>
        <general class="small_actuator" name="joint_5" joint="joint_5" ctrlrange="-6.2832 6.2832"/>
        <general class="small_actuator" name="joint_6" joint="joint_6" ctrlrange="-2.05 2.05"/>
        <general class="small_actuator" name="joint_7" joint="joint_7" ctrlrange="-6.2832 6.2832"/>
    </actuator>

    <!-- REMOVED KEYFRAMES SECTION TO PREVENT DOF MISMATCH ERRORS -->
    <!-- The original gen3.xml keyframes become invalid when freejoint objects are added -->

    </mujoco>'''
        
        return ET.fromstring(xml_content)

    def generate_scene(self, scene_id, config=None):
        """Generate a single scene with specified or random configuration"""
        print(f"Generating scene {scene_id}...")
        
        # Create base scene
        root = self.create_base_scene_xml()

        # Apply configuration or randomize
        if config is None:
            config = self.generate_random_config()
        
        # Apply configuration to scene
        scene_info = self.apply_config_to_scene(root, config)
        scene_info['scene_id'] = scene_id
        scene_info['config'] = config
        
        # Save scene XML
        scene_filename = f"scene_{scene_id:04d}.xml"
        scene_path = os.path.join(self.output_dir, "scenes", scene_filename)
        
        tree = ET.ElementTree(root)
        tree.write(scene_path, encoding="utf-8", xml_declaration=True)
        
        return scene_path, scene_info

    def generate_random_config(self):
        """Generate a random scene configuration with dynamic target positioning"""
        existing_objects = self.check_objects_exist()
        
        # Choose 3-4 objects only
        if existing_objects:
            num_objects = random.randint(3, 4)
            selected_objects = random.sample(existing_objects, min(num_objects, len(existing_objects)))
        else:
            num_objects = random.randint(3, 4)
            selected_objects = ["geometric"] * num_objects
        
        # Choose table position first
        table_position = random.choice(self.table_positions)
        
        # Generate dynamic target position based on table position
        table_center = [table_position[0], table_position[1], table_position[2] + 0.05]
        target_position = self.generate_valid_target_position(table_position, table_center)
        
        config = {
            'objects': selected_objects,
            'ground_material': random.choice(self.ground_materials),
            'table_material': random.choice(self.table_materials),
            'lighting_config': random.choice(self.lighting_configs),
            'target_position': target_position,  # Now dynamically generated
            'table_position': table_position,
            'lighting_intensity': {
                'main': [random.uniform(0.6, 0.9), random.uniform(0.6, 0.9), random.uniform(0.6, 0.9)],
                'aux1': [random.uniform(0.3, 0.5), random.uniform(0.3, 0.5), random.uniform(0.3, 0.5)],
                'aux2': [random.uniform(0.2, 0.4), random.uniform(0.2, 0.4), random.uniform(0.2, 0.4)]
            }
        }
        
        return config

    def apply_config_to_scene(self, root, config):
        """Apply configuration to scene XML"""
        worldbody = root.find("worldbody")
        assets = root.find("asset")
        
        # Update ground and table materials
        for geom in root.findall(".//geom[@name='floor']"):
            geom.set("material", config['ground_material'])

        for geom in root.findall(".//geom[@name='table']"):
            geom.set("material", config['table_material'])
            # Update table position
            table_pos = config['table_position']
            geom.set("pos", f"{table_pos[0]} {table_pos[1]} {table_pos[2]}")

        # Update table legs positions based on table position
        table_pos = config['table_position']
        leg_positions = [
            [table_pos[0] - 0.4, table_pos[1] + 0.3, 0.21],
            [table_pos[0] + 0.4, table_pos[1] + 0.3, 0.21],
            [table_pos[0] - 0.4, table_pos[1] - 0.3, 0.21],
            [table_pos[0] + 0.4, table_pos[1] - 0.3, 0.21]
        ]
        table_legs = []
        table_legs.extend(root.findall(".//geom[@name='table_leg1']"))
        table_legs.extend(root.findall(".//geom[@name='table_leg2']"))
        table_legs.extend(root.findall(".//geom[@name='table_leg3']"))
        table_legs.extend(root.findall(".//geom[@name='table_leg4']"))
        for i, geom in enumerate(table_legs):
            if i < len(leg_positions):
                pos = leg_positions[i]
                geom.set("pos", f"{pos[0]} {pos[1]} {pos[2]}")
        
        # Clear existing lights and add new ones
        for light in worldbody.findall("light"):
            worldbody.remove(light)
        
        # Add configured lights
        lighting_config = config['lighting_config']
        lighting_intensity = config['lighting_intensity']
        
        main_light = ET.SubElement(worldbody, "light")
        main_light.set("pos", f"{lighting_config['main'][0]} {lighting_config['main'][1]} {lighting_config['main'][2]}")
        main_light.set("directional", "true")
        main_light.set("diffuse", f"{lighting_intensity['main'][0]} {lighting_intensity['main'][1]} {lighting_intensity['main'][2]}")
        
        aux1_light = ET.SubElement(worldbody, "light")
        aux1_light.set("pos", f"{lighting_config['aux1'][0]} {lighting_config['aux1'][1]} {lighting_config['aux1'][2]}")
        aux1_light.set("diffuse", f"{lighting_intensity['aux1'][0]} {lighting_intensity['aux1'][1]} {lighting_intensity['aux1'][2]}")
        
        aux2_light = ET.SubElement(worldbody, "light")
        aux2_light.set("pos", f"{lighting_config['aux2'][0]} {lighting_config['aux2'][1]} {lighting_config['aux2'][2]}")
        aux2_light.set("diffuse", f"{lighting_intensity['aux2'][0]} {lighting_intensity['aux2'][1]} {lighting_intensity['aux2'][2]}")
        
        # Add objects
        scene_info = self.add_objects_to_scene(root, worldbody, assets, config)
        
        return scene_info
    
    def freeze_objects_after_settling(self, model, data):
        """Freeze objects in place after they have settled on the table"""
        print("Freezing objects in their settled positions...")
        
        # Find all freejoint objects (our table objects)
        freejoint_indices = []
        for i in range(model.njnt):
            try:
                joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
                if joint_name and 'object_' in joint_name and '_joint' in joint_name:
                    freejoint_indices.append(i)
            except:
                continue
        
        print(f"Found {len(freejoint_indices)} object joints to freeze")
        
        # Set all velocities to zero and add high damping
        for joint_idx in freejoint_indices:
            # Find the joint's DOF indices in qvel
            joint_type = model.jnt_type[joint_idx]
            if joint_type == mujoco.mjtJoint.mjJNT_FREE:  # Freejoint
                # Get the starting index of this joint's DOFs in qvel
                joint_dof_start = 0
                for j in range(joint_idx):
                    if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                        joint_dof_start += 6  # Freejoint has 6 DOFs in velocity space
                    elif model.jnt_type[j] in [mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE]:
                        joint_dof_start += 1
                
                # Zero out the 6 velocity DOFs for this freejoint (3 linear + 3 angular)
                for dof_offset in range(6):
                    if joint_dof_start + dof_offset < model.nv:
                        data.qvel[joint_dof_start + dof_offset] = 0.0
        
        # Add high damping to object joints to prevent movement
        # This is done by modifying the damping parameter for these joints
        for joint_idx in freejoint_indices:
            if joint_idx < len(model.dof_damping):
                # Set very high damping for the joint DOFs
                joint_dof_start = 0
                for j in range(joint_idx):
                    if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                        joint_dof_start += 6
                    elif model.jnt_type[j] in [mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE]:
                        joint_dof_start += 1
                
                # Set high damping for all 6 DOFs of the freejoint
                for dof_offset in range(6):
                    if joint_dof_start + dof_offset < model.nv:
                        model.dof_damping[joint_dof_start + dof_offset] = 100.0  # Very high damping
        
        print("Objects frozen with high damping and zero velocities")
        
    def settle_objects(self, model, data, num_steps=1000):
        """Force objects to settle on the table with gravity"""
        print("Settling objects with gravity...")
        
        # Find arm joint indices to keep arm stationary
        arm_joint_indices, _ = self.find_arm_indices(model)
        
        # Store current arm position
        arm_positions = []
        for joint_idx in arm_joint_indices:
            arm_positions.append(data.qpos[joint_idx])
        
        # Run physics simulation to let objects settle
        for step in range(num_steps):
            # Keep arm stationary
            for i, joint_idx in enumerate(arm_joint_indices):
                data.qpos[joint_idx] = arm_positions[i]
                data.qvel[joint_idx] = 0  # No arm movement
            
            # Step physics
            mujoco.mj_step(model, data)
            
            # Check if objects have settled (low velocities)
            if step > 500 and step % 100 == 0:
                max_vel = 0
                for i in range(model.nbody):
                    if i > 0:  # Skip world body
                        body_vel = np.linalg.norm(data.cvel[i][:3])  # Linear velocity
                        max_vel = max(max_vel, body_vel)
                
                if max_vel < 0.01:  # Objects have settled
                    print(f"Objects settled after {step} steps")
                    

        # Freeze objects after settling
        self.freeze_objects_after_settling(model, data)
        
        print("Object settling and freezing complete")

    def add_objects_to_scene(self, root, worldbody, assets, config):
        """Add objects to the scene and return object information"""
        objects_info = []
        table_pos = config['table_position']
        
        # Define table bounds relative to table position - smaller area to avoid overcrowding
        table_x_min, table_x_max = table_pos[0] - 0.25, table_pos[0] + 0.25
        table_y_min, table_y_max = table_pos[1] - 0.15, table_pos[1] + 0.15
        table_z = table_pos[2] + 0.04  # Slightly above table surface
        
        # Keep track of placed positions to avoid overlap
        placed_positions = []
        min_distance = 0.12  # Increased minimum distance between objects
        
        for i, obj_name in enumerate(config['objects']):
            # Try to find a non-overlapping position
            attempts = 0
            while attempts < 20:  # Max attempts to find valid position
                x = random.uniform(table_x_min, table_x_max)
                y = random.uniform(table_y_min, table_y_max)
                
                # Check if position is too close to existing objects
                valid_position = True
                for prev_pos in placed_positions:
                    distance = np.sqrt((x - prev_pos[0])**2 + (y - prev_pos[1])**2)
                    if distance < min_distance:
                        valid_position = False
                        break
                
                if valid_position:
                    break
                attempts += 1
            
            # Calculate proper height based on object type and scale
            scale = self.object_scales.get(obj_name, self.object_scales["default"])
            if obj_name in ["sphere.stl", "cylinder.stl", "cube.stl"]:
                # For geometric primitives, we know exact dimensions
                if obj_name == "sphere.stl":
                    object_height = scale * 2.0  # diameter
                elif obj_name == "cylinder.stl":
                    object_height = scale * 2.0  # height
                elif obj_name == "cube.stl":
                    object_height = scale * 2.0  # edge length
            else:
                # For STL meshes, estimate height
                object_height = scale * 10  # Conservative estimate
                
            z = table_z + object_height / 2  # Bottom of object sits on table
            placed_positions.append([x, y])

            # Place objects in their normal orientation with just random Z-axis rotation
            #rotation_z = random.uniform(0, 2 * np.pi)
            rotation_z = 0
            quat = f"0 0 {np.sin(rotation_z/2)} {np.cos(rotation_z/2)}"

            '''# More realistic orientations - mostly upright with some variation
            orientation_type = random.choice(['upright', 'lying', 'tilted'])

            if orientation_type == 'upright':
                # Mostly upright with small random rotation around Z-axis
                rotation = random.uniform(0, 2 * np.pi)
                quat = f"0 0 {np.sin(rotation/2)} {np.cos(rotation/2)}"
            elif orientation_type == 'lying':
                # Lying on side - rotate around X or Y axis
                if random.choice([True, False]):
                    # Lying on side (rotate around X-axis)
                    angle = np.pi/2 + random.uniform(-0.2, 0.2)
                    quat = f"{np.sin(angle/2)} 0 0 {np.cos(angle/2)}"
                else:
                    # Lying on other side (rotate around Y-axis)
                    angle = np.pi/2 + random.uniform(-0.2, 0.2)
                    quat = f"0 {np.sin(angle/2)} 0 {np.cos(angle/2)}"
            else:  # tilted
                # Slightly tilted but mostly upright
                tilt_x = random.uniform(-0.3, 0.3)
                tilt_y = random.uniform(-0.3, 0.3)
                rotation_z = random.uniform(0, 2 * np.pi)
                # Combine rotations (simplified)
                quat = f"{np.sin(tilt_x/2)} {np.sin(tilt_y/2)} {np.sin(rotation_z/2)} {np.cos(rotation_z/2)}"'''
            
            
            # Choose random color
            color_name, color_rgba = random.choice(self.object_colors)
            
            if obj_name == "geometric":
                # Use geometric primitives
                obj_info = self.add_geometric_object(worldbody, i, x, y, z, quat, color_name)
            else:
                # Use STL mesh with proper scaling
                obj_info = self.add_mesh_object(assets, worldbody, i, obj_name, x, y, z, quat, color_name)
            
            obj_info.update({
                'position': [x, y, z],
                'rotation': quat,
                'color': color_rgba
            })
            objects_info.append(obj_info)
        
        return {'objects': objects_info}

    def add_mesh_object(self, assets, worldbody, i, obj_file, x, y, z, quat, color_name):
        """Add a mesh object to the scene with proper scaling and physics"""
        mesh_name = f"object_{i}"
        
        # Get scaling factor for this object
        scale = self.object_scales.get(obj_file, self.object_scales["default"])
        
        # Add mesh asset with scaling
        mesh = ET.SubElement(assets, "mesh")
        mesh.set("name", mesh_name)
        mesh.set("file", obj_file)
        mesh.set("scale", f"{scale} {scale} {scale}")
        
        # Calculate proper Z position based on object dimensions and orientation
        # For mesh objects, estimate height based on scale and typical mesh dimensions
        if obj_file in ["sphere.stl", "cylinder.stl", "cube.stl"]:
            if obj_file == "sphere.stl":
                object_height = scale * 1.0  # radius
            elif obj_file == "cylinder.stl":
                object_height = scale * 1.0  # half height
            elif obj_file == "cube.stl":
                object_height = scale * 1.0  # half edge length
        else:
            # For STL meshes, conservative estimate
            object_height = scale * 5  # Estimated half-height
        
        # Parse quaternion to understand orientation
        quat_vals = [float(x) for x in quat.split()]
        # For simplicity, assume object bottom sits on table regardless of orientation
        table_z = z - object_height  # z was passed as table_z + estimated height
        proper_z = table_z + object_height  # Place center at proper height
        
        # Create object body with freejoint for physics
        obj_body = ET.SubElement(worldbody, "body")
        obj_body.set("name", f"object_{i}")
        obj_body.set("pos", f"{x} {y} {proper_z}")
        obj_body.set("quat", quat)

        # Add freejoint to allow the object to settle with gravity
        obj_joint = ET.SubElement(obj_body, "freejoint")
        obj_joint.set("name", f"object_{i}_joint")
        
        # Add visual geom
        obj_geom_visual = ET.SubElement(obj_body, "geom")
        obj_geom_visual.set("name", f"object_{i}_visual")
        obj_geom_visual.set("type", "mesh")
        obj_geom_visual.set("mesh", mesh_name)
        obj_geom_visual.set("material", color_name)
        obj_geom_visual.set("group", "2")
        obj_geom_visual.set("contype", "0")
        obj_geom_visual.set("conaffinity", "0")
        
        # Add collision geom with physics properties
        obj_geom_collision = ET.SubElement(obj_body, "geom")
        obj_geom_collision.set("name", f"object_{i}_collision")
        obj_geom_collision.set("type", "mesh")
        obj_geom_collision.set("mesh", mesh_name)
        obj_geom_collision.set("group", "3")
        obj_geom_collision.set("friction", "0.8 0.02 0.001")  # Better friction for stability
        obj_geom_collision.set("density", "1000")  # kg/m³
        obj_geom_collision.set("solref", "0.01 1")  # Softer contact
        obj_geom_collision.set("solimp", "0.9 0.95 0.001")  # Better contact solver
                
        return {
            'type': 'mesh',
            'file': obj_file,
            'name': f"object_{i}",
            'material': color_name,
            'scale': scale
        }

    def add_geometric_object(self, worldbody, i, x, y, z, quat, color_name):
        """Add a geometric primitive object to the scene with proper physics"""
        geometric_types = [
            {"type": "box", "size": "0.025 0.025 0.025", "height": 0.025},
            {"type": "sphere", "size": "0.02", "height": 0.02},
            {"type": "cylinder", "size": "0.02 0.03", "height": 0.03},
            {"type": "ellipsoid", "size": "0.02 0.025 0.022", "height": 0.022}
        ]

        geom_type = random.choice(geometric_types)
        object_height = geom_type["height"]
        
        # Position object so its bottom sits on the table
        table_z = z - object_height  # z was passed as table_z + estimated height
        proper_z = table_z + object_height  # Center at proper height

        # Create object body with freejoint for physics
        obj_body = ET.SubElement(worldbody, "body")
        obj_body.set("name", f"object_{i}")
        obj_body.set("pos", f"{x} {y} {proper_z}")
        obj_body.set("quat", quat)

        # Add freejoint to allow the object to settle with gravity
        obj_joint = ET.SubElement(obj_body, "freejoint")
        obj_joint.set("name", f"object_{i}_joint")
        
        # Add geom with physics properties
        obj_geom = ET.SubElement(obj_body, "geom")
        obj_geom.set("name", f"object_{i}")
        obj_geom.set("type", geom_type["type"])
        obj_geom.set("size", geom_type["size"])
        obj_geom.set("material", color_name)
        obj_geom.set("friction", "0.8 0.02 0.001")
        obj_geom.set("density", "800")
        obj_geom.set("solref", "0.01 1")  # Softer contact
        obj_geom.set("solimp", "0.9 0.95 0.001")  # Better contact solver
        
        return {
            'type': geom_type["type"],
            'size': geom_type["size"],
            'name': f"object_{i}",
            'material': color_name
        }

    def optimize_camera_alignment_ik(self, model, data, arm_joint_indices, ee_body_id, objects_center, initial_q):
        """Use IK to optimize camera alignment to objects center with BETTER alignment calculation"""
        
        class CameraAlignmentIK:
            def __init__(self, model, objects_center, step_size=0.2, tol=0.05, max_iter=100):
                self.model = model
                self.objects_center = np.array(objects_center)
                self.step_size = step_size
                self.tol = tol
                self.max_iter = max_iter
                self.jacp = np.zeros((3, model.nv))
                self.jacr = np.zeros((3, model.nv))
                
            def get_camera_info(self, data, ee_body_id):
                """Get camera position and direction"""
                try:
                    camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist")
                    if camera_id >= 0:
                        camera_pos = data.cam_xpos[camera_id].copy()
                        camera_mat = data.cam_xmat[camera_id].reshape(3, 3)
                        camera_direction = -camera_mat[:, 2]  # Camera looks down -Z
                    else:
                        raise ValueError("No camera found")
                except:
                    camera_pos = data.body(ee_body_id).xpos.copy()
                    ee_rotation_matrix = data.body(ee_body_id).xmat.reshape(3, 3)
                    camera_direction = -ee_rotation_matrix[:, 2]
                
                return camera_pos, camera_direction
                
            def calculate_alignment_error(self, data, ee_body_id):
                """Calculate alignment error - FIXED VERSION"""
                camera_pos, camera_direction = self.get_camera_info(data, ee_body_id)
                
                # Vector from camera to objects center
                to_objects = self.objects_center - camera_pos
                distance = np.linalg.norm(to_objects)
                
                if distance < 0.1:  # Too close, large penalty
                    return np.array([5.0, 5.0, 5.0])
                
                to_objects_normalized = to_objects / distance
                
                # FIXED: We want camera_direction to ALIGN with to_objects_normalized
                # The error should be the NEGATIVE of alignment (so error = 0 when aligned)
                alignment_dot = np.dot(camera_direction, to_objects_normalized)
                
                # Convert alignment to error: 
                # alignment_dot = 1.0 means perfect alignment (error should be 0)
                # alignment_dot = -1.0 means opposite direction (error should be large)
                alignment_error_magnitude = 1.0 - alignment_dot  # Range: 0 to 2
                
                # Create 3D error vector pointing in correction direction
                # Use cross product to find the rotation axis needed
                cross_product = np.cross(camera_direction, to_objects_normalized)
                cross_magnitude = np.linalg.norm(cross_product)
                
                if cross_magnitude < 1e-6:  # Already aligned or opposite
                    if alignment_dot > 0:
                        error_vector = np.array([0.0, 0.0, 0.0])  # Perfect alignment
                    else:
                        error_vector = np.array([1.0, 0.0, 0.0])  # Opposite, need to rotate
                else:
                    # Error vector in direction of needed rotation
                    error_vector = (cross_product / cross_magnitude) * alignment_error_magnitude
                
                return error_vector
                
            def solve(self, data, arm_joint_indices, ee_body_id, init_q):
                # Set initial position
                for i, joint_idx in enumerate(arm_joint_indices):
                    data.qpos[joint_idx] = init_q[i]
                mujoco.mj_forward(self.model, data)
                
                best_q = init_q.copy()
                best_error_norm = float('inf')
                
                for iteration in range(self.max_iter):
                    error = self.calculate_alignment_error(data, ee_body_id)
                    error_norm = np.linalg.norm(error)
                    
                    if error_norm < best_error_norm:
                        best_error_norm = error_norm
                        best_q = np.array([data.qpos[idx] for idx in arm_joint_indices])
                    
                    if error_norm < self.tol:
                        print(f"Camera alignment converged in {iteration} iterations")
                        break
                    
                    # Get current joint positions
                    current_q = np.array([data.qpos[idx] for idx in arm_joint_indices])
                    
                    # Calculate jacobian numerically (more stable)
                    jacobian = np.zeros((3, len(arm_joint_indices)))
                    delta = 0.001
                    
                    for j in range(len(arm_joint_indices)):
                        # Positive perturbation
                        test_q = current_q.copy()
                        test_q[j] += delta
                        for k, joint_idx in enumerate(arm_joint_indices):
                            data.qpos[joint_idx] = test_q[k]
                        mujoco.mj_forward(self.model, data)
                        error_pos = self.calculate_alignment_error(data, ee_body_id)
                        
                        # Negative perturbation  
                        test_q[j] = current_q[j] - delta
                        for k, joint_idx in enumerate(arm_joint_indices):
                            data.qpos[joint_idx] = test_q[k]
                        mujoco.mj_forward(self.model, data)
                        error_neg = self.calculate_alignment_error(data, ee_body_id)
                        
                        # Finite difference
                        jacobian[:, j] = (error_pos - error_neg) / (2 * delta)
                        
                        # Restore position
                        for k, joint_idx in enumerate(arm_joint_indices):
                            data.qpos[joint_idx] = current_q[k]
                        mujoco.mj_forward(self.model, data)
                    
                    # Solve for joint update using pseudoinverse
                    try:
                        # Use smaller regularization for better convergence
                        JTJ = jacobian.T @ jacobian
                        reg = 1e-6 * np.eye(JTJ.shape[0])
                        delta_q = -np.linalg.solve(JTJ + reg, jacobian.T @ error)
                    except:
                        delta_q = -np.linalg.pinv(jacobian, rcond=1e-4) @ error
                    
                    # Limit step size for stability
                    delta_q_norm = np.linalg.norm(delta_q)
                    if delta_q_norm > 0.1:  # Smaller max step
                        delta_q = delta_q * (0.1 / delta_q_norm)
                    
                    # Update joints
                    new_q = current_q + self.step_size * delta_q
                    
                    # Apply joint limits
                    for i, joint_idx in enumerate(arm_joint_indices):
                        if joint_idx < len(self.model.jnt_range):
                            joint_range = self.model.jnt_range[joint_idx]
                            if not (np.isinf(joint_range[0]) or np.isinf(joint_range[1])):
                                new_q[i] = np.clip(new_q[i], joint_range[0], joint_range[1])
                    
                    # Set new position
                    for i, joint_idx in enumerate(arm_joint_indices):
                        data.qpos[joint_idx] = new_q[i]
                    mujoco.mj_forward(self.model, data)
                
                # Set to best found position
                for i, joint_idx in enumerate(arm_joint_indices):
                    data.qpos[joint_idx] = best_q[i]
                mujoco.mj_forward(self.model, data)
                
                return best_q, best_error_norm
        
        # Use the improved camera alignment IK
        camera_ik = CameraAlignmentIK(model, objects_center)
        optimized_q, final_error = camera_ik.solve(data, arm_joint_indices, ee_body_id, initial_q)
        
        # Calculate final alignment score for verification
        try:
            camera_pos, camera_direction = camera_ik.get_camera_info(data, ee_body_id)
            to_objects = np.array(objects_center) - camera_pos
            distance = np.linalg.norm(to_objects)
            if distance > 0.05:
                to_objects_normalized = to_objects / distance
                final_alignment_score = np.dot(camera_direction, to_objects_normalized)
            else:
                final_alignment_score = 0.0
        except:
            final_alignment_score = 0.0
        
        print(f"Camera alignment optimization completed. Final error: {final_error:.4f}, Alignment score: {final_alignment_score:.3f}")
        return optimized_q

    def validate_scene_interactively(self, scene_path, scene_info):
        """Launch interactive viewer for user to validate the scene using IMPROVED IK positioning"""
        try:
            # Load model for final processing
            model = mujoco.MjModel.from_xml_path(scene_path)
            data = mujoco.MjData(model)

            # Handle keyframe size mismatch by removing keyframes instead of padding
            # This avoids issues with freejoint objects
            if model.nkey > 0:  # If keyframes exist
                # Option 1: Remove keyframes entirely (safest for freejoint objects)
                model.key_time = np.array([])
                model.key_qpos = np.array([]).reshape(0, model.nq)
                model.key_qvel = np.array([]).reshape(0, model.nv) 
                model.key_act = np.array([]).reshape(0, model.na)
                model.nkey = 0
                
                # Option 2: If you want to keep keyframes, pad them properly
                # for i in range(model.nkey):
                #     key_qpos_size = len(model.key_qpos[i])
                #     if key_qpos_size < model.nq:
                #         # Calculate how many freejoint DOFs we added (7 DOFs per freejoint object)
                #         num_objects = len(scene_info['objects'])
                #         freejoint_dofs = num_objects * 7  # Each freejoint adds 7 DOFs (3 pos + 4 quat)
                #         
                #         # Pad keyframe with zeros for the freejoint DOFs
                #         padded_qpos = np.zeros(model.nq)
                #         padded_qpos[:key_qpos_size] = model.key_qpos[i]
                #         # Initialize freejoint positions to reasonable values
                #         for obj_idx in range(num_objects):
                #             start_idx = key_qpos_size + obj_idx * 7
                #             if start_idx + 6 < model.nq:
                #                 # Set object position to table height
                #                 padded_qpos[start_idx + 2] = scene_info['config']['table_position'][2] + 0.1
                #                 # Set quaternion to identity (w=1, x=y=z=0)
                #                 padded_qpos[start_idx + 6] = 1.0  # w component
                #         
                #         model.key_qpos[i] = padded_qpos

            # Find arm components
            arm_joint_indices, arm_actuator_indices = self.find_arm_indices(model)
            ee_body_id = self.get_end_effector_body_id(model)

            # Initialize IK solver
            ik_solver = GaussNewtonIK(model, step_size=0.5, tol=0.01, max_iter=1000)

            # Calculate objects center FIRST
            objects_center = self.calculate_objects_center(scene_info)
            table_pos = scene_info['config']['table_position']
            
            # STEP 1: Position arm at strategic location for good overview
            # Calculate optimal camera position based on objects center and table
            '''camera_height = max(objects_center[2] + 0.3, table_pos[2] + 0.4)  # At least 30cm above objects
            camera_target_pos = [
                objects_center[0] - 0.2,  # 20cm back from objects center
                objects_center[1],        # Aligned with objects Y
                camera_height             # Good height for overview
            ]'''
            # STEP 1: Use the configured target position (don't override it!)
            camera_target_pos = scene_info['config']['target_position']
            print(f"Using configured target position: {camera_target_pos}")
            
            print(f"Objects center: {objects_center}")
            print(f"Strategic camera position: {camera_target_pos}")
            
            # Move to strategic position first
            result_q, final_pos, error = self.move_arm_to_target_ik(
                model, data, camera_target_pos, arm_joint_indices, ee_body_id, ik_solver
            )
            
            if error > 0.1:  # If strategic position failed, try alternative positions
                print("Strategic position failed, trying alternatives...")
                # Use much higher alternative positions for full table view
                alternative_positions = [
                    [objects_center[0], objects_center[1] - 0.3, 1.4],  # INCREASED: Left side much higher
                    [objects_center[0], objects_center[1] + 0.3, 1.4],  # INCREASED: Right side much higher  
                    [objects_center[0] - 0.4, objects_center[1], 1.3],  # INCREASED: Back higher
                    [objects_center[0] + 0.2, objects_center[1], 1.3],  # INCREASED: Front higher
                    [objects_center[0] - 0.2, objects_center[1], 1.5],  # NEW: Even higher back position
                ]
                
                for alt_pos in alternative_positions:
                    result_q, final_pos, error = self.move_arm_to_target_ik(
                        model, data, alt_pos, arm_joint_indices, ee_body_id, ik_solver
                    )
                    if error < 0.1:
                        print(f"Successfully moved to alternative position: {alt_pos}")
                        camera_target_pos = alt_pos
                        break
            
            # STEP 2: Use specialized IK to optimize camera alignment
            print("Optimizing camera alignment with objects center...")
            optimized_q = self.optimize_camera_alignment_ik(
                model, data, arm_joint_indices, ee_body_id, objects_center, result_q
            )
            
            # STEP 3: Fine-tune with wrist adjustments if needed
            print("Fine-tuning wrist orientation...")
            best_q = optimized_q.copy()
            best_alignment = -1.0
            
            # More focused wrist adjustments
            wrist_adjustments = [
                [0, 0, 0],           # No change
                [0.1, 0, 0],         # Small pitch down
                [-0.1, 0, 0],        # Small pitch up
                [0, 0.1, 0],         # Small yaw right
                [0, -0.1, 0],        # Small yaw left
                [0, 0, 0.1],         # Small roll right
                [0, 0, -0.1],        # Small roll left
                [0.05, 0.05, 0],     # Combined adjustments
                [0.05, -0.05, 0],
                [-0.05, 0.05, 0],
                [-0.05, -0.05, 0],
            ]
            
            for adjustment in wrist_adjustments:
                try:
                    test_q = optimized_q.copy()
                    if len(test_q) >= 3:
                        test_q[-3] += adjustment[0]  # Wrist pitch
                        test_q[-2] += adjustment[1]  # Wrist yaw  
                        test_q[-1] += adjustment[2]  # Wrist roll

                    # Apply joint limits
                    test_q = ik_solver.check_joint_limits(test_q, arm_joint_indices)

                    # Set joints and test alignment
                    for j, joint_idx in enumerate(arm_joint_indices):
                        data.qpos[joint_idx] = test_q[j]
                    mujoco.mj_forward(model, data)

                    # Calculate alignment quality
                    try:
                        camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist")
                        if camera_id >= 0:
                            camera_pos = data.cam_xpos[camera_id].copy()
                            camera_mat = data.cam_xmat[camera_id].reshape(3, 3)
                            camera_direction = -camera_mat[:, 2]
                        else:
                            camera_pos = data.body(ee_body_id).xpos.copy()
                            ee_rotation_matrix = data.body(ee_body_id).xmat.reshape(3, 3)
                            camera_direction = -ee_rotation_matrix[:, 2]
                    except:
                        camera_pos = data.body(ee_body_id).xpos.copy()
                        ee_rotation_matrix = data.body(ee_body_id).xmat.reshape(3, 3)
                        camera_direction = -ee_rotation_matrix[:, 2]

                    # Check alignment
                    to_objects = np.array(objects_center) - camera_pos
                    distance = np.linalg.norm(to_objects)
                    
                    if 0.3 < distance < 2.0:  # Allow much larger distances for overview
                        to_objects_normalized = to_objects / distance
                        alignment = np.dot(camera_direction, to_objects_normalized)
                        
                        if alignment > best_alignment:
                            best_alignment = alignment
                            best_q = test_q.copy()
                            print(f"Improved alignment: {alignment:.3f} with adjustment {adjustment}")

                except Exception as e:
                    continue

            # Set final best position
            for j, joint_idx in enumerate(arm_joint_indices):
                data.qpos[joint_idx] = best_q[j]
            mujoco.mj_forward(model, data)

            # Add debug visuals
            ee_pos = data.body(ee_body_id).xpos.copy()
            try:
                camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist")
                if camera_id >= 0:
                    camera_mat = data.cam_xmat[camera_id].reshape(3, 3)
                    camera_direction = -camera_mat[:, 2]
                else:
                    ee_rotation_matrix = data.body(ee_body_id).xmat.reshape(3, 3)
                    camera_direction = -ee_rotation_matrix[:, 2]
            except:
                ee_rotation_matrix = data.body(ee_body_id).xmat.reshape(3, 3)
                camera_direction = -ee_rotation_matrix[:, 2]

            # Regenerate scene with debug markers
            scene_root = ET.parse(scene_path).getroot()
            #self.add_debug_markers_to_scene(scene_root, objects_center, ee_pos, camera_direction)   # Optional debug markers. Never delete this line of code though at all
            
            # Save debug scene
            debug_scene_path = scene_path.replace('.xml', '_debug.xml')
            debug_tree = ET.ElementTree(scene_root)
            debug_tree.write(debug_scene_path, encoding="utf-8", xml_declaration=True)
            
            # Reload with debug markers
            model = mujoco.MjModel.from_xml_path(debug_scene_path)
            data = mujoco.MjData(model)
            
            # Re-find components (indices might change)
            arm_joint_indices, arm_actuator_indices = self.find_arm_indices(model)
            ee_body_id = self.get_end_effector_body_id(model)
            
            # Set arm to final position
            for j, joint_idx in enumerate(arm_joint_indices):
                data.qpos[joint_idx] = best_q[j]
            mujoco.mj_forward(model, data)

            # Let objects settle with gravity for proper positioning
            self.settle_objects(model, data, num_steps=1000)

            print(f"\nValidating scene {scene_info['scene_id']}...")
            print("Check if the scene looks valid:")
            print("- Are objects properly sized and positioned on the table?")
            print("- Is the camera pointing at the objects center? (Red marker)")
            print("- Are there 3-4 objects visible?")
            print("- Does the lighting look good?")
            print(f"- Final camera alignment: {best_alignment:.3f}")
            print("- Blue marker = camera position, Red marker = objects center")
            print("Close the viewer when done checking.")

            # Launch interactive viewer
            with mujoco.viewer.launch_passive(model, data) as viewer_instance:
                while viewer_instance.is_running():
                    mujoco.mj_step(model, data)
                    
                    # Keep arm in position (prevent drift)
                    for i, joint_idx in enumerate(arm_joint_indices):
                        data.qpos[joint_idx] = best_q[i]
                    
                    mujoco.mj_forward(model, data)
                    viewer_instance.sync()

            # User validation
            while True:
                valid = input(f"Is scene {scene_info['scene_id']} valid? (y/n/q to quit): ").lower().strip()
                if valid in ['y', 'yes']:
                    return True
                elif valid in ['n', 'no']:
                    return False
                elif valid in ['q', 'quit']:
                    return None
                else:
                    print("Please enter 'y' for yes, 'n' for no, or 'q' to quit")

        except Exception as e:
            print(f"Error validating scene {scene_info['scene_id']}: {e}")
            import traceback
            traceback.print_exc()
            return False

    def capture_scene_data(self, scene_path, scene_info, capture_images=True, validate=True):
        """Load scene and capture images with IMPROVED camera alignment"""
        try:
            # Validate scene if requested
            if validate:
                validation_result = self.validate_scene_interactively(scene_path, scene_info)
                if validation_result is None:
                    return None
                elif not validation_result:
                    print(f"Scene {scene_info['scene_id']} marked as invalid, skipping...")
                    return False

            # Load model for final processing
            model = mujoco.MjModel.from_xml_path(scene_path)
            data = mujoco.MjData(model)

            # Handle keyframe size mismatch by removing keyframes instead of padding
            # This avoids issues with freejoint objects
            if model.nkey > 0:  # If keyframes exist
                # Option 1: Remove keyframes entirely (safest for freejoint objects)
                model.key_time = np.array([])
                model.key_qpos = np.array([]).reshape(0, model.nq)
                model.key_qvel = np.array([]).reshape(0, model.nv) 
                model.key_act = np.array([]).reshape(0, model.na)
                model.nkey = 0
                
                # Option 2: If you want to keep keyframes, pad them properly
                # for i in range(model.nkey):
                #     key_qpos_size = len(model.key_qpos[i])
                #     if key_qpos_size < model.nq:
                #         # Calculate how many freejoint DOFs we added (7 DOFs per freejoint object)
                #         num_objects = len(scene_info['objects'])
                #         freejoint_dofs = num_objects * 7  # Each freejoint adds 7 DOFs (3 pos + 4 quat)
                #         
                #         # Pad keyframe with zeros for the freejoint DOFs
                #         padded_qpos = np.zeros(model.nq)
                #         padded_qpos[:key_qpos_size] = model.key_qpos[i]
                #         # Initialize freejoint positions to reasonable values
                #         for obj_idx in range(num_objects):
                #             start_idx = key_qpos_size + obj_idx * 7
                #             if start_idx + 6 < model.nq:
                #                 # Set object position to table height
                #                 padded_qpos[start_idx + 2] = scene_info['config']['table_position'][2] + 0.1
                #                 # Set quaternion to identity (w=1, x=y=z=0)
                #                 padded_qpos[start_idx + 6] = 1.0  # w component
                #         
                #         model.key_qpos[i] = padded_qpos

            # Find arm components
            arm_joint_indices, arm_actuator_indices = self.find_arm_indices(model)
            ee_body_id = self.get_end_effector_body_id(model)

            # Initialize IK solver
            ik_solver = GaussNewtonIK(model, step_size=0.5, tol=0.01, max_iter=1000)

            # Move to target position (same as validation)
            target_position = scene_info['config']['target_position']
            result_q, final_pos, error = self.move_arm_to_target_ik(
                model, data, target_position, arm_joint_indices, ee_body_id, ik_solver
            )

            # Store IK results
            scene_info['ik_results'] = {
                'target_position': target_position,
                'achieved_position': final_pos.tolist(),
                'joint_angles': result_q.tolist(),
                'error': error,
                'converged': ik_solver.converged,
                'iterations': ik_solver.iterations
            }

            # Let objects settle with gravity for proper positioning
            self.settle_objects(model, data, num_steps=1000)

            # Calculate initial alignment score for metadata
            try:
                camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist")
                if camera_id >= 0:
                    camera_pos = data.cam_xpos[camera_id].copy()
                    camera_mat = data.cam_xmat[camera_id].reshape(3, 3)
                    camera_direction = -camera_mat[:, 2]
                else:
                    camera_pos = data.body(ee_body_id).xpos.copy()
                    ee_rotation_matrix = data.body(ee_body_id).xmat.reshape(3, 3)
                    camera_direction = -ee_rotation_matrix[:, 2]
                    
                objects_center = self.calculate_objects_center(scene_info)
                to_objects = np.array(objects_center) - camera_pos
                distance = np.linalg.norm(to_objects)
                
                if distance > 0.05:
                    to_objects_normalized = to_objects / distance
                    initial_alignment_score = np.dot(camera_direction, to_objects_normalized)
                else:
                    initial_alignment_score = 0.0
            except:
                initial_alignment_score = 0.0

            scene_info['ik_results']['initial_camera_alignment'] = initial_alignment_score

            if capture_images:
                # Use same improved alignment as validation
                objects_center = self.calculate_objects_center(scene_info)
                table_pos = scene_info['config']['table_position']
                
                '''# Calculate optimal camera position
                camera_height = max(objects_center[2] + 0.3, table_pos[2] + 0.4)
                camera_target_pos = [
                    objects_center[0] - 0.2,
                    objects_center[1],
                    camera_height
                ]'''
                # STEP 1: Use the configured target position (don't override it!)
                camera_target_pos = scene_info['config']['target_position']
                print(f"Using configured target position: {camera_target_pos}")

                # ADD HEIGHT BOOST FOR OVERVIEW:
                # Ensure the camera position is high enough for full table view
                if camera_target_pos[2] < table_pos[2] + 0.5:  # If too low
                    camera_target_pos[2] = table_pos[2] + 0.6   # Boost to overview height
                    print(f"Boosted camera height for overview: {camera_target_pos}")

                
                # Move to strategic position
                overview_q, overview_final_pos, overview_error = self.move_arm_to_target_ik(
                    model, data, camera_target_pos, arm_joint_indices, ee_body_id, ik_solver
                )
                
                # Optimize alignment with retry logic
                max_alignment_attempts = 3
                alignment_threshold = 0.8
                best_alignment_score = -1.0

                for alignment_attempt in range(max_alignment_attempts):
                    print(f"Camera alignment attempt {alignment_attempt + 1}/{max_alignment_attempts}")
                    
                    # Try different starting positions for better alignment
                    if alignment_attempt == 0:
                        start_q = overview_q
                    elif alignment_attempt == 1:
                        # Try alternative position slightly higher
                        alt_pos = [camera_target_pos[0], camera_target_pos[1], camera_target_pos[2] + 0.1]
                        start_q, _, _ = self.move_arm_to_target_ik(
                            model, data, alt_pos, arm_joint_indices, ee_body_id, ik_solver
                        )
                    else:
                        # Try alternative position from different angle
                        alt_pos = [camera_target_pos[0] - 0.1, camera_target_pos[1] + 0.1, camera_target_pos[2]]
                        start_q, _, _ = self.move_arm_to_target_ik(
                            model, data, alt_pos, arm_joint_indices, ee_body_id, ik_solver
                        )
                    
                    # Optimize alignment from this starting position
                    candidate_q = self.optimize_camera_alignment_ik(
                        model, data, arm_joint_indices, ee_body_id, objects_center, start_q
                    )
                    
                    # Calculate final alignment score
                    for j, joint_idx in enumerate(arm_joint_indices):
                        data.qpos[joint_idx] = candidate_q[j]
                    mujoco.mj_forward(model, data)
                    
                    # Get camera alignment score
                    try:
                        camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist")
                        if camera_id >= 0:
                            camera_pos = data.cam_xpos[camera_id].copy()
                            camera_mat = data.cam_xmat[camera_id].reshape(3, 3)
                            camera_direction = -camera_mat[:, 2]
                        else:
                            camera_pos = data.body(ee_body_id).xpos.copy()
                            ee_rotation_matrix = data.body(ee_body_id).xmat.reshape(3, 3)
                            camera_direction = -ee_rotation_matrix[:, 2]
                    except:
                        camera_pos = data.body(ee_body_id).xpos.copy()
                        ee_rotation_matrix = data.body(ee_body_id).xmat.reshape(3, 3)
                        camera_direction = -ee_rotation_matrix[:, 2]
                    
                    # Calculate alignment with objects center
                    to_objects = np.array(objects_center) - camera_pos
                    distance = np.linalg.norm(to_objects)
                    
                    if distance > 0.05:
                        to_objects_normalized = to_objects / distance
                        alignment_score = np.dot(camera_direction, to_objects_normalized)
                    else:
                        alignment_score = 0.0
                    
                    print(f"Alignment attempt {alignment_attempt + 1}: score = {alignment_score:.3f}")
                    
                    # Keep best result
                    if alignment_score > best_alignment_score:
                        best_alignment_score = alignment_score
                        best_q = candidate_q
                    
                    # Check if we meet the threshold
                    if alignment_score >= alignment_threshold:
                        print(f"✓ Alignment threshold met: {alignment_score:.3f} >= {alignment_threshold}")
                        best_q = candidate_q
                        break
                    else:
                        print(f"✗ Alignment below threshold: {alignment_score:.3f} < {alignment_threshold}")

                # Check final alignment
                if best_alignment_score < alignment_threshold:
                    print(f"WARNING: Final alignment {best_alignment_score:.3f} below threshold {alignment_threshold}")
                    print("Consider regenerating this scene or adjusting parameters")
                    # Optionally return False to skip this scene
                    return False

                # Set to best found position
                for j, joint_idx in enumerate(arm_joint_indices):
                    data.qpos[joint_idx] = best_q[j]
                mujoco.mj_forward(model, data)

                # Store final alignment score in scene info
                scene_info['ik_results']['final_camera_alignment'] = best_alignment_score

                # Generate multiple viewpoints with small variations
                viewpoint_variations = [
                    ([0, 0, 0, 0, 0, 0, 0], 'center'),
                    ([0, 0, 0, 0, 0, 0.15, 0], 'right'),
                    ([0, 0, 0, 0, 0, -0.15, 0], 'left'), 
                    ([0, 0, 0, 0, 0.1, 0, 0], 'up'),
                    ([0, 0, 0, 0, -0.1, 0, 0], 'down'),
                    ([0, 0, 0, 0, 0, 0, 0.2], 'rotate_cw'),
                    ([0, 0, 0, 0, 0, 0, -0.2], 'rotate_ccw'),
                ]

                captured_images = []

                for variation, angle_name in viewpoint_variations:
                    renderer = None  # Initialize renderer variable
                    try:
                        # Apply variation to best aligned position
                        varied_q = best_q.copy()
                        for i, delta in enumerate(variation):
                            if i < len(varied_q):
                                varied_q[i] += delta

                        # Apply joint limits
                        varied_q = ik_solver.check_joint_limits(varied_q, arm_joint_indices)

                        # Set arm position
                        for j, joint_idx in enumerate(arm_joint_indices):
                            data.qpos[joint_idx] = varied_q[j]
                        mujoco.mj_forward(model, data)

                        # Capture image with proper cleanup
                        camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist")
                        if camera_id >= 0:
                            renderer = mujoco.Renderer(model, height=480, width=640)
                            renderer.update_scene(data, camera=camera_id)
                            image = renderer.render()

                            # Check image quality
                            if np.mean(image) > 10:  # Not all black
                                image_filename = f"scene_{scene_info['scene_id']:04d}_{angle_name}.png"
                                image_path = os.path.join(self.output_dir, "images", image_filename)
                                cv2.imwrite(image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

                                captured_images.append({
                                    'angle': angle_name,
                                    'path': image_path,
                                    'joint_variation': variation,
                                    'joint_angles': varied_q.tolist()
                                })

                    except Exception as e:
                        print(f"Warning: Could not capture {angle_name} view: {e}")
                    finally:
                        # Properly close renderer to prevent memory leaks
                        if renderer is not None:
                            renderer.close()

                scene_info['captured_images'] = captured_images
                scene_info['objects_center'] = objects_center
                scene_info['camera_alignment_method'] = 'improved_ik_optimization'
                
                if captured_images:
                    scene_info['image_path'] = captured_images[0]['path']

            # Save annotation
            annotation_filename = f"scene_{scene_info['scene_id']:04d}.json"
            annotation_path = os.path.join(self.output_dir, "annotations", annotation_filename)

            with open(annotation_path, 'w') as f:
                json.dump(scene_info, f, indent=2)

            return True

        except Exception as e:
            print(f"Error processing scene {scene_info['scene_id']}: {e}")
            import traceback
            traceback.print_exc()
            return False

    def generate_dataset(self, num_scenes=100, capture_images=True, validate_scenes=True):
        """Generate a complete dataset with optional validation"""
        print(f"Generating dataset with {num_scenes} scenes...")
        print(f"Output directory: {self.output_dir}")
        
        if validate_scenes:
            print("Interactive validation enabled - you'll review each scene before it's saved")
        
        successful_scenes = 0
        failed_scenes = 0
        skipped_scenes = 0
        
        # Create dataset metadata
        dataset_info = {
            'creation_time': datetime.now().isoformat(),
            'num_scenes': num_scenes,
            'assets_directory': self.assets_dir,
            'available_objects': self.check_objects_exist(),
            'total_object_types': len(self.available_objects),
            'target_positioning': 'dynamic_workspace_based',
            'workspace_limits': self.workspace_limits,
            'arm_reach': self.arm_reach,
            'dynamic_target_generation': True,
            'lighting_configs_count': len(self.lighting_configs),
            'validation_enabled': validate_scenes,
            'ik_method': 'GaussNewtonIK (FIXED from )',
        }
        
        for scene_id in range(num_scenes):
            try:
                # Generate scene
                scene_path, scene_info = self.generate_scene(scene_id)
                
                # Capture data with optional validation
                result = self.capture_scene_data(scene_path, scene_info, capture_images, validate_scenes)
                
                if result is None:  # User quit validation
                    print("Dataset generation stopped by user.")
                    break
                elif result:
                    successful_scenes += 1
                    if (scene_id + 1) % 10 == 0:
                        print(f"Generated {scene_id + 1}/{num_scenes} scenes...")
                else:
                    if validate_scenes:
                        skipped_scenes += 1
                    else:
                        failed_scenes += 1
                    
            except Exception as e:
                print(f"Failed to generate scene {scene_id}: {e}")
                failed_scenes += 1
        
        # Save dataset metadata
        dataset_info['successful_scenes'] = successful_scenes
        dataset_info['failed_scenes'] = failed_scenes
        dataset_info['skipped_scenes'] = skipped_scenes
        
        metadata_path = os.path.join(self.output_dir, "dataset_info.json")
        with open(metadata_path, 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        print(f"\nDataset generation complete!")
        print(f"Successful scenes: {successful_scenes}")
        print(f"Failed scenes: {failed_scenes}")
        if validate_scenes:
            print(f"Skipped scenes: {skipped_scenes}")
        print(f"Dataset saved to: {self.output_dir}")
        
        return dataset_info


def main():
    generator = DatasetGenerator()

    num_scenes = int(input("Enter number of scenes to automatically generate (default is 3): ") or 3)
    
    dataset_info = generator.generate_dataset(num_scenes=num_scenes, validate_scenes=False)
    print("✓ Dataset generation test completed!")
    print(f"✓ Successfully generated {dataset_info['successful_scenes']} scenes")

if __name__ == "__main__":
    main()