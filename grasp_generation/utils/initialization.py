"""
Hand Initialization Utilities for Affordance-Aligned Grasping
"""

import numpy as np
import torch
import math
import transforms3d
import trimesh as tm
import open3d as o3d
from utils.hand_model import HandModel
from utils.object_model import ObjectModel


def extract_contact_normal(object_model, contact_threshold=0.5):
    """
    Extract representative contact point and normal from affordance data
    
    Args:
        object_model: ObjectModel instance containing affordance data
        contact_threshold: Threshold for contact points
        
    Returns:
        closest_points: Representative contact points (5, 3)
        representative_normals: Contact normal vectors (5, 3)
        closest_indices: Indices of closest points (5,)
    """
    # Extract data from object_model
    xyz = object_model.affordance_xyz.detach().cpu().numpy()
    contact_vals = object_model.affordance_contact_maps.detach().cpu().numpy()
    
    # Compute point cloud normals once
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    normals = np.asarray(pcd.normals)
    center = xyz.mean(axis=0)
    
    closest_points = np.zeros((5, 3))
    representative_normals = np.zeros((5, 3))
    closest_indices = np.zeros(5, dtype=int)
    
    # Process each of the 5 contact maps
    for i in range(5):
        contact_vals_i = contact_vals[i]
        contact_mask = contact_vals_i > contact_threshold
        
        contact_points = xyz[contact_mask]
        mean_point = np.mean(contact_points, axis=0)
        
        # Find closest point to mean contact point
        distances = np.linalg.norm(xyz - mean_point, axis=1)
        closest_idx = np.argmin(distances)
        representative_normal = normals[closest_idx]
        
        # Ensure normal points outward
        if np.dot(representative_normal, xyz[closest_idx] - center) < 0:
            representative_normal = -representative_normal
        
        closest_points[i] = xyz[closest_idx]
        representative_normals[i] = representative_normal
        closest_indices[i] = closest_idx
    return closest_points, representative_normals, closest_indices


def initialize_hand_with_contact_normal(hand_model, object_model, closest_points, contact_normals, config):
    """
    Initialize grasp using contact normal directions
    
    Args:
        hand_model: HandModel instance
        object_model: ObjectModel instance  
        closest_points: Representative contact points (5, 3)
        contact_normals: Contact normal vectors (5, 3)
        config: Configuration object with initialization parameters
    """
    device = hand_model.device
    n_objects = len(object_model.object_mesh_list)
    n_contact_normals = 5
    total_batch_size = n_objects * n_contact_normals * config.poses_per_contact

    closest_points_torch = torch.tensor(closest_points, dtype=torch.float, device=device)
    contact_normals_torch = torch.tensor(contact_normals, dtype=torch.float, device=device)
    
    translation = torch.zeros([total_batch_size, 3], dtype=torch.float, device=device)
    rotation = torch.zeros([total_batch_size, 3, 3], dtype=torch.float, device=device)

    pose_idx = 0
    
    for i in range(n_objects):
        mesh_origin = object_model.object_mesh_list[i].convex_hull
        vertices = mesh_origin.vertices.copy()
        faces = mesh_origin.faces
        vertices *= object_model.object_scale_tensor[i].max().item()
        mesh_origin = tm.Trimesh(vertices, faces)
        vertices += 0.2 * vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
        mesh = tm.Trimesh(vertices=vertices, faces=faces).convex_hull
        
        # Process each of the 5 contact normals
        for normal_idx in range(n_contact_normals):
            closest_point_torch = closest_points_torch[normal_idx]
            contact_normal_torch = contact_normals_torch[normal_idx]
            
            # Find intersection point along normal direction
            ray_origins = closest_point_torch.detach().cpu().numpy().reshape(1, -1)
            ray_directions = contact_normal_torch.detach().cpu().numpy().reshape(1, -1)
            
            locations, _, _ = mesh.ray.intersects_location(
                ray_origins=ray_origins, ray_directions=ray_directions
            )
            
            if len(locations) > 0:
                intersection_point = locations[0]
                p_single = torch.tensor(intersection_point, dtype=torch.float, device=device)
            else:
                extension_distance = 0.3
                p_single = closest_point_torch + extension_distance * contact_normal_torch
            
            p = p_single.unsqueeze(0).repeat(config.poses_per_contact, 1)
            n = -contact_normal_torch / contact_normal_torch.norm()
            n = n.view(1, 3).repeat(config.poses_per_contact, 1)

            distance = config.distance_lower + (config.distance_upper - config.distance_lower) * torch.rand([config.poses_per_contact], device=device)
            deviate_theta = config.theta_lower + (config.theta_upper - config.theta_lower) * torch.rand([config.poses_per_contact], device=device)
            process_theta = 2 * math.pi * torch.rand([config.poses_per_contact], device=device)
            rotate_theta = 2 * math.pi * torch.rand([config.poses_per_contact], device=device)

            rotation_local = torch.zeros([config.poses_per_contact, 3, 3], dtype=torch.float, device=device)
            rotation_global = torch.zeros([config.poses_per_contact, 3, 3], dtype=torch.float, device=device)
            
            for j in range(config.poses_per_contact):
                rotation_local[j] = torch.tensor(
                    transforms3d.euler.euler2mat(process_theta[j], deviate_theta[j], rotate_theta[j], axes='rzxz'),
                    dtype=torch.float, device=device
                )
                rotation_global[j] = torch.tensor(
                    transforms3d.euler.euler2mat(math.atan2(n[j, 1], n[j, 0]) - math.pi / 2, -math.acos(n[j, 2]), 0, axes='rzxz'),
                    dtype=torch.float, device=device
                )
            
            start_idx = pose_idx
            end_idx = pose_idx + config.poses_per_contact
            
            translation[start_idx:end_idx] = p - distance.unsqueeze(1) * (
                rotation_global @ rotation_local @ torch.tensor([0, 0, 1], dtype=torch.float, device=device).reshape(1, -1, 1)
            ).squeeze(2)
            
            rotation_hand = torch.tensor(transforms3d.euler.euler2mat(0, -np.pi / 3, 0, axes='rzxz'), dtype=torch.float, device=device)
            rotation[start_idx:end_idx] = rotation_global @ rotation_local @ rotation_hand
            pose_idx += config.poses_per_contact
    
    # Initialize joint angles
    joint_angles_mu = torch.tensor([0.1, 0, 0.6, 0, 0, 0, 0.6, 0, -0.1, 0, 0.6, 0, 0, -0.2, 0, 0.6, 0, 0, 1.2, 0, -0.2, 0], dtype=torch.float, device=device)
    joint_angles_sigma = config.jitter_strength * (hand_model.joints_upper - hand_model.joints_lower)
    joint_angles = torch.zeros([total_batch_size, hand_model.n_dofs], dtype=torch.float, device=device)
    
    for i in range(hand_model.n_dofs):
        torch.nn.init.trunc_normal_(
            joint_angles[:, i], joint_angles_mu[i], joint_angles_sigma[i],
            hand_model.joints_lower[i] - 1e-6, hand_model.joints_upper[i] + 1e-6
        )

    hand_pose = torch.cat([
        translation,
        rotation.transpose(1, 2)[:, :2].reshape(-1, 6),
        joint_angles
    ], dim=1)
    hand_pose.requires_grad_()

    contact_point_indices = torch.randint(hand_model.n_contact_candidates, size=[total_batch_size, config.n_contact], device=device)
    hand_model.set_parameters(hand_pose, contact_point_indices)