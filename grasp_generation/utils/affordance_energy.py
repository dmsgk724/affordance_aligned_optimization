"""
Energy Functions for Affordance-Aligned Grasping

This module implements the energy functions used in affordance-aligned grasp optimization:
- Distance Energy (E_dis): Equation (3) from the paper
- Barrier Energy (E_bar): Part-contact avoidance using affordance maps
- Direction Alignment Energy (E_dir): Contact normal alignment
- Force Closure Energy (E_fc): Standard force closure computation
- Regularization Energies: Penetration, self-penetration, and joint limits
"""

import torch
import numpy as np


def compute_distance_energy(hand_model, object_model, d0=0.01):
    """
    Compute distance energy as defined in equation (3):
    E_dis = sum_{i=1}^{n} d(x_i, O) + w_palm * |d(x_palm, O) - d0|
    
    Args:
        hand_model: HandModel instance
        object_model: ObjectModel instance
        d0: Desired palm distance (default: 1cm = 0.01m)
        w_palm: Weight for palm distance term
        
    Returns:
        E_dis_fingertips: Distance energy for fingertip contact points
    """
    contact_points = hand_model.contact_points  # (batch_size, n_contact, 3)
    
    # Calculate distances from contact points to object surface
    distances, _ = object_model.cal_distance(contact_points)  # (batch_size, n_contact)
    distances = distances.abs()  # Ensure positive distances
    
    # Fingertip distance energy: sum of distances for all contact points
    E_dis_fingertips = torch.sum(distances, dim=1)  # (batch_size,)
    
    return E_dis_fingertips


def barrier_function(d, d_thr):
    """
    Truncated barrier function from IPC simulator.
    
    b(d, d_thr) = {
        -(d - d_thr)^2 * ln(d / d_thr),  if 0 < d < d_thr
        0,                               if d >= d_thr
    }
    
    Args:
        d: Distance tensor
        d_thr: Barrier threshold distance
        
    Returns:
        barrier_values: Barrier function values
    """
    device = d.device
    
    # Create mask for valid distance range (0 < d < d_thr)
    valid_mask = (d > 0) & (d < d_thr)
    
    # Initialize result tensor
    barrier_values = torch.zeros_like(d, device=device)
    
    # Compute barrier function only for valid distances
    if valid_mask.any():
        d_valid = d[valid_mask]
        d_ratio = d_valid / d_thr
        
        # Clamp d_ratio to prevent log(0) issues
        d_ratio = torch.clamp(d_ratio, min=1e-8, max=1.0-1e-8)
        
        # Compute barrier: -(d - d_thr)^2 * ln(d / d_thr)
        barrier_valid = -(d_valid - d_thr).pow(2) * torch.log(d_ratio)
        barrier_values[valid_mask] = barrier_valid
    
    return barrier_values


def extract_target_part_from_affordance(point_cloud_xyz, contact_map_values, contact_threshold=0.3):
    """
    Extract target part points from affordance grounding contact map.
    
    Args:
        point_cloud_xyz: Point cloud coordinates (N, 3)
        contact_map_values: Contact values from affordance map (N,)
        contact_threshold: Threshold for defining target part
        
    Returns:
        target_part_points: Points in target part
        target_mask: Boolean mask for target points
        non_target_mask: Boolean mask for non-target points
    """
    # Points with contact values above threshold are considered target part
    target_mask = contact_map_values > contact_threshold
    target_part_points = point_cloud_xyz[target_mask]
    return target_part_points, target_mask, ~target_mask


def compute_E_bar_affordance(hand_model, object_model, point_cloud_xyz, target_masks, non_target_masks, poses_num, d_thr=0.01):
    """
    Compute part-contact energy E_bar using affordance grounding contact map.
    
    Contact points should be close to target regions and far from non-target regions.
    For each 5000 poses, use different contact map's non_target_mask.
    
    Args:
        hand_model: HandModel instance
        object_model: ObjectModel instance
        point_cloud_xyz: Point cloud coordinates (N, 3)
        target_masks: Target masks for each contact map (5, N)
        non_target_masks: Non-target masks for each contact map (5, N)
        d_thr: Barrier threshold distance
        
    Returns:
        E_bar_total: Barrier energy for each batch element (batch_size,)
        target_part_info: Dictionary with target part statistics
    """
    batch_size = hand_model.contact_points.shape[0]
    device = hand_model.device
    n_contact_maps = 5
    
    # Convert numpy arrays to torch tensors
    point_cloud_xyz_torch = torch.tensor(point_cloud_xyz, dtype=torch.float, device=device)  # (N, 3)
    target_masks_torch = torch.tensor(target_masks, dtype=torch.bool, device=device)  # (5, N)
    non_target_masks_torch = torch.tensor(non_target_masks, dtype=torch.bool, device=device)  # (5, N)
    
    # Get fingertip positions (using contact points as proxy for fingertips)
    fingertips = hand_model.contact_points  # (batch_size, n_contact, 3)
    n_fingertips = fingertips.shape[1]
    
    # Initialize E_bar tensor
    E_bar_total = torch.zeros(batch_size, device=device)
    
    # Process each group of 5000 poses with different contact maps
    for contact_map_idx in range(n_contact_maps):
        start_idx = contact_map_idx * poses_num
        end_idx = min(start_idx + poses_num, batch_size)
        
        if start_idx >= batch_size:
            break
            
        current_batch_size = end_idx - start_idx
        
        # Get non-target points for this contact map
        non_target_mask = non_target_masks_torch[contact_map_idx]  # (N,)
        non_target_points = point_cloud_xyz_torch[non_target_mask]  # (n_non_target, 3)
        n_non_target = non_target_points.shape[0]
        
        if n_non_target == 0:
            continue
            
        # Get fingertips for current batch slice
        current_fingertips = fingertips[start_idx:end_idx]  # (current_batch_size, n_contact, 3)
        
        # Reshape for batch distance computation
        fingertips_reshaped = current_fingertips.reshape(-1, 3)  # (current_batch_size * n_contact, 3)
        
        # Compute distances from all fingertips to all non-target points
        distances = torch.cdist(fingertips_reshaped.unsqueeze(0), non_target_points.unsqueeze(0)).squeeze(0)  # (current_batch_size * n_contact, n_non_target)
        
        # Apply barrier function to encourage distance from non-target points
        barrier_values = barrier_function(distances, d_thr)  # (current_batch_size * n_contact, n_non_target)
        
        # Sum over non-target points and fingertips for each batch element
        barrier_per_fingertip = barrier_values.sum(dim=1)  # (current_batch_size * n_contact,)
        barrier_per_batch = barrier_per_fingertip.reshape(current_batch_size, n_fingertips).sum(dim=1)  # (current_batch_size,)
        
        # Assign to corresponding batch indices
        E_bar_total[start_idx:end_idx] = barrier_per_batch
    
    # Prepare target part information (using first contact map as representative)
    first_target_mask = target_masks[0]
    target_part_info = {
        'n_target_points': int(np.sum(first_target_mask)),
        'n_total_points': len(point_cloud_xyz),
        'target_ratio': float(np.sum(first_target_mask)) / len(point_cloud_xyz),
        'poses_per_contact_map': poses_num,
        'n_contact_maps': n_contact_maps
    }
    
    return E_bar_total, target_part_info


def compute_force_closure_energy(hand_model, object_model):
    """
    Compute force closure energy
    
    Args:
        hand_model: HandModel instance
        object_model: ObjectModel instance
        
    Returns:
        E_fc: Force closure energy
    """
    batch_size, n_contact, _ = hand_model.contact_points.shape
    device = hand_model.device
    
    distance, contact_normal = object_model.cal_distance(hand_model.contact_points)
    contact_normal = contact_normal.reshape(batch_size, 1, 3 * n_contact)

    transformation_matrix = torch.tensor([[0, 0, 0, 0, 0, -1, 0, 1, 0],
                                          [0, 0, 1, 0, 0, 0, -1, 0, 0],
                                          [0, -1, 0, 1, 0, 0, 0, 0, 0]], 
                                         dtype=torch.float, device=device)

    g = torch.cat([
        torch.eye(3, dtype=torch.float, device=device).expand(batch_size, n_contact, 3, 3).reshape(batch_size, 3 * n_contact, 3),
        (hand_model.contact_points @ transformation_matrix).view(batch_size, 3 * n_contact, 3)
    ], dim=2).float().to(device)
    
    norm = torch.norm(contact_normal @ g, dim=[1, 2])
    return norm * norm


def compute_hand_surface_normals(hand_model):
    """
    Compute surface normals for each contact point on the hand meshes.
    For each contact point, find the nearest face and return its normal vector.
    
    Args:
        hand_model: HandModel instance
        
    Returns:
        hand_normals: (batch_size, n_contact, 3) tensor of hand surface normals
    """
    batch_size, n_contact, _ = hand_model.contact_points.shape
    device = hand_model.device
    
    # Initialize tensor to store hand normals
    hand_normals = torch.zeros(batch_size, n_contact, 3, device=device)
    
    # Get contact points in local coordinates for each link
    contact_points_local = hand_model.contact_candidates[hand_model.contact_point_indices]  # (batch_size, n_contact, 3)
    
    for batch_idx in range(batch_size):
        for contact_idx in range(n_contact):
            # Get the link index for this contact point
            global_contact_idx = hand_model.contact_point_indices[batch_idx, contact_idx]
            link_idx = hand_model.global_index_to_link_index[global_contact_idx]
            link_name = list(hand_model.mesh.keys())[link_idx]
            
            # Get mesh data for this link
            vertices = hand_model.mesh[link_name]['vertices']
            faces = hand_model.mesh[link_name]['faces']
            
            # Get contact point in local coordinates
            contact_point = contact_points_local[batch_idx, contact_idx]
            
            # Compute face normals for all faces in this link
            face_vertices = vertices[faces]  # (n_faces, 3, 3)
            
            # Compute normal vectors using cross product
            v1 = face_vertices[:, 1] - face_vertices[:, 0]  # (n_faces, 3)
            v2 = face_vertices[:, 2] - face_vertices[:, 0]  # (n_faces, 3)
            face_normals = torch.cross(v1, v2, dim=1)  # (n_faces, 3)
            face_normals = face_normals / (torch.norm(face_normals, dim=1, keepdim=True) + 1e-8)
            
            # Find closest face to the contact point
            face_centers = face_vertices.mean(dim=1)  # (n_faces, 3)
            distances_to_faces = torch.norm(face_centers - contact_point.unsqueeze(0), dim=1)
            closest_face_idx = torch.argmin(distances_to_faces)
            
            # Get the normal of the closest face and ensure it points outward (front side)
            face_normal = face_normals[closest_face_idx]
            
            # Ensure normal points outward from the hand (front side)
            hand_normals[batch_idx, contact_idx] = face_normal
    
    return hand_normals


def compute_E_dir(hand_model, object_model):
    """
    Compute E_dir energy function for contact normal alignment.
    
    E_dir = sum over contact points of (1 - ci · Ni)
    where:
    - ci: contact normal vector on object surface pointing inwards into object
    - Ni: normal vector of front surface of contact link, pointing outwards from hand
    
    Args:
        hand_model: HandModel instance
        object_model: ObjectModel instance
        
    Returns:
        E_dir: (batch_size,) tensor of directional alignment energies
    """
    batch_size, n_contact, _ = hand_model.contact_points.shape
    
    # Get object surface normals at contact points (ci)
    # These point inward into the object
    _, object_contact_normals = object_model.cal_distance(hand_model.contact_points)
    object_contact_normals = object_contact_normals.reshape(batch_size, n_contact, 3)
    
    # Get hand surface normals at contact points (Ni)
    # These should point outward from the hand (front side)
    hand_surface_normals = compute_hand_surface_normals(hand_model)
    
    # Compute cosine similarity for each contact point
    # ci · Ni = dot product between object normal (inward) and hand normal (outward)
    cosine_similarities = torch.sum(object_contact_normals * hand_surface_normals, dim=2)  # (batch_size, n_contact)
    
    # E_dir = sum over all contact points of (1 - ci · Ni)
    # This encourages alignment between inward object normals and outward hand normals
    E_dir = torch.sum(1.0 - cosine_similarities, dim=1)  # (batch_size,)
    
    return E_dir


def compute_regularization_energies(hand_model, object_model):
    """
    Compute regularization energies
    
    Args:
        hand_model: HandModel instance
        object_model: ObjectModel instance
        
    Returns:
        E_joints: Joint limit penalty
        E_pen: Penetration penalty
        E_spen: Self-penetration penalty
    """
    # Joint limit penalty
    E_joints = torch.sum((hand_model.hand_pose[:, 9:] > hand_model.joints_upper) * (hand_model.hand_pose[:, 9:] - hand_model.joints_upper), dim=-1) + \
               torch.sum((hand_model.hand_pose[:, 9:] < hand_model.joints_lower) * (hand_model.joints_lower - hand_model.hand_pose[:, 9:]), dim=-1)

    # Penetration penalty
    object_scale = object_model.object_scale_tensor.flatten().unsqueeze(1).unsqueeze(2)
    object_surface_points = object_model.surface_points_tensor * object_scale
    distances = hand_model.cal_distance(object_surface_points)
    distances[distances <= 0] = 0
    E_pen = distances.sum(-1)

    # Self-penetration penalty
    E_spen = hand_model.self_penetration()
    
    return E_joints, E_pen, E_spen


def compute_total_energy_for_annealing(hand_model, object_model, xyz, contact_vals, target_masks, non_target_masks, config):
    """
    Compute total energy function for annealing optimization
    
    Args:
        hand_model: HandModel instance
        object_model: ObjectModel instance
        xyz: Point cloud coordinates
        contact_vals: Contact values from affordance map
        config: Configuration object with energy weights
        
    Returns:
        energy: Total energy
        E_fc: Force closure energy
        E_dis: Distance energy
        E_pen: Penetration energy
        E_spen: Self-penetration energy
        E_joints: Joint limit energy
        E_bar: Barrier energy
        E_dir: Direction alignment energy
    """
    # Distance Energy
    E_dis = compute_distance_energy(hand_model, object_model, d0=0.01)
    
    # # Barrier Energy 
    E_bar, _ = compute_E_bar_affordance(
        hand_model, object_model, xyz, target_masks, non_target_masks,
        poses_num = config.poses_per_contact,
        d_thr=config.barrier_threshold
    )
    
    # Force Closure Energy
    E_fc = compute_force_closure_energy(hand_model, object_model)
    
    # # Direction Alignment Energy
    E_dir = compute_E_dir(hand_model, object_model)
    
    # # Regularization Energies
    E_joints, E_pen, E_spen = compute_regularization_energies(hand_model, object_model)
    
    # # Total energy (Equation 4)
    energy = (config.w_fc * E_fc + 
             config.w_dis * E_dis + 
             config.w_pen * E_pen + 
             config.w_spen * E_spen + 
             config.w_joints * E_joints + 
             config.w_bar * E_bar + 
             config.w_dir * E_dir)
    
    return energy, E_fc, E_dis, E_pen, E_spen, E_joints, E_bar, E_dir