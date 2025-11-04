"""
Visualization script for initial grasp poses using initialize_hand_with_contact_normal function
Creates 100 initial poses and visualizes them using plotly
"""

import os
import sys
import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import argparse
from types import SimpleNamespace

# Add utils path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from utils.hand_model import HandModel
from utils.object_model import ObjectModel
from utils.initialization import extract_contact_normal, initialize_hand_with_contact_normal


class Config:
    """Configuration class for initialization parameters"""
    def __init__(self):
        self.poses_per_contact = 20  # 20 poses per contact normal (5 normals * 20 = 100 total)
        self.distance_lower = 0.8
        self.distance_upper = 1.2
        self.theta_lower = -np.pi / 6
        self.theta_upper = np.pi / 6
        self.jitter_strength = 0.1
        self.n_contact = 4


def load_object_with_affordance(object_code, data_root_path, device='cuda'):
    """Load a single object with affordance data"""
    object_model = ObjectModel(data_root_path, batch_size_each=1, device=device)
    
    # Load single object
    object_model.initialize([object_code])
    
    # Load affordance data if available
    affordance_path = os.path.join(data_root_path, object_code, 'affordance_data.npz')
    if os.path.exists(affordance_path):
        affordance_data = np.load(affordance_path)
        object_model.affordance_xyz = torch.tensor(affordance_data['xyz'], device=device)
        object_model.affordance_contact_maps = torch.tensor(affordance_data['contact_maps'], device=device)
        print(f"Loaded affordance data: xyz shape {object_model.affordance_xyz.shape}, contact_maps shape {object_model.affordance_contact_maps.shape}")
    else:
        # Generate dummy affordance data for demonstration
        print(f"No affordance data found at {affordance_path}, generating dummy data")
        mesh = object_model.object_mesh_list[0]
        n_points = 1000
        surface_points = mesh.sample(n_points)
        object_model.affordance_xyz = torch.tensor(surface_points, dtype=torch.float, device=device)
        # Create 5 random contact maps
        contact_maps = np.random.rand(5, n_points) * 0.8 + 0.1  # values between 0.1 and 0.9
        object_model.affordance_contact_maps = torch.tensor(contact_maps, dtype=torch.float, device=device)
    
    return object_model


def visualize_initial_poses_plotly(hand_model, object_model, hand_poses, contact_points, contact_normals):
    """Visualize 100 initial poses using plotly"""
    
    # Convert to numpy
    if isinstance(hand_poses, torch.Tensor):
        hand_poses = hand_poses.detach().cpu().numpy()
    if isinstance(contact_points, torch.Tensor):
        contact_points = contact_points.detach().cpu().numpy()
    if isinstance(contact_normals, torch.Tensor):
        contact_normals = contact_normals.detach().cpu().numpy()
    
    # Extract translations (first 3 columns of hand_poses)
    translations = hand_poses[:, :3]
    
    # Get object mesh
    object_mesh = object_model.object_mesh_list[0]
    object_vertices = object_mesh.vertices
    object_faces = object_mesh.faces
    object_scale = object_model.object_scale_tensor[0].cpu().numpy()
    object_vertices_scaled = object_vertices * object_scale
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}],
               [{"type": "xy"}, {"type": "xy"}]],
        subplot_titles=("All 100 Initial Hand Poses", "Poses Grouped by Contact Normal", 
                       "Distance Distribution", "Height Distribution"),
        vertical_spacing=0.1
    )
    
    # Plot 1: All poses overview (3D)
    # Object mesh
    fig.add_trace(
        go.Mesh3d(
            x=object_vertices_scaled[:, 0],
            y=object_vertices_scaled[:, 1], 
            z=object_vertices_scaled[:, 2],
            i=object_faces[:, 0],
            j=object_faces[:, 1],
            k=object_faces[:, 2],
            color='lightgray',
            opacity=0.3,
            name='Object Mesh'
        ),
        row=1, col=1
    )
    
    # Contact points
    fig.add_trace(
        go.Scatter3d(
            x=contact_points[:, 0],
            y=contact_points[:, 1],
            z=contact_points[:, 2],
            mode='markers',
            marker=dict(size=8, color='red'),
            name='Contact Points'
        ),
        row=1, col=1
    )
    
    # Contact normals as arrows
    for i in range(len(contact_points)):
        start_point = contact_points[i]
        end_point = start_point + 0.1 * contact_normals[i]
        fig.add_trace(
            go.Scatter3d(
                x=[start_point[0], end_point[0]],
                y=[start_point[1], end_point[1]],
                z=[start_point[2], end_point[2]],
                mode='lines',
                line=dict(color='red', width=3),
                showlegend=False
            ),
            row=1, col=1
        )
    
    # All hand positions
    fig.add_trace(
        go.Scatter3d(
            x=translations[:, 0],
            y=translations[:, 1],
            z=translations[:, 2],
            mode='markers',
            marker=dict(size=4, color='blue', opacity=0.6),
            name='Hand Poses'
        ),
        row=1, col=1
    )
    
    # Plot 2: Poses by contact normal (3D)
    # Object mesh
    fig.add_trace(
        go.Mesh3d(
            x=object_vertices_scaled[:, 0],
            y=object_vertices_scaled[:, 1], 
            z=object_vertices_scaled[:, 2],
            i=object_faces[:, 0],
            j=object_faces[:, 1],
            k=object_faces[:, 2],
            color='lightgray',
            opacity=0.3,
            showlegend=False
        ),
        row=1, col=2
    )
    
    # Poses grouped by contact normal
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for i in range(5):
        start_idx = i * 20
        end_idx = (i + 1) * 20
        fig.add_trace(
            go.Scatter3d(
                x=translations[start_idx:end_idx, 0],
                y=translations[start_idx:end_idx, 1],
                z=translations[start_idx:end_idx, 2],
                mode='markers',
                marker=dict(size=5, color=colors[i]),
                name=f'Contact {i+1}'
            ),
            row=1, col=2
        )
    
    # Plot 3: Distance distribution
    object_center = object_vertices_scaled.mean(axis=0)
    distances = np.linalg.norm(translations - object_center, axis=1)
    
    fig.add_trace(
        go.Histogram(
            x=distances,
            nbinsx=20,
            opacity=0.7,
            marker_color='blue',
            name='Distance Distribution'
        ),
        row=2, col=1
    )
    
    # Plot 4: Height distribution
    heights = translations[:, 2]  # Z-coordinate
    
    fig.add_trace(
        go.Histogram(
            x=heights,
            nbinsx=20,
            opacity=0.7,
            marker_color='green',
            name='Height Distribution'
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title="Initial Grasp Poses Visualization (100 poses)",
        height=800,
        showlegend=True
    )
    
    # Update 3D scene layouts
    scene_layout = dict(
        xaxis_title="X",
        yaxis_title="Y", 
        zaxis_title="Z",
        aspectmode='cube'
    )
    
    fig.update_layout(scene=scene_layout, scene2=scene_layout)
    
    # Update 2D plot layouts
    fig.update_xaxes(title_text="Distance from Object Center", row=2, col=1)
    fig.update_yaxes(title_text="Frequency", row=2, col=1)
    fig.update_xaxes(title_text="Height (Z-coordinate)", row=2, col=2)
    fig.update_yaxes(title_text="Frequency", row=2, col=2)
    
    # Save and show
    fig.write_image("initial_poses_visualization.png")
    fig.show()
    
    # Print statistics
    print(f"\n=== POSE STATISTICS ===")
    print(f"Total poses generated: {len(translations)}")
    print(f"Distance range: {distances.min():.3f} - {distances.max():.3f}")
    print(f"Mean distance: {distances.mean():.3f}")
    print(f"Height range: {heights.min():.3f} - {heights.max():.3f}")
    print(f"Mean height: {heights.mean():.3f}")
    print(f"Pose distribution per contact normal: 20 poses each")
    print(f"Visualization saved as 'initial_poses_visualization.html'")


def main():
    parser = argparse.ArgumentParser(description='Visualize initial grasp poses')
    parser.add_argument('--object_code', type=str, default='sem-Car-2f28e2bd754977da8cfac9da0ff28f62',
                        help='Object code to visualize')
    parser.add_argument('--data_root_path', type=str, default='/home/dmsgk724/CVPR_2026/grasping/Grasp-as-You-Say/asset_process/data/meshdata',
                        help='Path to meshdata directory')
    parser.add_argument('--hand_model_path', type=str, default='/home/dmsgk724/CVPR_2026/grasping/DexGraspNet/grasp_generation/mjcf',
                        help='Path to hand model assets')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                        help='Device to use')
    
    args = parser.parse_args()
    
    print(f"Visualizing initial poses for object: {args.object_code}")
    
    # Check if CUDA is available
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    try:
        # Load hand model
        print("Loading hand model...")
        mjcf_path = os.path.join(args.hand_model_path, 'shadow_hand_wrist_free.xml')
        mesh_path = os.path.join(args.hand_model_path, 'meshes')
        contact_points_path = os.path.join(args.hand_model_path, 'contact_points.json')
        penetration_points_path = os.path.join(args.hand_model_path, 'penetration_points.json')
        
        hand_model = HandModel(
            mjcf_path=mjcf_path,
            mesh_path=mesh_path,
            contact_points_path=contact_points_path,
            penetration_points_path=penetration_points_path,
            device=args.device
        )
        print(f"Hand model loaded with {hand_model.n_dofs} DOFs")
        
        # Load object model
        print("Loading object model...")
        object_model = load_object_with_affordance(args.object_code, args.data_root_path, args.device)
        print("Object model loaded")
        
        # Extract contact normals
        print("Extracting contact normals...")
        closest_points, contact_normals, closest_indices = extract_contact_normal(object_model)
        print(f"Extracted {len(closest_points)} contact points and normals")
        
        # Create configuration
        config = Config()
        
        # Initialize hand poses
        print("Initializing hand poses...")
        initialize_hand_with_contact_normal(hand_model, object_model, closest_points, contact_normals, config)
        
        # Get the generated poses
        hand_poses = hand_model.hand_pose.detach().cpu()
        print(f"Generated {hand_poses.shape[0]} poses")
        
        # Visualize
        print("Creating visualization...")
        visualize_initial_poses_plotly(hand_model, object_model, hand_poses, closest_points, contact_normals)
        
        print("Visualization complete! Check 'initial_poses_visualization.html'")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()