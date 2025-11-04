"""
Last modified date: 2023.02.23
Author: Ruicheng Wang, Jialiang Zhang
Description: Class ObjectModel
"""

import os
import trimesh as tm
import plotly.graph_objects as go
import torch
import pytorch3d.structures
import pytorch3d.ops
import numpy as np

from torchsdf import index_vertices_by_faces, compute_sdf


class ObjectModel:

    def __init__(self, data_root_path, batch_size_each, num_samples=2000, device="cuda"):
        """
        Create a Object Model
        
        Parameters
        ----------
        data_root_path: str
            directory to object meshes
        batch_size_each: int
            batch size for each objects
        num_samples: int
            numbers of object surface points, sampled with fps
        device: str | torch.Device
            device for torch tensors
        """

        self.device = device
        self.batch_size_each = batch_size_each
        self.data_root_path = data_root_path
        self.num_samples = num_samples

        self.object_code_list = None
        self.object_scale_tensor = None
        self.object_mesh_list = None
        self.object_face_verts_list = None
        
        # Affordance data attributes
        self.affordance_xyz = None
        self.affordance_contact_maps = None
        self.affordance_target_masks = None
        self.affordance_non_target_masks = None
        # self.scale_choice = torch.tensor([0.06, 0.08, 0.1, 0.12, 0.15], dtype=torch.float, device=self.device)
        self.scale_choice = torch.tensor([1],dtype=torch.float, device=self.device )
    def initialize(self, object_code_list, affordance_data_path=None, contact_threshold=0.5):
        """
        Initialize Object Model with list of objects
        
        Choose scales, load meshes, sample surface points, and optionally load affordance data
        
        Parameters
        ----------
        object_code_list: list | str
            list of object codes
        affordance_data_path: str, optional
            path to .npy file containing point cloud and contact maps
        contact_threshold: float, optional
            threshold for defining target/non-target masks (default: 0.5)
        """
        if not isinstance(object_code_list, list):
            object_code_list = [object_code_list]
        self.object_code_list = object_code_list
        self.object_scale_tensor = []
        self.object_mesh_list = []
        self.object_face_verts_list = []
        self.surface_points_tensor = []
        for object_code in object_code_list:
            self.object_scale_tensor.append(self.scale_choice[torch.randint(0, self.scale_choice.shape[0], (self.batch_size_each, ), device=self.device)])
            self.object_mesh_list.append(tm.load(os.path.join(self.data_root_path, object_code, "coacd", "decomposed.obj"), force="mesh", process=False))
            object_verts = torch.Tensor(self.object_mesh_list[-1].vertices).to(self.device)
            object_faces = torch.Tensor(self.object_mesh_list[-1].faces).long().to(self.device)
            self.object_face_verts_list.append(index_vertices_by_faces(object_verts, object_faces))
            if self.num_samples != 0:
                vertices = torch.tensor(self.object_mesh_list[-1].vertices, dtype=torch.float, device=self.device)
                faces = torch.tensor(self.object_mesh_list[-1].faces, dtype=torch.float, device=self.device)
                mesh = pytorch3d.structures.Meshes(vertices.unsqueeze(0), faces.unsqueeze(0))
                dense_point_cloud = pytorch3d.ops.sample_points_from_meshes(mesh, num_samples=100 * self.num_samples)
                surface_points = pytorch3d.ops.sample_farthest_points(dense_point_cloud, K=self.num_samples)[0][0]
                surface_points.to(dtype=float, device=self.device)
                self.surface_points_tensor.append(surface_points)
        self.object_scale_tensor = torch.stack(self.object_scale_tensor, dim=0)
        if self.num_samples != 0:
            self.surface_points_tensor = torch.stack(self.surface_points_tensor, dim=0).repeat_interleave(self.batch_size_each, dim=0)  # (n_objects * batch_size_each, num_samples, 3)
        
        # Load affordance data if provided
        if affordance_data_path is not None:
            self._load_affordance_data(affordance_data_path, contact_threshold)

    def _load_affordance_data(self, data_path, contact_threshold=0.5, distance_threshold=0.005):
        """
        Load point cloud and contact map from processed DexGYS dataset

        ----------
        data_path: str
            path to .npy file containing point cloud and contact maps
        contact_threshold: float
            threshold for defining target/non-target masks
        distance_threshold: float
            distance threshold for mapping contact regions to surface points (default: 0.5cm)
        """
        print(f"Loading affordance data from: {data_path}")
        
        point_cloud = np.load(data_path)
        xyz = point_cloud[:, :3]
        contact_maps = point_cloud[:, 3:8].T
        
        # Store original affordance data
        self.affordance_xyz = torch.tensor(xyz, dtype=torch.float, device=self.device)
        self.affordance_contact_maps = torch.tensor(contact_maps, dtype=torch.float, device=self.device)
        
        # Generate affordance masks for surface points using distance-based mapping
        print(f"Mapping affordance to surface points with distance threshold: {distance_threshold}")
        
        # Get surface points (first object)
        surface_points = self.surface_points_tensor[0]  # (num_samples, 3)
        n_surface = surface_points.shape[0]
        
        # Initialize masks for surface points
        surface_target_masks = torch.zeros(5, n_surface, dtype=torch.bool, device=self.device)
        
        # Process each contact map
        for i in range(5):
            # Find contact points above threshold
            target_mask = contact_maps[i] > contact_threshold
            if np.any(target_mask):
                target_points = torch.tensor(xyz[target_mask], dtype=torch.float, device=self.device)
                
                # Compute distances from surface points to target points
                distances = torch.cdist(surface_points.unsqueeze(0), target_points.unsqueeze(0)).squeeze(0)
                min_distances, _ = torch.min(distances, dim=1)
                
                # Mark surface points within distance threshold as targets
                surface_target_masks[i] = min_distances < distance_threshold
            
            n_target = surface_target_masks[i].sum().item()

        self.affordance_target_masks = surface_target_masks  # (5, num_samples)
        self.affordance_non_target_masks = ~surface_target_masks  # (5, num_samples)
    
    def _visualize_affordance_mapping(self, surface_points, surface_target_masks, data_path):
        """
        Visualize surface points colored by contact maps
        
        Parameters
        ----------
        surface_points: torch.Tensor (num_samples, 3)
            surface points coordinates
        surface_target_masks: torch.Tensor (5, num_samples)  
            target masks for each contact map
        data_path: str
            path to original data file for naming output
        """
        import os
        
        # Convert to numpy for visualization
        surface_points_np = surface_points.detach().cpu().numpy()
        
        # Create output directory
        base_name = os.path.splitext(os.path.basename(data_path))[0]
        output_dir = os.path.join(os.path.dirname(data_path), f"{base_name}_visualization")
        os.makedirs(output_dir, exist_ok=True)
        
        # Color palette for different contact maps
        colors = ['red', 'green', 'blue', 'orange', 'purple']
        
        # Create visualization for each contact map
        for i in range(5):
            target_mask_np = surface_target_masks[i].detach().cpu().numpy()
            
            # Create plotly figure
            fig = go.Figure()
            
            # Add non-target points (gray)
            non_target_indices = ~target_mask_np
            if np.any(non_target_indices):
                fig.add_trace(go.Scatter3d(
                    x=surface_points_np[non_target_indices, 0],
                    y=surface_points_np[non_target_indices, 1], 
                    z=surface_points_np[non_target_indices, 2],
                    mode='markers',
                    marker=dict(color='lightgray', size=2),
                    name=f'Non-target points',
                    showlegend=True
                ))
            
            # Add target points (colored)
            if np.any(target_mask_np):
                fig.add_trace(go.Scatter3d(
                    x=surface_points_np[target_mask_np, 0],
                    y=surface_points_np[target_mask_np, 1],
                    z=surface_points_np[target_mask_np, 2],
                    mode='markers',
                    marker=dict(color=colors[i], size=4),
                    name=f'Contact map {i} targets',
                    showlegend=True
                ))
            
            # Calculate bounds for better camera positioning
            x_min, x_max = surface_points_np[:, 0].min(), surface_points_np[:, 0].max()
            y_min, y_max = surface_points_np[:, 1].min(), surface_points_np[:, 1].max()
            z_min, z_max = surface_points_np[:, 2].min(), surface_points_np[:, 2].max()
            
            # Calculate center and range
            x_center, y_center, z_center = (x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2
            max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
            
            # Set layout with optimized camera view
            fig.update_layout(
                title=f'Surface Points - Contact Map {i} (Target: {target_mask_np.sum()}/{len(target_mask_np)})',
                scene=dict(
                    xaxis=dict(
                        title='X',
                        range=[x_center - max_range*0.6, x_center + max_range*0.6],
                        showgrid=True,
                        gridwidth=1,
                        gridcolor="LightGray"
                    ),
                    yaxis=dict(
                        title='Y', 
                        range=[y_center - max_range*0.6, y_center + max_range*0.6],
                        showgrid=True,
                        gridwidth=1,
                        gridcolor="LightGray"
                    ),
                    zaxis=dict(
                        title='Z',
                        range=[z_center - max_range*0.6, z_center + max_range*0.6],
                        showgrid=True,
                        gridwidth=1,
                        gridcolor="LightGray"
                    ),
                    aspectmode='cube',
                    camera=dict(
                        eye=dict(x=1.5, y=1.5, z=1.5),
                        center=dict(x=0, y=0, z=0),
                        up=dict(x=0, y=0, z=1)
                    ),
                    bgcolor='white'
                ),
                width=1000,
                height=800,
                margin=dict(l=0, r=0, t=50, b=0)
            )
            
            # Save visualization as PNG
            output_file = os.path.join(output_dir, f"contact_map_{i}.png")
            try:
                fig.write_image(output_file, width=1000, height=800)
                print(f"Saved visualization: {output_file}")
            except Exception as e:
                print(f"Warning: Could not save PNG (missing kaleido?). Saving as HTML instead.")
                output_file_html = os.path.join(output_dir, f"contact_map_{i}.html")
                fig.write_html(output_file_html)
                print(f"Saved HTML visualization: {output_file_html}")
        
        # Create combined visualization showing all contact maps
        fig_combined = go.Figure()
        
        # Add surface points colored by contact map assignment
        surface_colors = np.full(len(surface_points_np), -1, dtype=int)  # -1 for non-target
        
        for i in range(5):
            target_mask_np = surface_target_masks[i].detach().cpu().numpy()
            surface_colors[target_mask_np] = i
        
        # Add points for each color
        for i in range(-1, 5):
            if i == -1:
                # Non-target points
                mask = surface_colors == -1
                color = 'lightgray'
                name = 'Non-target'
                size = 2
            else:
                # Target points for contact map i
                mask = surface_colors == i
                color = colors[i]
                name = f'Contact map {i}'
                size = 4
            
            if np.any(mask):
                fig_combined.add_trace(go.Scatter3d(
                    x=surface_points_np[mask, 0],
                    y=surface_points_np[mask, 1],
                    z=surface_points_np[mask, 2],
                    mode='markers',
                    marker=dict(color=color, size=size),
                    name=name,
                    showlegend=True
                ))
        
        # Set layout for combined figure with optimized camera view
        fig_combined.update_layout(
            title='Surface Points - All Contact Maps Combined',
            scene=dict(
                xaxis=dict(
                    title='X',
                    range=[x_center - max_range*0.6, x_center + max_range*0.6],
                    showgrid=True,
                    gridwidth=1,
                    gridcolor="LightGray"
                ),
                yaxis=dict(
                    title='Y', 
                    range=[y_center - max_range*0.6, y_center + max_range*0.6],
                    showgrid=True,
                    gridwidth=1,
                    gridcolor="LightGray"
                ),
                zaxis=dict(
                    title='Z',
                    range=[z_center - max_range*0.6, z_center + max_range*0.6],
                    showgrid=True,
                    gridwidth=1,
                    gridcolor="LightGray"
                ),
                aspectmode='cube',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5),
                    center=dict(x=0, y=0, z=0),
                    up=dict(x=0, y=0, z=1)
                ),
                bgcolor='white'
            ),
            width=1200,
            height=900,
            margin=dict(l=0, r=0, t=50, b=0)
        )
        
        # Save combined visualization as PNG
        combined_output = os.path.join(output_dir, "all_contact_maps.png")
        try:
            fig_combined.write_image(combined_output, width=1200, height=900)
            print(f"Saved combined visualization: {combined_output}")
        except Exception as e:
            print(f"Warning: Could not save PNG (missing kaleido?). Saving as HTML instead.")
            combined_output_html = os.path.join(output_dir, "all_contact_maps.html")
            fig_combined.write_html(combined_output_html)
            print(f"Saved HTML visualization: {combined_output_html}")
            

    def cal_distance(self, x, with_closest_points=False):
        """
        Calculate signed distances from hand contact points to object meshes and return contact normals
        
        Interiors are positive, exteriors are negative
        
        Use our modified Kaolin package
        
        Parameters
        ----------
        x: (B, `n_contact`, 3) torch.Tensor
            hand contact points
        with_closest_points: bool
            whether to return closest points on object meshes
        
        Returns
        -------
        distance: (B, `n_contact`) torch.Tensor
            signed distances from hand contact points to object meshes, inside is positive
        normals: (B, `n_contact`, 3) torch.Tensor
            contact normal vectors defined by gradient
        closest_points: (B, `n_contact`, 3) torch.Tensor
            contact points on object meshes, returned only when `with_closest_points is True`
        """
        _, n_points, _ = x.shape
        x = x.reshape(-1, self.batch_size_each * n_points, 3)
        distance = []
        normals = []
        closest_points = []
        scale = self.object_scale_tensor.repeat_interleave(n_points, dim=1)
        x = x / scale.unsqueeze(2)
        for i in range(len(self.object_mesh_list)):
            face_verts = self.object_face_verts_list[i]
            dis, dis_signs, normal, _ = compute_sdf(x[i], face_verts)
            if with_closest_points:
                closest_points.append(x[i] - dis.sqrt().unsqueeze(1) * normal)
            dis = torch.sqrt(dis + 1e-8)
            dis = dis * (-dis_signs)
            distance.append(dis)
            normals.append(normal * dis_signs.unsqueeze(1))
        distance = torch.stack(distance)
        normals = torch.stack(normals)
        distance = distance * scale
        distance = distance.reshape(-1, n_points)
        normals = normals.reshape(-1, n_points, 3)
        if with_closest_points:
            closest_points = (torch.stack(closest_points) * scale.unsqueeze(2)).reshape(-1, n_points, 3)
            return distance, normals, closest_points
        return distance, normals

    def get_plotly_data(self, i, color='lightgreen', opacity=0.5, pose=None):
        """
        Get visualization data for plotly.graph_objects
        
        Parameters
        ----------
        i: int
            index of data
        color: str
            color of mesh
        opacity: float
            opacity
        pose: (4, 4) matrix
            homogeneous transformation matrix
        
        Returns
        -------
        data: list
            list of plotly.graph_object visualization data
        """
        model_index = i // self.batch_size_each
        model_scale = self.object_scale_tensor[model_index, i % self.batch_size_each].detach().cpu().numpy()
        mesh = self.object_mesh_list[model_index]
        vertices = mesh.vertices * model_scale
        if pose is not None:
            pose = np.array(pose, dtype=np.float32)
            vertices = vertices @ pose[:3, :3].T + pose[:3, 3]
        data = go.Mesh3d(x=vertices[:, 0],y=vertices[:, 1], z=vertices[:, 2], i=mesh.faces[:, 0], j=mesh.faces[:, 1], k=mesh.faces[:, 2], color=color, opacity=opacity)
        return [data]
