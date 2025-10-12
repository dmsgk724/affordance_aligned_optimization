#!/usr/bin/env python3
"""
Affordance-Aligned Grasping Optimization Script

This script implements the optimization pipeline from affordance_aligned_grasping.ipynb
with new energy functions for distance, barrier, and direction alignment.

Usage:
    python affordance_aligned_optimization.py --object_code s20349 --grasp_idx 0
"""

import argparse
import math
import numpy as np
import torch
from pathlib import Path
import pickle
import wandb
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go

# Import DexGraspNet modules
from utils.hand_model import HandModel
from utils.object_model import ObjectModel
from utils.optimizer import Annealing
from utils.initialization import extract_contact_normal, initialize_hand_with_contact_normal, load_affordance_data
from utils.affordance_energy import compute_total_energy_for_annealing


class AffordanceGraspConfig:
    """Configuration for affordance-aligned grasping optimization"""
    
    def __init__(self):
        # Distance parameters for hand positioning
        self.distance_lower = 0.2
        self.distance_upper = 0.3
        self.theta_lower = -math.pi / 6
        self.theta_upper = math.pi / 6
        self.jitter_strength = 0.1
        self.n_contact = 4
        
        # Energy weights (following generate_grasps.py)
        self.w_dis = 100.0      # Original distance weight
        self.w_pen = 100.0      # Penetration weight  
        self.w_spen = 10.0      # Self-penetration weight
        self.w_joints = 1.0     # Joint limits weight
        
        self.w_bar = 1.0        # Barrier energy weight
        self.w_dir = 10.0       # Direction alignment weight
        self.w_fc = 1.0         # Force closure weight
        
        # Annealing parameters (same as generate_grasps.py)
        self.switch_possibility = 0.5
        self.starting_temperature = 18.0
        self.temperature_decay = 0.95
        self.annealing_period = 30
        self.step_size = 0.005
        self.stepsize_period = 50
        self.mu = 0.98
        
        # Affordance parameters
        self.contact_threshold = 0.5  # Threshold for contact map
        self.barrier_threshold = 0.02  # d_thr for barrier function


class AffordanceAlignedOptimizer:
    """Main optimizer class for affordance-aligned grasping"""
    
    def __init__(self, config, device='cuda', use_wandb=True):
        self.config = config
        self.device = device
        self.hand_model = None
        self.object_model = None
        self.use_wandb = use_wandb
        
    def initialize_models(self, object_code, data_root_path):
        """Initialize hand and object models"""
        print("Initializing models...")
        
        # Initialize hand model
        self.hand_model = HandModel(
            mjcf_path='mjcf/shadow_hand_wrist_free.xml',
            mesh_path='mjcf/meshes',
            contact_points_path='mjcf/contact_points.json',
            penetration_points_path='mjcf/penetration_points.json',
            device=self.device
        )
        
        # Initialize object model
        self.object_model = ObjectModel(
            data_root_path=data_root_path,
            batch_size_each=1,
            num_samples=2000,
            device=self.device
        )
        
        self.object_model.initialize([object_code])
        print("Models initialized successfully!")
        
    def optimize_grasp_annealing(self, xyz, contact_vals, num_steps=6000, verbose_step=100, output_dir=None, object_code=None, grasp_idx=None, closest_point=None, contact_normal=None, closest_idx=None):
        """Optimize grasp using Annealing optimizer"""
        
        # Store initial pose
        hand_pose_st = self.hand_model.hand_pose.detach()
        
        # Configure annealing optimizer
        optim_config = {
            'switch_possibility': self.config.switch_possibility,
            'starting_temperature': self.config.starting_temperature,
            'temperature_decay': self.config.temperature_decay,
            'annealing_period': self.config.annealing_period,
            'step_size': self.config.step_size,
            'stepsize_period': self.config.stepsize_period,
            'mu': self.config.mu,
            'device': self.hand_model.device
        }
        optimizer = Annealing(self.hand_model, **optim_config)
        
        
        print(f"=== Starting Annealing Optimization ===")
        print(f"Steps: {num_steps}")
        print(f"Annealing Config: temp={self.config.starting_temperature}, decay={self.config.temperature_decay}")
        
        # Initial energy computation
        energy, E_fc, E_dis, E_pen, E_spen, E_joints, E_bar, E_dir = compute_total_energy_for_annealing(
            self.hand_model, self.object_model, xyz, contact_vals, self.config
        )
        
        print(f"Initial Total Energy: {energy.sum().item():.6f}")
        
        # Backward pass for initial gradients
        energy.sum().backward(retain_graph=True)
        
        for step in range(1, num_steps + 1):
            # Try step (annealing)
            s = optimizer.try_step()
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Compute new energy
            new_energy, new_E_fc, new_E_dis, new_E_pen, new_E_spen, new_E_joints, new_E_bar, new_E_dir = compute_total_energy_for_annealing(
                self.hand_model, self.object_model, xyz, contact_vals, self.config
            )
            
            # Backward pass
            new_energy.sum().backward(retain_graph=True)
            
            # Accept/reject step
            with torch.no_grad():
                accept, temperature = optimizer.accept_step(energy, new_energy)
                
                # Update energies for accepted steps
                energy[accept] = new_energy[accept]
                E_fc[accept] = new_E_fc[accept]
                E_dis[accept] = new_E_dis[accept]
                E_pen[accept] = new_E_pen[accept]
                E_spen[accept] = new_E_spen[accept]
                E_joints[accept] = new_E_joints[accept]
                E_bar[accept] = new_E_bar[accept]
                E_dir[accept] = new_E_dir[accept]
            
            
            # Wandb logging
            if self.use_wandb:
                wandb.log({
                    'step': step,
                    'energy/total': energy.mean().item(),
                    'energy/force_closure': E_fc.mean().item(),
                    'energy/distance': E_dis.mean().item(),
                    'energy/penetration': E_pen.mean().item(),
                    'energy/self_penetration': E_spen.mean().item(),
                    'energy/joints': E_joints.mean().item(),
                    'energy/barrier': E_bar.mean().item(),
                    'energy/direction': E_dir.mean().item(),
                    'optimization/temperature': temperature.item(),
                    'optimization/step_size': s.item(),
                    'optimization/acceptance_rate': accept.float().mean().item()
                })
            
            # Verbose output and intermediate visualization
            if step % verbose_step == 0 or step == num_steps:
                print(f"Step {step:4d}: E_total={energy.sum().item():.6f}, "
                      f"E_fc={E_fc.mean().item():.4f}, "
                      f"E_dis={E_dis.mean().item():.4f}, "
                      f"E_bar={E_bar.mean().item():.4f}, "
                      f"temp={temperature.item():.3f}, "
                      f"accept={accept.float().mean().item():.2f}")
                
                # Save intermediate visualization if output directory is provided
                if output_dir is not None and object_code is not None and grasp_idx is not None:
                    self.save_intermediate_visualization(
                        step, xyz, contact_vals, closest_point, contact_normal, closest_idx,
                        output_dir, object_code, grasp_idx
                    )
        
        print(f"\n=== Annealing Optimization Complete ===")
        print(f"Final Total Energy: {energy.sum().item():.6f}")
        print(f"Final Temperature: {temperature.item():.6f}")
        
        return self.hand_model.hand_pose.clone()
        
    def save_intermediate_visualization(self, step, xyz, contact_vals, closest_point, contact_normal, closest_idx, output_dir, object_code, grasp_idx):
        """Save intermediate hand pose visualization at verbose steps using plotly with HTML output"""
        try:
            # Get hand and object plotly data
            hand_plotly_opt = self.hand_model.get_plotly_data(i=0, opacity=0.7, color='lightblue', with_contact_points=True)
            object_plotly_opt = self.object_model.get_plotly_data(i=0, color='lightgreen', opacity=0.6)
            
            # Create figure with hand and object data
            fig_opt = go.Figure(hand_plotly_opt + object_plotly_opt)
            
            # Add contact points from affordance map
            contact_threshold = self.config.contact_threshold
            high_contact_mask = contact_vals > contact_threshold
            if np.any(high_contact_mask):
                contact_points = xyz[high_contact_mask]
                fig_opt.add_trace(go.Scatter3d(
                    x=contact_points[:, 0],
                    y=contact_points[:, 1], 
                    z=contact_points[:, 2],
                    mode='markers',
                    marker=dict(color='red', size=3),
                    name='Affordance Contact Points'
                ))
            
            # Add closest point and normal
            representative_normal = contact_normal
            normal_end = closest_point + representative_normal * 0.05
            
            fig_opt.add_trace(go.Scatter3d(
                x=[closest_point[0]],
                y=[closest_point[1]],
                z=[closest_point[2]],
                mode='markers',
                marker=dict(color='blue', size=8),
                name='Closest Point'
            ))
            
            fig_opt.add_trace(go.Scatter3d(
                x=[closest_point[0], normal_end[0]],
                y=[closest_point[1], normal_end[1]],
                z=[closest_point[2], normal_end[2]],
                mode='lines',
                line=dict(color='orange', width=6),
                name='Contact Normal'
            ))
            
            # Add optimized contact points
            opt_contact_points = self.hand_model.contact_points[0].detach().cpu().numpy()
            fig_opt.add_trace(go.Scatter3d(
                x=opt_contact_points[:, 0],
                y=opt_contact_points[:, 1],
                z=opt_contact_points[:, 2],
                mode='markers',
                marker=dict(color='yellow', size=6, symbol='diamond'),
                name='Optimized Contact Points'
            ))
            
            fig_opt.update_layout(
                title=f"Step {step}: Optimized Affordance-Aligned Grasping - {object_code} Grasp {grasp_idx}",
                scene=dict(aspectmode='data'),
                width=800,
                height=600
            )
            
            # Save as HTML file
            png_path = output_dir / f"{object_code}_grasp_{grasp_idx}_step_{step:04d}.png"
            fig_opt.write_image(png_path)
            
            print(f"Intermediate visualization saved: {png_path}")
            
            # Log to wandb if enabled
            if self.use_wandb:
                wandb.log({f"visualization/step_{step}": wandb.Html(str(png_path))})
                
        except Exception as e:
            print(f"Visualization failed at step {step}: {e}")
            import traceback
            traceback.print_exc()
            print("Continuing optimization...")
            
    def run_pipeline(self, object_code, data_path, mesh_data_path, grasp_idx=0, 
                    num_steps=6000, output_dir=None):
        """Run complete affordance-aligned optimization pipeline"""
        print(f"=== Affordance-Aligned Grasping Pipeline ===")
        print(f"Object: {object_code}, Grasp Index: {grasp_idx}")
        
        # Initialize models
        self.initialize_models(object_code, mesh_data_path)
        
        # Load affordance data
        xyz, contact_vals = load_affordance_data(data_path, grasp_idx)
        
        # Extract contact normal
        closest_point, contact_normal, closest_idx = extract_contact_normal(xyz, contact_vals, self.config.contact_threshold)
        
        # Initialize hand with contact normal
        initialize_hand_with_contact_normal(self.hand_model, self.object_model, closest_point, contact_normal, self.config)
        
        # Setup output directory for intermediate visualizations
        viz_output_dir = None
        if output_dir:
            viz_output_dir = Path(output_dir)
            viz_output_dir.mkdir(parents=True, exist_ok=True)
            print(f"Intermediate visualizations will be saved to: {viz_output_dir}")
        
        # Run optimization with intermediate visualization saving
        optimized_pose = self.optimize_grasp_annealing(
            xyz, contact_vals, 
            num_steps=num_steps,
            output_dir=viz_output_dir,
            object_code=object_code,
            grasp_idx=grasp_idx,
            closest_point=closest_point,
            contact_normal=contact_normal,
            closest_idx=closest_idx
        )
        
        return {
            'optimized_pose': optimized_pose,
            'xyz': xyz,
            'contact_vals': contact_vals,
            'closest_point': closest_point,
            'contact_normal': contact_normal
        }


def main():
    parser = argparse.ArgumentParser(description='Affordance-Aligned Grasping Optimization')
    parser.add_argument('--object_code', type=str, required=True, 
                       help='Object code (e.g., s20349)')
    parser.add_argument('--grasp_idx', type=int, default=0,
                       help='Index of grasp in contact map (default: 0)')
    parser.add_argument('--data_path', type=str, 
                       default='/home/dmsgk724/CVPR_2026/dataset/processed_dexgys3/{object_code}/xyzc.npy',
                       help='Path to affordance data file')
    parser.add_argument('--mesh_data_path', type=str, 
                       default='/home/dmsgk724/CVPR_2026/grasping/Grasp-as-You-Say/asset_process/data/meshdata',
                       help='Root path for mesh data')
    parser.add_argument('--num_steps', type=int, default=1000,
                       help='Number of optimization steps')
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    
    # Energy weight arguments
    parser.add_argument('--w_dis', type=float, default=100.0,
                       help='Distance energy weight')
    parser.add_argument('--w_bar', type=float, default=1.0,
                       help='Barrier energy weight')
    parser.add_argument('--w_dir', type=float, default=10.0,
                       help='Direction alignment energy weight')
    parser.add_argument('--w_fc', type=float, default=1.0,
                       help='Force closure energy weight')
    parser.add_argument('--wandb_project', type=str, default='affordance-aligned-grasping',
                       help='Wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                       help='Wandb run name (default: object_code_grasp_idx)')
    parser.add_argument('--no_wandb', action='store_true',
                       help='Disable wandb logging')
    
    args = parser.parse_args()
    
    # Format data path with object code
    data_path = args.data_path.format(object_code=args.object_code)
    
    # Initialize wandb if enabled
    use_wandb = not args.no_wandb
    if use_wandb:
        run_name = args.wandb_run_name or f"{args.object_code}_grasp_{args.grasp_idx}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                'object_code': args.object_code,
                'grasp_idx': args.grasp_idx,
                'num_steps': args.num_steps,
                'device': args.device,
                'w_dis': args.w_dis,
                'w_bar': args.w_bar,
                'w_dir': args.w_dir,
                'w_fc': args.w_fc,
                'data_path': data_path,
                'mesh_data_path': args.mesh_data_path
            }
        )
        print(f"Wandb initialized: project={args.wandb_project}, run={run_name}")
    
    # Create configuration
    config = AffordanceGraspConfig()
    config.w_dis = args.w_dis
    config.w_bar = args.w_bar
    config.w_dir = args.w_dir
    config.w_fc = args.w_fc
    
    # Initialize optimizer
    optimizer = AffordanceAlignedOptimizer(config, device=args.device, use_wandb=use_wandb)
    
    # Run pipeline
    results = optimizer.run_pipeline(
        object_code=args.object_code,
        data_path=data_path,
        mesh_data_path=args.mesh_data_path,
        grasp_idx=args.grasp_idx,
        num_steps=args.num_steps,
        output_dir=args.output_dir
    )
    
    print(f"\n=== Pipeline Complete ===")
    if args.output_dir:
        print(f"Intermediate visualizations saved to: {args.output_dir}")
    else:
        print("No output directory specified - visualizations not saved")
    
    # Finish wandb run
    if use_wandb:
        wandb.finish()
        print("Wandb run completed")


if __name__ == "__main__":
    main()