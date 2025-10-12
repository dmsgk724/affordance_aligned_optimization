
import os
import sys
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
import multiprocessing
from tqdm import tqdm
import random

from torch.multiprocessing import set_start_method

try:
    set_start_method('spawn')
except RuntimeError:
    pass

# Import DexGraspNet modules
from utils.hand_model import HandModel
from utils.object_model import ObjectModel
from utils.optimizer import Annealing
from utils.initialization import extract_contact_normal, initialize_hand_with_contact_normal, load_affordance_data
from utils.affordance_energy import compute_total_energy_for_annealing


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
np.seterr(all='raise')

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
        self.poses_per_contact = 1000 # Number of initial poses per contact map
        
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

def generate(args_list):
    args, object_code, worker_id, gpu_list = args_list
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Set GPU
    worker = multiprocessing.current_process()._identity[0]
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list[worker - 1]
    device = torch.device('cuda')
    
    try:
        # Create configuration
        config = AffordanceGraspConfig()
        config.w_dis = args.w_dis
        config.w_bar = args.w_bar
        config.w_dir = args.w_dir
        config.w_fc = args.w_fc
        config.w_pen = args.w_pen
        config.w_spen = args.w_spen
        config.w_joints = args.w_joints
        
        # Initialize models
        hand_model = HandModel(
            mjcf_path='mjcf/shadow_hand_wrist_free.xml',
            mesh_path='mjcf/meshes',
            contact_points_path='mjcf/contact_points.json',
            penetration_points_path='mjcf/penetration_points.json',
            device=device
        )
        
        # Calculate required batch size (5 contact normals * 5000 poses each)
        batch_size_each = 5000
        
        object_model = ObjectModel(
            data_root_path=args.mesh_data_path,
            batch_size_each=batch_size_each,
            num_samples=2000,
            device=device
        )
        object_model.initialize([object_code])
        
        # Load affordance data
        data_path = args.data_path.format(object_code=object_code)
        if not os.path.exists(data_path):
            print(f"Warning: Affordance data not found for {object_code}: {data_path}")
            return
            
        xyz, contact_vals, target_masks, non_target_masks = load_affordance_data(data_path)
        
        # Extract contact normal
        closest_point, contact_normal, closest_idx = extract_contact_normal(
            xyz, contact_vals, config.contact_threshold
        )
        
        # Initialize hand with contact normal
        initialize_hand_with_contact_normal(
            hand_model, object_model, closest_point, contact_normal, config
        )
        
        # # Store initial pose
        hand_pose_st = hand_model.hand_pose.detach()
        
        # # Configure annealing optimizer
        optim_config = {
            'switch_possibility': config.switch_possibility,
            'starting_temperature': config.starting_temperature,
            'temperature_decay': config.temperature_decay,
            'annealing_period': config.annealing_period,
            'step_size': config.step_size,
            'stepsize_period': config.stepsize_period,
            'mu': config.mu,
            'device': device
        }
        optimizer = Annealing(hand_model, **optim_config)
        
        # # Initial energy computation
        # compute_total_energy_for_annealing(hand_model, object_model, xyz, contact_vals, target_masks, non_target_masks, config)
        energy, E_fc, E_dis, E_pen, E_spen, E_joints, E_bar, E_dir = compute_total_energy_for_annealing(
            hand_model, object_model, xyz, contact_vals, target_masks, non_target_masks, config
        )
        
        energy.sum().backward(retain_graph=True)
        
        pbar = tqdm(range(1, args.n_iter + 1), desc=f"Optimizing {object_code}", 
                   unit="step", dynamic_ncols=True)
        
        
        for step in pbar:
            
            s = optimizer.try_step()
            optimizer.zero_grad()
            
            new_energy, new_E_fc, new_E_dis, new_E_pen, new_E_spen, new_E_joints, new_E_bar, new_E_dir = compute_total_energy_for_annealing(
                hand_model, object_model, xyz, contact_vals, target_masks, non_target_masks, config
            )
            
            new_energy.sum().backward(retain_graph=True)
            
            with torch.no_grad():
                accept, t = optimizer.accept_step(energy, new_energy)
                
                energy[accept] = new_energy[accept]
                E_fc[accept] = new_E_fc[accept]
                E_dis[accept] = new_E_dis[accept]
                E_pen[accept] = new_E_pen[accept]
                E_spen[accept] = new_E_spen[accept]
                E_joints[accept] = new_E_joints[accept]
                E_bar[accept] = new_E_bar[accept]
                E_dir[accept] = new_E_dir[accept]

                # Update progress bar with timing and energy statistics
                if step % 100 == 0 or step == 1:  # Update every 100 steps
                    avg_energy = energy.mean().item()
                    min_energy = energy.min().item()
                    accept_rate = accept.float().mean().item()
    
                    pbar.set_postfix({
                        'Avg_E': f'{avg_energy:.3f}',
                        'Min_E': f'{min_energy:.3f}',
                        'Accept': f'{accept_rate:.2%}',
                        'Temp': f'{t:.2f}',
                    })
        
        pbar.close()
        
        
        # Save results
        result_data = {
            'object_code': object_code,
            'grasp_idx': args.grasp_idx,
            'optimized_pose': hand_model.hand_pose[0].detach().cpu().numpy(),
            'initial_pose': hand_pose_st[0].detach().cpu().numpy(),
            'scale': object_model.object_scale_tensor[0][0].item(),
            'energy': energy[0].item(),
            'E_fc': E_fc[0].item(),
            'E_dis': E_dis[0].item(),
            'E_pen': E_pen[0].item(),
            'E_spen': E_spen[0].item(),
            'E_joints': E_joints[0].item(),
            'E_bar': E_bar[0].item(),
            'E_dir': E_dir[0].item(),
            'closest_point': closest_point,
            'contact_normal': contact_normal,
            'xyz': xyz,
            'contact_vals': contact_vals
        }
        
        # # Save to file
        # output_file = os.path.join(args.result_path, f"{object_code}_grasp_{args.grasp_idx}.npy")
        # np.save(output_file, result_data, allow_pickle=True)
        
        return True
        
    except Exception as e:
        print(f"Error processing {object_code}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description='Affordance-Aligned Grasping Multi-Object Optimization')
    
    # Object selection
    parser.add_argument('--object_code_list', nargs='*', type=str,
                       help='List of object codes to process')
    parser.add_argument('--all', action='store_true',
                       help='Process all objects in data directory')
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite existing results')
    parser.add_argument('--todo', action='store_true',
                       help='Process objects from todo.txt')
    
    # Paths
    parser.add_argument('--result_path', type=str, default='./results/affordance_grasps',
                       help='Output directory for results')
    parser.add_argument('--data_path', type=str, 
                       default='/home/dmsgk724/CVPR_2026/dataset/processed_dexgys3/{object_code}/xyzc.npy',
                       help='Path template for affordance data file')
    parser.add_argument('--mesh_data_path', type=str, 
                       default='/home/dmsgk724/CVPR_2026/grasping/Grasp-as-You-Say/asset_process/data/meshdata',
                       help='Root path for mesh data')
    
    # Optimization parameters
    parser.add_argument('--n_iter', type=int, default=6000,
                       help='Number of optimization steps')
    parser.add_argument('--seed', type=int, default=1,
                       help='Random seed')
    
    # Energy weights
    parser.add_argument('--w_dis', type=float, default=100.0,
                       help='Distance energy weight')
    parser.add_argument('--w_pen', type=float, default=100.0,
                       help='Penetration weight')
    parser.add_argument('--w_spen', type=float, default=10.0,
                       help='Self-penetration weight')
    parser.add_argument('--w_joints', type=float, default=1.0,
                       help='Joint limits weight')
    parser.add_argument('--w_bar', type=float, default=1.0,
                       help='Barrier energy weight')
    parser.add_argument('--w_dir', type=float, default=10.0,
                       help='Direction alignment energy weight')
    parser.add_argument('--w_fc', type=float, default=1.0,
                       help='Force closure energy weight')
    
    # Annealing parameters
    parser.add_argument('--switch_possibility', type=float, default=0.5)
    parser.add_argument('--mu', type=float, default=0.98)
    parser.add_argument('--step_size', type=float, default=0.005)
    parser.add_argument('--stepsize_period', type=int, default=50)
    parser.add_argument('--starting_temperature', type=float, default=18.0)
    parser.add_argument('--annealing_period', type=int, default=30)
    parser.add_argument('--temperature_decay', type=float, default=0.95)
    
    args = parser.parse_args()
    
    # Get GPU list
    gpu_list = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")
    print(f'GPU list: {gpu_list}')
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    
    # Create result directory
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)
    
    # Check mesh data path
    if not os.path.exists(args.mesh_data_path):
        raise ValueError(f'mesh_data_path {args.mesh_data_path} doesn\'t exist')
    
    # Validate object selection
    if (args.object_code_list is not None) + args.all + args.todo != 1:
        raise ValueError('exactly one among \'object_code_list\', \'all\', \'todo\' should be specified')
    
    # Build object list
    if args.todo:
        with open("todo.txt", "r") as f:
            lines = f.readlines()
            object_code_list_all = [line.strip() for line in lines if line.strip()]
    else:
        object_code_list_all = os.listdir(args.mesh_data_path)
    
    if args.object_code_list is not None:
        object_code_list = args.object_code_list
        if not set(object_code_list).issubset(set(object_code_list_all)):
            raise ValueError('object_code_list isn\'t a subset of available objects')
    else:
        object_code_list = object_code_list_all
    
    # Filter out already processed objects if not overwriting
    if not args.overwrite:
        for object_code in object_code_list.copy():
            result_file = os.path.join(args.result_path, f"{object_code}_grasp.npy")
            if os.path.exists(result_file):
                object_code_list.remove(object_code)
    
    print(f'Number of objects to process: {len(object_code_list)}')
    
    if len(object_code_list) == 0:
        print("No objects to process!")
        return
    
    # Shuffle object list
    random.seed(args.seed)
    random.shuffle(object_code_list)
    
    # Prepare multiprocessing arguments
    process_args = []
    for i, object_code in enumerate(object_code_list):
        process_args.append((args, object_code, i + 1, gpu_list))
    
    # Run multiprocessing
    print(f"\n=== Starting Multi-Object Affordance-Aligned Grasping ===")
    print(f"Objects: {len(object_code_list)}")
    print(f"GPUs: {len(gpu_list)}")
    print(f"Iterations per object: {args.n_iter}")
    
    with multiprocessing.Pool(len(gpu_list)) as p:
        it = tqdm(p.imap(generate, process_args), 
                 total=len(process_args), 
                 desc='Processing objects',
                 maxinterval=1000)
        results = list(it)
    
    # Summary
    successful = sum(1 for r in results if r)
    print(f"\n=== Processing Complete ===")
    print(f"Successfully processed: {successful}/{len(object_code_list)} objects")
    print(f"Results saved to: {args.result_path}")


if __name__ == "__main__":
    main()