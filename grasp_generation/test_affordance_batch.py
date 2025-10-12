#!/usr/bin/env python3
"""
Test script for affordance-aligned optimization with batch processing

This script tests the modified affordance_aligned_optimization.py to ensure it correctly
handles multiple objects (2 objects) with batch processing (100 grasps each).

Usage:
    python test_affordance_batch.py
"""

import os
import sys
import subprocess
import time
import numpy as np
import torch
from pathlib import Path

# Add the current directory to path for imports
sys.path.append(os.path.realpath('.'))

def create_test_data():
    """Create dummy affordance data for testing"""
    test_data_dir = Path("test_data")
    test_data_dir.mkdir(exist_ok=True)
    
    # Create dummy affordance data for 2 objects
    objects = [
        "core-mug-8570d9a8d24cb0acbebd3c0c0c70fb03",
        "sem-Bottle-437678d4bc6be981c8724d5673a063a6"
    ]
    
    for obj_code in objects:
        obj_dir = test_data_dir / obj_code
        obj_dir.mkdir(exist_ok=True)
        
        # Create dummy point cloud with contact values
        n_points = 2000
        xyz = np.random.rand(n_points, 3) * 0.2  # Small object size
        xyz[:, 2] = np.abs(xyz[:, 2])  # Keep z positive
        
        # Create contact map with some high-contact regions
        contact_vals = np.random.rand(n_points, 1) * 0.3  # Base low contact
        # Add some high-contact regions
        high_contact_mask = np.random.rand(n_points) < 0.1  # 10% high contact
        contact_vals[high_contact_mask] = 0.6 + np.random.rand(np.sum(high_contact_mask), 1) * 0.4
        
        # Combine xyz and contact values
        xyzc = np.hstack([xyz, contact_vals])
        
        # Save to test data directory
        output_path = obj_dir / "xyzc.npy"
        np.save(output_path, xyzc, allow_pickle=True)
        print(f"Created test data: {output_path}")
    
    return objects, str(test_data_dir)


def test_single_threaded():
    """Test single-threaded batch processing"""
    print("\n=== Testing Single-Threaded Batch Processing ===")
    
    objects, test_data_dir = create_test_data()
    
    # Create test command
    cmd = [
        "python", "scripts/affordance_aligned_optimization.py",
        "--object_code_list"] + objects + [
        "--batch_size_each", "100",
        "--max_total_batch_size", "200",
        "--n_iter", "100",  # Short run for testing
        "--data_path", f"{test_data_dir}/{{object_code}}/xyzc.npy",
        "--mesh_data_path", "../data/meshdata",
        "--result_path", "./test_results",
        "--seed", "42",
        "--overwrite"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    # Run the command
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
    end_time = time.time()
    
    # Check results
    print(f"Exit code: {result.returncode}")
    print(f"Execution time: {end_time - start_time:.2f} seconds")
    
    if result.returncode != 0:
        print("STDERR:", result.stderr)
        print("STDOUT:", result.stdout)
        return False
    
    # Verify output files
    results_dir = Path("test_results")
    success = True
    for obj_code in objects:
        output_file = results_dir / f"{obj_code}_affordance.npy"
        if output_file.exists():
            try:
                data = np.load(output_file, allow_pickle=True)
                print(f"✓ {obj_code}: {len(data)} grasps generated")
                if len(data) != 100:
                    print(f"  WARNING: Expected 100 grasps, got {len(data)}")
                    success = False
                
                # Check data structure
                if len(data) > 0:
                    sample = data[0]
                    required_keys = ['scale', 'qpos', 'qpos_st', 'energy', 'E_fc', 'E_dis', 'E_pen', 'E_spen', 'E_joints', 'E_bar', 'E_dir']
                    for key in required_keys:
                        if key not in sample:
                            print(f"  ERROR: Missing key {key} in grasp data")
                            success = False
                        
            except Exception as e:
                print(f"✗ {obj_code}: Error loading results - {e}")
                success = False
        else:
            print(f"✗ {obj_code}: No output file found")
            success = False
    
    return success


def test_multiprocessing():
    """Test multiprocessing batch processing"""
    print("\n=== Testing Multiprocessing Batch Processing ===")
    
    objects, test_data_dir = create_test_data()
    
    # Create test command
    cmd = [
        "python", "scripts/affordance_aligned_optimization.py",
        "--object_code_list"] + objects + [
        "--batch_size_each", "100",
        "--max_total_batch_size", "200",
        "--multiprocessing",
        "--n_iter", "100",  # Short run for testing
        "--data_path", f"{test_data_dir}/{{object_code}}/xyzc.npy",
        "--mesh_data_path", "../data/meshdata",
        "--result_path", "./test_results_mp",
        "--seed", "42",
        "--overwrite"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    # Set CUDA_VISIBLE_DEVICES for testing
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    
    # Run the command
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=".", env=env)
    end_time = time.time()
    
    # Check results
    print(f"Exit code: {result.returncode}")
    print(f"Execution time: {end_time - start_time:.2f} seconds")
    
    if result.returncode != 0:
        print("STDERR:", result.stderr)
        print("STDOUT:", result.stdout)
        return False
    
    # Verify output files
    results_dir = Path("test_results_mp")
    success = True
    for obj_code in objects:
        output_file = results_dir / f"{obj_code}_affordance.npy"
        if output_file.exists():
            try:
                data = np.load(output_file, allow_pickle=True)
                print(f"✓ {obj_code}: {len(data)} grasps generated")
                if len(data) != 100:
                    print(f"  WARNING: Expected 100 grasps, got {len(data)}")
                    success = False
                    
                # Check data structure
                if len(data) > 0:
                    sample = data[0]
                    required_keys = ['scale', 'qpos', 'qpos_st', 'energy', 'E_fc', 'E_dis', 'E_pen', 'E_spen', 'E_joints', 'E_bar', 'E_dir']
                    for key in required_keys:
                        if key not in sample:
                            print(f"  ERROR: Missing key {key} in grasp data")
                            success = False
                        
            except Exception as e:
                print(f"✗ {obj_code}: Error loading results - {e}")
                success = False
        else:
            print(f"✗ {obj_code}: No output file found")
            success = False
    
    return success


def test_backward_compatibility():
    """Test backward compatibility with single object mode"""
    print("\n=== Testing Backward Compatibility (Single Object Mode) ===")
    
    objects, test_data_dir = create_test_data()
    obj_code = objects[0]  # Use first object
    
    # Create test command
    cmd = [
        "python", "scripts/affordance_aligned_optimization.py",
        "--object_code", obj_code,
        "--num_steps", "50",  # Very short run for testing
        "--data_path", f"{test_data_dir}/{{object_code}}/xyzc.npy",
        "--mesh_data_path", "../data/meshdata",
        "--output_dir", "./test_results_single",
        "--device", "cuda" if torch.cuda.is_available() else "cpu"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    # Run the command
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
    end_time = time.time()
    
    # Check results
    print(f"Exit code: {result.returncode}")
    print(f"Execution time: {end_time - start_time:.2f} seconds")
    
    if result.returncode != 0:
        print("STDERR:", result.stderr)
        print("STDOUT:", result.stdout)
        return False
    
    print("✓ Single object mode completed successfully")
    return True


def test_energy_computation():
    """Test that energy computation works correctly with batch processing"""
    print("\n=== Testing Energy Computation ===")
    
    try:
        from utils.hand_model import HandModel
        from utils.object_model import ObjectModel
        from utils.affordance_energy import compute_total_energy_for_annealing
        from scripts.affordance_aligned_optimization import AffordanceGraspConfig
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize models
        hand_model = HandModel(
            mjcf_path='mjcf/shadow_hand_wrist_free.xml',
            mesh_path='mjcf/meshes', 
            contact_points_path='mjcf/contact_points.json',
            penetration_points_path='mjcf/penetration_points.json',
            device=device
        )
        
        # Test with available objects
        available_objects = [f for f in os.listdir("../data/meshdata") if os.path.isdir(os.path.join("../data/meshdata", f))]
        if len(available_objects) < 2:
            print("Warning: Not enough objects in meshdata for energy test")
            return True
        
        test_objects = available_objects[:2]
        
        object_model = ObjectModel(
            data_root_path="../data/meshdata",
            batch_size_each=10,  # Small batch for testing
            num_samples=1000,
            device=device
        )
        object_model.initialize(test_objects)
        
        # Initialize hand poses
        from utils.initializations import initialize_convex_hull
        
        class DummyArgs:
            def __init__(self):
                self.distance_lower = 0.2
                self.distance_upper = 0.3
                self.theta_lower = -np.pi / 6
                self.theta_upper = np.pi / 6
                self.jitter_strength = 0.1
        
        args = DummyArgs()
        initialize_convex_hull(hand_model, object_model, args)
        
        # Create dummy affordance data
        xyz = np.random.rand(1000, 3) * 0.2
        contact_vals = np.random.rand(1000)
        
        # Test energy computation
        config = AffordanceGraspConfig()
        
        energy, E_fc, E_dis, E_pen, E_spen, E_joints, E_bar, E_dir = compute_total_energy_for_annealing(
            hand_model, object_model, xyz, contact_vals, config
        )
        
        # Verify energy shapes
        expected_batch_size = len(test_objects) * 10
        if energy.shape[0] != expected_batch_size:
            print(f"✗ Energy shape mismatch: expected {expected_batch_size}, got {energy.shape[0]}")
            return False
        
        # Verify all energies are computed
        energy_components = [E_fc, E_dis, E_pen, E_spen, E_joints, E_bar, E_dir]
        for i, comp in enumerate(energy_components):
            if comp.shape[0] != expected_batch_size:
                print(f"✗ Energy component {i} shape mismatch")
                return False
        
        print(f"✓ Energy computation test passed")
        print(f"  Batch size: {expected_batch_size}")
        print(f"  Total energy range: {energy.min().item():.4f} to {energy.max().item():.4f}")
        print(f"  Energy components: E_fc={E_fc.mean().item():.4f}, E_dis={E_dis.mean().item():.4f}, E_bar={E_bar.mean().item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Energy computation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def cleanup():
    """Clean up test files"""
    import shutil
    
    cleanup_dirs = ["test_data", "test_results", "test_results_mp", "test_results_single"]
    for dir_name in cleanup_dirs:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
            print(f"Cleaned up {dir_name}")


def main():
    """Run all tests"""
    print("=== Affordance-Aligned Optimization Batch Processing Tests ===")
    
    # Check if required directories exist
    if not os.path.exists("../data/meshdata"):
        print("Error: ../data/meshdata directory not found")
        print("Please make sure you're running from the grasp_generation directory")
        return False
    
    # Run tests
    tests = [
        ("Energy Computation", test_energy_computation),
        ("Backward Compatibility", test_backward_compatibility),
        ("Single-Threaded Batch", test_single_threaded),
        ("Multiprocessing Batch", test_multiprocessing),
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"Running {test_name} Test")
        print(f"{'='*60}")
        
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"✗ {test_name} test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:25s}: {status}")
        if not passed:
            all_passed = False
    
    print(f"\nOverall: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    
    # Cleanup
    cleanup()
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)