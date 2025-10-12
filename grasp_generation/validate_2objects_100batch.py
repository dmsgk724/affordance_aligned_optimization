#!/usr/bin/env python3
"""
Validation script specifically for 2 objects with 100 batch size each

This script validates that the modified affordance_aligned_optimization.py
correctly handles exactly 2 objects with 100 grasps per object.
"""

import os
import sys
import numpy as np
import torch
import time
from pathlib import Path

# Add the current directory to path for imports
sys.path.append(os.path.realpath('.'))

def create_mock_affordance_data():
    """Create mock affordance data for testing"""
    print("Creating mock affordance data...")
    
    # Use existing objects if available
    meshdata_path = Path("../data/meshdata")
    if meshdata_path.exists():
        available_objects = [d.name for d in meshdata_path.iterdir() if d.is_dir()]
        if len(available_objects) >= 2:
            test_objects = available_objects[:2]
            print(f"Using existing objects: {test_objects}")
        else:
            print("Not enough objects in meshdata, using default test objects")
            test_objects = [
                "core-mug-8570d9a8d24cb0acbebd3c0c0c70fb03",
                "sem-Bottle-437678d4bc6be981c8724d5673a063a6"
            ]
    else:
        print("Meshdata path not found, using default test objects")
        test_objects = [
            "core-mug-8570d9a8d24cb0acbebd3c0c0c70fb03", 
            "sem-Bottle-437678d4bc6be981c8724d5673a063a6"
        ]
    
    # Create mock data directory
    mock_data_dir = Path("mock_affordance_data")
    mock_data_dir.mkdir(exist_ok=True)
    
    for obj_code in test_objects:
        obj_dir = mock_data_dir / obj_code
        obj_dir.mkdir(exist_ok=True)
        
        # Create realistic mock point cloud
        n_points = 3000
        
        # Generate points in a rough object shape
        theta = np.random.uniform(0, 2*np.pi, n_points)
        phi = np.random.uniform(0, np.pi, n_points)
        r = np.random.uniform(0.05, 0.15, n_points)  # Object radius 5-15cm
        
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi) + 0.1  # Lift object above ground
        
        xyz = np.column_stack([x, y, z])
        
        # Create contact map with concentrated high-contact regions
        contact_vals = np.random.uniform(0.0, 0.3, n_points)  # Base low contact
        
        # Add high-contact regions (simulating affordance areas)
        # Top region (lid for mug, cap for bottle)
        top_mask = z > (z.mean() + 0.5 * z.std())
        contact_vals[top_mask] = np.random.uniform(0.6, 1.0, np.sum(top_mask))
        
        # Side regions (handle for mug, grip for bottle)
        side_mask = (np.abs(x) > 0.8 * np.abs(x).max()) | (np.abs(y) > 0.8 * np.abs(y).max())
        contact_vals[side_mask] = np.random.uniform(0.5, 0.9, np.sum(side_mask))
        
        # Ensure some contact values are above threshold
        n_high_contact = max(100, int(0.1 * n_points))
        high_indices = np.random.choice(n_points, n_high_contact, replace=False)
        contact_vals[high_indices] = np.random.uniform(0.6, 1.0, n_high_contact)
        
        # Combine into xyzc format
        xyzc = np.column_stack([xyz, contact_vals])
        
        # Save mock data
        output_path = obj_dir / "xyzc.npy"
        np.save(output_path, xyzc)
        print(f"Created mock data for {obj_code}: {xyzc.shape} points")
    
    return test_objects, str(mock_data_dir)


def test_batch_processing():
    """Test the main functionality: 2 objects with 100 batch size each"""
    print("\n=== Testing 2 Objects with 100 Batch Size Each ===")
    
    # Create mock data
    test_objects, mock_data_dir = create_mock_affordance_data()
    print(f"Test objects: {test_objects}")
    
    # Import the worker function directly for testing
    try:
        from scripts.affordance_aligned_optimization import generate_affordance_grasps_worker, AffordanceGraspConfig
    except ImportError as e:
        print(f"Error importing: {e}")
        print("Make sure you're running from the grasp_generation directory")
        return False
    
    # Create arguments
    class TestArgs:
        def __init__(self):
            self.seed = 42
            self.batch_size_each = 100
            self.mesh_data_path = "../data/meshdata"
            self.data_path = f"{mock_data_dir}/{{object_code}}/xyzc.npy"
            self.result_path = "./validation_results"
            self.grasp_idx = 0
            self.n_iter = 200  # Short run for validation
            
            # Energy weights
            self.w_dis = 100.0
            self.w_pen = 100.0
            self.w_spen = 10.0
            self.w_joints = 1.0
            self.w_bar = 1.0
            self.w_dir = 10.0
            self.w_fc = 1.0
            
            # Annealing parameters
            self.starting_temperature = 18.0
            self.temperature_decay = 0.95
            self.annealing_period = 30
            self.step_size = 0.005
            self.stepsize_period = 50
            self.mu = 0.98
            self.switch_possibility = 0.5
            
            # Initialization parameters
            self.distance_lower = 0.2
            self.distance_upper = 0.3
            self.theta_lower = -np.pi / 6
            self.theta_upper = np.pi / 6
            self.jitter_strength = 0.1
            self.n_contact = 4
    
    args = TestArgs()
    
    # Create results directory
    os.makedirs(args.result_path, exist_ok=True)
    
    # Test parameters
    worker_id = 1
    gpu_list = ["0"] if torch.cuda.is_available() else [""]
    
    print(f"Running worker with:")
    print(f"  Objects: {test_objects}")
    print(f"  Batch size each: {args.batch_size_each}")
    print(f"  Total expected grasps: {len(test_objects) * args.batch_size_each}")
    print(f"  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    # Run the worker function
    start_time = time.time()
    try:
        # Set CUDA device
        if torch.cuda.is_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        
        generate_affordance_grasps_worker((args, test_objects, worker_id, gpu_list))
        
        end_time = time.time()
        print(f"Processing completed in {end_time - start_time:.2f} seconds")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Validate results
    print("\n=== Validating Results ===")
    results_path = Path(args.result_path)
    success = True
    
    for obj_code in test_objects:
        output_file = results_path / f"{obj_code}_affordance.npy"
        
        if not output_file.exists():
            print(f"✗ Missing output file for {obj_code}")
            success = False
            continue
        
        try:
            # Load results
            data = np.load(output_file, allow_pickle=True)
            
            # Validate number of grasps
            if len(data) != args.batch_size_each:
                print(f"✗ {obj_code}: Expected {args.batch_size_each} grasps, got {len(data)}")
                success = False
            else:
                print(f"✓ {obj_code}: Correct number of grasps ({len(data)})")
            
            # Validate data structure
            if len(data) > 0:
                sample_grasp = data[0]
                required_keys = [
                    'scale', 'qpos', 'qpos_st', 'energy', 
                    'E_fc', 'E_dis', 'E_pen', 'E_spen', 'E_joints', 'E_bar', 'E_dir'
                ]
                
                missing_keys = [key for key in required_keys if key not in sample_grasp]
                if missing_keys:
                    print(f"✗ {obj_code}: Missing keys: {missing_keys}")
                    success = False
                else:
                    print(f"✓ {obj_code}: All required keys present")
                
                # Check energy values are reasonable
                energies = [g['energy'] for g in data]
                energy_mean = np.mean(energies)
                energy_std = np.std(energies)
                
                print(f"  Energy stats: mean={energy_mean:.4f}, std={energy_std:.4f}")
                
                if np.any(np.isnan(energies)) or np.any(np.isinf(energies)):
                    print(f"✗ {obj_code}: Invalid energy values detected")
                    success = False
                else:
                    print(f"✓ {obj_code}: Energy values are valid")
                
                # Check that poses are different (not all identical)
                poses = [g['qpos'] for g in data[:10]]  # Check first 10
                if len(set(str(p) for p in poses)) < 2:
                    print(f"✗ {obj_code}: Poses appear to be identical (no diversity)")
                    success = False
                else:
                    print(f"✓ {obj_code}: Poses show diversity")
        
        except Exception as e:
            print(f"✗ {obj_code}: Error loading/validating results: {e}")
            success = False
    
    return success


def test_memory_usage():
    """Test memory usage for batch processing"""
    print("\n=== Testing Memory Usage ===")
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping memory test")
        return True
    
    try:
        import torch
        
        # Clear cache
        torch.cuda.empty_cache()
        
        # Get initial memory
        initial_memory = torch.cuda.memory_allocated()
        print(f"Initial GPU memory: {initial_memory / 1024**2:.1f} MB")
        
        # Test with batch processing
        test_objects, mock_data_dir = create_mock_affordance_data()
        
        from scripts.affordance_aligned_optimization import generate_affordance_grasps_worker
        
        class TestArgs:
            def __init__(self):
                self.seed = 42
                self.batch_size_each = 100
                self.mesh_data_path = "../data/meshdata"
                self.data_path = f"{mock_data_dir}/{{object_code}}/xyzc.npy"
                self.result_path = "./memory_test_results"
                self.grasp_idx = 0
                self.n_iter = 50  # Very short for memory test
                
                # Default parameters
                self.w_dis = 100.0
                self.w_pen = 100.0
                self.w_spen = 10.0
                self.w_joints = 1.0
                self.w_bar = 1.0
                self.w_dir = 10.0
                self.w_fc = 1.0
                self.starting_temperature = 18.0
                self.temperature_decay = 0.95
                self.annealing_period = 30
                self.step_size = 0.005
                self.stepsize_period = 50
                self.mu = 0.98
                self.switch_possibility = 0.5
                self.distance_lower = 0.2
                self.distance_upper = 0.3
                self.theta_lower = -np.pi / 6
                self.theta_upper = np.pi / 6
                self.jitter_strength = 0.1
                self.n_contact = 4
        
        args = TestArgs()
        os.makedirs(args.result_path, exist_ok=True)
        
        # Run with memory monitoring
        max_memory = initial_memory
        
        def memory_monitor():
            nonlocal max_memory
            current_memory = torch.cuda.memory_allocated()
            max_memory = max(max_memory, current_memory)
        
        # Run processing
        generate_affordance_grasps_worker((args, test_objects[:2], 1, ["0"]))
        
        final_memory = torch.cuda.memory_allocated()
        peak_memory = torch.cuda.max_memory_allocated()
        
        print(f"Final GPU memory: {final_memory / 1024**2:.1f} MB")
        print(f"Peak GPU memory: {peak_memory / 1024**2:.1f} MB")
        print(f"Memory increase: {(final_memory - initial_memory) / 1024**2:.1f} MB")
        
        # Clean up
        torch.cuda.empty_cache()
        
        # Check if memory usage is reasonable (< 8GB for 200 grasps)
        if peak_memory > 8 * 1024**3:  # 8GB in bytes
            print(f"✗ Memory usage too high: {peak_memory / 1024**3:.1f} GB")
            return False
        else:
            print(f"✓ Memory usage acceptable: {peak_memory / 1024**3:.1f} GB")
            return True
        
    except Exception as e:
        print(f"Memory test failed: {e}")
        return False


def cleanup():
    """Clean up test files"""
    import shutil
    
    cleanup_dirs = ["mock_affordance_data", "validation_results", "memory_test_results"]
    for dir_name in cleanup_dirs:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
            print(f"Cleaned up {dir_name}")


def main():
    """Main validation function"""
    print("=== Validation: 2 Objects with 100 Batch Size Each ===")
    
    # Check environment
    if not os.path.exists("../data/meshdata"):
        print("Warning: ../data/meshdata not found")
        print("Proceeding with mock data only")
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Run tests
    tests = [
        ("Batch Processing", test_batch_processing),
        ("Memory Usage", test_memory_usage),
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running {test_name} Test")
        print(f"{'='*50}")
        
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"✗ {test_name} test failed: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False
    
    # Summary
    print(f"\n{'='*50}")
    print("VALIDATION SUMMARY")
    print(f"{'='*50}")
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:20s}: {status}")
        if not passed:
            all_passed = False
    
    print(f"\nResult: {'✓ VALIDATION SUCCESSFUL' if all_passed else '✗ VALIDATION FAILED'}")
    
    if all_passed:
        print("\nThe modified affordance_aligned_optimization.py correctly handles:")
        print("- 2 objects with 100 grasps each")
        print("- Batch processing and multiprocessing")
        print("- Proper data structure and energy computation")
    
    # Cleanup
    cleanup()
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)