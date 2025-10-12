# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DexGraspNet is a large-scale robotic dexterous grasp dataset for general objects based on simulation. This repository contains tools for asset processing, grasp synthesis, and grasp validation using the ShadowHand robotic hand and Isaac Gym simulator.

## Key Commands

### Environment Setup
```bash
# Create conda environment with Python 3.7 (required for Isaac Gym)
conda create -n dexgraspnet python=3.7
conda activate dexgraspnet

# Install PyTorch with CUDA support (~1.10 with cudatoolkit ~11.3)
conda install pytorch==1.10.* cudatoolkit=11.3 -c pytorch

# Install core dependencies
conda install pytorch3d transforms3d trimesh plotly rtree -c conda-forge
pip install urdf_parser_py scipy networkx pyyaml lxml

# Install custom pytorch_kinematics
cd thirdparty/pytorch_kinematics
pip install -e .

# Build and install third-party tools
cd ../ManifoldPlus
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8

cd ../../CoACD
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make main -j

# Install TorchSDF (custom Kaolin version)
cd ../../TorchSDF
git checkout 0.1.0
bash install.sh
```

### Asset Processing Pipeline
```bash
cd asset_process

# Extract models from datasets
python extract.py --src data/ShapeNetCore.v2 --dst data/raw_models --set core
python extract.py --src data/ShapeNetSem/models --dst data/raw_models --set sem --meta data/ShapeNetSem/metadata.csv
python extract.py --src data/mujoco_scanned_objects/models --dst data/raw_models --set mujoco
python extract.py --src data/Grasp_Dataset/good_shapes --dst data/raw_models --set ddg

# Convert to manifolds
python manifold.py --src data/raw_models --dst data/manifolds --manifold_path ../thirdparty/ManifoldPlus/build/manifold
bash run.sh  # or: python poolrun.py -p 32

# Normalize models
python normalize.py --src data/manifolds --dst data/normalized_models

# Decompose into convex parts for physics simulation
python decompose_list.py --src data/normalized_models --dst data/meshdata --coacd_path ../thirdparty/CoACD/build/main
bash run.sh  # or: python poolrun.py -p 32
```

### Grasp Generation
```bash
cd grasp_generation

# Set visible GPUs for multi-GPU execution
export CUDA_VISIBLE_DEVICES=0,1,2

# Generate grasps for specific objects
python main.py --object_code_list "['sem-Car-2f28e2bd754977da8cfac9da0ff28f62']" --name exp_test

# Large-scale grasp generation
python scripts/generate_grasps.py --all
python scripts/generate_grasps.py --split 0 --n_splits 8  # for parallel processing

# Validate grasps using Isaac Gym
python scripts/validate_grasps.py
python scripts/validate_grasps_batch.py  # batch processing
python scripts/poolrun.py -p 8  # parallel validation

# Visualize results
python tests/visualize_result.py --object_code <object_code>
python tests/visualize_hand_model.py  # visualize hand model
```

### Quick Testing
```bash
# Minimal setup for testing (CPU-only)
conda install pytorch cpuonly -c pytorch
conda install ipykernel transforms3d trimesh -c conda-forge
pip install pyyaml lxml

cd grasp_generation
jupyter notebook quick_example.ipynb
```

### Testing and Development
```bash
# Run tests for specific components
cd grasp_generation
python -m pytest tests/

# Check hand model visualization
python tests/visualize_hand_model.py

# Test object loading and SDF computation
python -c "from utils.object_model import ObjectModel; print('Object model working')"

# Test energy computation
python -c "from utils.energy import cal_energy; print('Energy computation working')"
```

## Project Architecture

### Core Components

1. **Asset Processing (`asset_process/`)**
   - Pipeline: Extract → Manifold → Normalize → Decompose
   - Processes objects from ShapeNet, Mujoco, DDG datasets
   - Uses ManifoldPlus for mesh correction and CoACD for convex decomposition
   - Outputs URDF files with convex pieces for physics simulation

2. **Grasp Generation (`grasp_generation/`)**
   - Energy-based optimization using Differentiable Force Closure Estimator
   - Simulated annealing optimization with multiple energy terms
   - Supports multi-GPU batch processing
   - Outputs grasp poses with hand joint configurations

3. **Physics Validation**
   - Isaac Gym simulator integration for grasp stability testing
   - Validates synthesized grasps under physics simulation
   - Filters out unstable or penetrating grasps

### Key Algorithm Components

- **Energy Function (`utils/energy.py`)**:
  - `E_fc`: Force closure quality
  - `E_dis`: Distance to object surface
  - `E_pen`: Object-hand penetration penalty
  - `E_spen`: Self-penetration penalty
  - `E_joints`: Joint limit violations

- **Hand Model (`utils/hand_model.py`)**:
  - ShadowHand kinematics using pytorch_kinematics
  - Contact point and penetration keypoint management
  - Collision detection and SDF computation

- **Object Model (`utils/object_model.py`)**:
  - Batch processing of multiple objects
  - Surface point sampling with FPS
  - SDF-based distance computation

- **Optimization (`utils/optimizer.py`)**:
  - Simulated annealing with adaptive step sizes
  - Multi-start initialization from convex hull
  - Temperature scheduling and convergence criteria

### Data Pipeline

```
Raw Models → ManifoldPlus → Normalized Models → CoACD → URDF Files
     ↓              ↓              ↓             ↓          ↓
  extract.py → manifold.py → normalize.py → decompose.py → meshdata/
```

### Data Structure

```
data/
├── raw_models/         # Extracted models from datasets
├── manifolds/          # Manifold-corrected meshes
├── normalized_models/  # Size and pose normalized models
├── meshdata/           # Final processed models with URDF files
│   └── {source}(-{category})-{code}/
│       ├── coacd/
│       │   ├── coacd.urdf
│       │   ├── decomposed.obj
│       │   └── coacd_convex_piece_*.obj
├── graspdata/          # Generated grasps awaiting validation
├── dataset/            # Validated grasp dataset (final output)
└── experiments/        # Small-scale experimental results
```

### Third-party Dependencies

- **TorchSDF**: Custom SDF computation (modified Kaolin) for distance queries
- **pytorch_kinematics**: Modified for faster ShadowHand forward kinematics
- **ManifoldPlus**: Robust mesh manifold correction
- **CoACD**: Collision-aware convex decomposition for physics simulation
- **Isaac Gym**: High-performance physics simulation for grasp validation

## Development Patterns

### Energy Function Weights
Default energy weights in optimization (magical numbers - avoid changing):
- `w_dis=100.0`: Surface distance weight
- `w_pen=100.0`: Penetration penalty weight  
- `w_spen=10.0`: Self-penetration penalty weight
- `w_joints=1.0`: Joint limit penalty weight

### GPU Memory Management
Batch sizes depend on GPU memory:
- RTX 3090/4090 (24GB): `batch_size_each=128`, `max_total_batch_size=1024`
- RTX 2080 Ti (11GB): `batch_size_each=64`, `max_total_batch_size=512`
- Adjust based on available VRAM and object complexity

### Multi-GPU Processing
Use environment variables to control GPU usage:
```bash
export CUDA_VISIBLE_DEVICES=0,1,2  # Specify available GPUs
export FORCE_CUDA=1  # Force CUDA compilation in containers
```

### Object Code Format
Objects follow naming convention: `{source}(-{category})-{code}`
- `source`: core (ShapeNetCore), sem (ShapeNetSem), mujoco, ddg
- `category`: object category (optional, e.g., Car, Mug)
- `code`: unique identifier from original dataset

### File I/O Patterns
- Grasp data stored as `.npy` files with list of dictionaries
- Each grasp contains: `scale`, `qpos`, and energy terms during generation
- Energy terms removed after validation to reduce file size
- Experimental results include full optimization history

## Common Issues and Solutions

### Isaac Gym Installation
Isaac Gym requires Python 3.7 and specific PyTorch versions. If installation fails:
1. Ensure Python 3.7 environment
2. Install PyTorch 1.10.x with matching CUDA toolkit
3. Install Isaac Gym from downloaded wheel file

### CUDA Compilation Issues
For TorchSDF and other CUDA extensions:
1. Check CUDA toolkit version matches PyTorch CUDA version
2. Set `TORCH_CUDA_ARCH_LIST` for specific GPU architectures
3. Use `FORCE_CUDA=1` for cross-compilation in containers

### Memory Issues
For large-scale processing:
1. Reduce `batch_size_each` for GPU memory constraints
2. Use `max_total_batch_size` to limit total memory usage
3. Process objects in smaller chunks for very large datasets