import argparse
import trimesh
import numpy as np
import tqdm
import json
import os
from utils.extract_utils import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, required=True)
    parser.add_argument("--dst", type=str, required=True)
    parser.add_argument("--scale_file", type=str, default="scale_info.json")
    args = parser.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    
    scale_info = {}

    for code in (os.listdir(args.src)):
        mesh = trimesh.load(os.path.join(args.src, code),
                            force="mesh", process=False)
        verts = np.array(mesh.vertices)
        xcenter = (np.max(verts[:, 0]) + np.min(verts[:, 0])) / 2
        ycenter = (np.max(verts[:, 1]) + np.min(verts[:, 1])) / 2
        zcenter = (np.max(verts[:, 2]) + np.min(verts[:, 2])) / 2
        center = np.array([xcenter, ycenter, zcenter])
        
        verts_ = verts - center
        dmax = np.max(np.sqrt(np.sum(np.square(verts_), axis=1))) * 1.03
        verts_ /= dmax
        
        # 스케일 정보 저장
        scale_info[code] = {
            "original_center": center.tolist(),
            "scale_factor": float(dmax),
            "inverse_scale": 1.0 / float(dmax)  # 원본 크기로 되돌리려면 이 값을 곱함
        }
        
        mesh_ = trimesh.Trimesh(
            vertices=verts_, faces=mesh.faces, process=False)
        if(mesh_.is_watertight and mesh_.volume > 0.05):
            mesh_.export(os.path.join(args.dst, code))
    
    # 스케일 정보를 JSON 파일로 저장
    with open(os.path.join(args.dst, args.scale_file), 'w') as f:
        json.dump(scale_info, f, indent=2)
    
    print(f"Saved scale information to {os.path.join(args.dst, args.scale_file)}")