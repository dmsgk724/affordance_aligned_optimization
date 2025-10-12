import argparse
from utils.extract_utils import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, required=True)
    parser.add_argument("--dst", type=str, required=True)
    parser.add_argument("--manifold_path", type=str, default="../thirdparty/ManifoldPlus/build/manifold")
    args = parser.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    
    with open("run.sh", "w") as f:
        for mesh in os.listdir(args.src):
            f.write(
                f"{args.manifold_path} --input {os.path.join(args.src, mesh)} --output {os.path.join(args.dst, mesh)}\n")

    # with open("run.sh", "w") as f:
    #     for root, dirs, files in os.walk(args.src):
    #         for file in files:
    #             if file.endswith(('.obj', '.off', '.ply')):
    #                 input_path = os.path.join(root, file)
    #                 relative_path = os.path.relpath(input_path, args.src)
    #                 output_path = os.path.join(args.dst, relative_path)
                    
    #                 os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    
    #                 f.write(
    #                     f"{args.manifold_path} --input {input_path} --output {output_path}\n")
