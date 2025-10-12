# python manifold.py --src /home/dmsgk724/CVPR_2026/dataset/OakInk/shape/OakInkObjectsV2/bottle_s001/align --dst data/manifolds --manifold_path ../ManifoldPlus/build/manifold

# bash run.sh

# # python normalize.py --src data/manifolds --dst data/normalized_models


# python normalize_with_scale.py --src data/manifolds --dst data/normalized_models --scale_file scale_info.json


# python decompose_list.py --src data/manifolds --dst data/meshdata --coacd_path ../CoACD/build/main

# bash run.sh

python decompose_list.py --src data/raw_models --dst data/meshdata --coacd_path ../CoACD/build/main