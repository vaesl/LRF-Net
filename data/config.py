import os.path

LRF_ROOT = os.path.dirname(os.getcwd())
# access to the COCO dataset
COCOdir = os.path.join('/home/zxy/', "data/coco/")

# LRF CONFIGS
COCO_300 = {
    'feature_maps': [38, 19, 10, 5, 3, 1],

    'min_dim': 300,

    'steps': [8, 16, 32, 64, 100, 300],

    'min_sizes': [21, 45, 99, 153, 207, 261],

    'max_sizes': [45, 99, 153, 207, 261, 315],

    'aspect_ratios': [[2, 3], [2, 3], [2, 3], [2, 3], [2], [2]],

    'variance': [0.1, 0.2],

    'clip': True,
}

COCO_512 = {
    'feature_maps': [64, 32, 16, 8, 4, 2, 1],

    'min_dim': 512,

    'steps': [8, 16, 32, 64, 128, 256, 512],

    'min_sizes': [20.48, 51.2, 133.12, 215.04, 296.96, 378.88, 460.8],

    'max_sizes': [51.2, 133.12, 215.04, 296.96, 378.88, 460.8, 542.72],

    'aspect_ratios': [[2, 3], [2, 3], [2, 3], [2, 3], [2,3], [2], [2]],

    'variance': [0.1, 0.2],

    'clip': True,
}



