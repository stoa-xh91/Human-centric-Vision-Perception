from models.registry import BACKBONE
from models.registry import CLASSIFIER
from models.registry import LOSSES


model_dict = {
    'swin_t': 768,
    'swin_s': 768,
    'swin_b': 1024,
    'vit_tiny': 192,
    'kd_vit_tiny': 192,
    'vit_small': 384,
    'vit_base': 768,
    'cspnext_5m':512,
    'edgenext_small':304,
    'starnet_s3':256,
    'kd_starnet_s3':192,
    'tiny_vit': 256,  
    'tiny_vit_21m': 256,
    'kd_tiny_vit': 192,
    'kd_vit_small': 384,
}

print(BACKBONE.keys)

def build_backbone(key, multi_scale=False):


    model = BACKBONE[key]()
    output_d = model_dict[key]

    return model, output_d


def build_classifier(key):

    return CLASSIFIER[key]


def build_loss(key):

    return LOSSES[key]

