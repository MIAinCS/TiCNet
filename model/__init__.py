from .main_net import MainNet
from .position_encoding import build_position_encoding
from .transformer import build_transformer
from .feature_net import build_feature_net

def build_model(config):
    return MainNet(config)
