from models.madnet2 import MADNet2
from models.deepmoe import CustomMadNet2
from models.losses import *

models_lut = {
    'madnet2': MADNet2,
    'madnet2-custom': CustomMadNet2
}
