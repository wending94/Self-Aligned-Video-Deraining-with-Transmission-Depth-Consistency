from .vgg import *
from .TransformNet import *

from .SingleTransformNet import *
from .MultiTransformNet import *

from .FeatFlow import *
from .TDModel import *
from .FusionModel import *
from .RViDeNet import *

from .cycle_gan_model import CycleGANModel 

def create_model(opt):

    instance = CycleGANModel(opt)
    return instance 
