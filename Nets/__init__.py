import Nets.DispNet
import Nets.MadNet
import Nets.MadNet_fusion

STEREO_FACTORY = {
    Nets.DispNet.DispNet._netName: Nets.DispNet.DispNet,
    Nets.MadNet.MadNet._netName: Nets.MadNet.MadNet,
    Nets.MadNet_fusion.MadNet_fusion._netName: Nets.MadNet_fusion.MadNet_fusion

}

def get_stereo_net(name,args):
    if name not in STEREO_FACTORY:
        raise Exception('Unrecognized network name: {}'.format(name))
    else:
        return STEREO_FACTORY[name](**args)