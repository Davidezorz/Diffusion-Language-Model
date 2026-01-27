import torch
import matplotlib.pyplot as plt
import datetime
import pydoc

def getDevice(device: str = None) -> str:                                       #   ╭ Device auto
    """Selects the best available device or verifies the requested one."""      # ◀─┤ detection  
    if (device in [None, 'cuda']) and torch.cuda.is_available():                #   │
        return 'cuda'                                                           #   │
    if (device in [None, 'mps']) and torch.backends.mps.is_available():         #   │
        return 'mps'                                                            #   ╰
    return 'cpu'
    


def setupMatplotlib():
    plt.style.use('ggplot')
    plt.rcParams['axes.facecolor'] = '#FFFFFF'
    plt.rcParams['grid.linewidth'] = 1
    plt.rcParams['grid.color'] = '#F9F9F9'




def numberOfparameters(model):
    n = sum([p.numel() for p in model.parameters()])
    return n




