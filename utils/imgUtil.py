import numpy as np
import matplotlib.pyplot as plt
from torchvision import utils


def show(images, title="", text="", block=False):
    images = utils.make_grid(images, normalize=True)
    fig, ax = plt.subplots(1, squeeze=False, frameon=False, dpi=300)
    ax[0][0].imshow(np.transpose(images.cpu().detach().numpy(), (1,2,0)), interpolation='nearest')
    ax[0][0].axis('off')
    ax[0][0].set_title(title)
    plt.text(122, -20, text,
             horizontalalignment='center',
             verticalalignment='bottom')
    plt.show(block=block)
    
    
def prediction_report(images, label, title=""):
    images = utils.make_grid(images, normalize=True)
    fig, ax = plt.subplots(1, squeeze=False, frameon=False, dpi=300)