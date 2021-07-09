import numpy as np
import matplotlib.pyplot as plt
from torchvision import utils, transforms


def show_tensor(images, title="", text="", block=False):
    images = utils.make_grid(images)
    
    fig, ax = plt.subplots(1, squeeze=False, frameon=False, dpi=300)
    ax[0][0].imshow(np.transpose(images.cpu().detach().numpy(), (1,2,0)), interpolation='nearest')
    ax[0][0].axis('off')
    ax[0][0].set_title(title)
    plt.text(122, -20, text,
             horizontalalignment='center',
             verticalalignment='bottom')
    plt.show(block=block)


def show_batch_data(images, block, normalize=False, title=""):
    batch = utils.make_grid(images.cpu(), nrow=4, padding=20, normalize=normalize)
    plt.imshow(batch.permute(1,2,0))
    plt.axis('off')
    plt.title(title)
    plt.show(block=block)


def tensor_to_PIL(images, mean, std):
    image = image.clone()
    out = transforms.Functional.normalize(image, mean=[0, 0, 0], std=[1/std[0], 1/std[1], 1/std[2]])
    out = transforms.Functional.normalize(out, mean=[-mean[0], -mean[1], -mean[2]], std=[1, 1, 1])
    out = transforms.Functional.to_pil_image(out.detach().cpu())
    return out

