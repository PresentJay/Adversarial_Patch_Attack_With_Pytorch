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
    images = images.clone()
    out = transforms.functional.normalize(images, mean=[0, 0, 0], std=[1/std[0], 1/std[1], 1/std[2]])
    out = transforms.functional.normalize(out, mean=[-mean[0], -mean[1], -mean[2]], std=[1, 1, 1])
    out = transforms.functional.to_pil_image(out.detach().cpu())
    return out


def to_circle(shape, sharpness = 40):
    x = np.linspace(-1, 1, shape[1])
    y = np.linspace(-1, 1, shape[2])
    xx, yy = np.meshgrid(x, y, sparse=True)
    z = (xx**2 + yy**2) ** sharpness
    
    circle_mask = 1 - np.clip(z, -1, 1)
    circle_mask = np.expand_dims(circle_mask, axis=0)
    circle_mask = np.broadcast_to(circle_mask, shape)

    return circle_mask