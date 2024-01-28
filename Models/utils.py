import matplotlib.pyplot as plt
import numpy as np


def sample_batch(dataset):
    batch = dataset.take(1).get_single_element()
    if isinstance(batch, tuple):
        batch = batch[0]
    return batch.numpy()


# def display(
#     images, n=10, size=(20, 3), cmap="gray_r", as_type="float32", save_to=None
# ):
#     """
#     Displays n random images from each one of the supplied arrays.
#     """
#     if images.max() > 1.0:
#         images = images / 255.0
#     elif images.min() < 0.0:
#         images = (images + 1.0) / 2.0

#     plt.figure(figsize=size)
#     for i in range(n):
#         _ = plt.subplot(1, n, i + 1)
#         plt.imshow(images[i].astype(as_type), cmap=cmap)
#         plt.axis("off")

#     if save_to:
#         plt.savefig(save_to)
#         print(f"\nSaved to {save_to}")

#     plt.show()


def display(
    images, rows=5, columns=5, size=(15, 15), cmap="gray_r", as_type="float32", save_to=None
):
    """
    Displays a grid of images from the supplied array and saves the figure as a PDF.
    """
    
    assert rows * columns == len(images), "Number of rows times columns must equal the number of images."

    if images.max() > 1.0:
        images = images / 255.0
    elif images.min() < 0.0:
        images = (images + 1.0) / 2.0

    plt.figure(figsize=size)

    for i in range(len(images)):
        _ = plt.subplot(rows, columns, i + 1)
        plt.imshow(images[i].astype(as_type), cmap=cmap)
        plt.axis("off")

    if save_to:
        plt.savefig(save_to, format='pdf', bbox_inches='tight')
        print(f"\nSaved to {save_to}")

    # Display the figure
    plt.show()
