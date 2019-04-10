import cv2
import numpy as np
import os
from keras.utils import to_categorical


def load_image(fname, mode="color", target_size=None):
    if mode == "color":
        img = cv2.imread(fname, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)

    if target_size is not None:
        img = cv2.resize(img, dsize=target_size)

    return img


def image_flip(image, label, left_right_axis=True):
    """ Flip the src and label images
    :param image: numpy array
    :param label: numpy array
    :param left_right_axis: True / False
    :return: the processed numpy arrays
    """
    axis = 1 if left_right_axis==True else 0
    image = np.flip(image, axis=axis)
    label = np.flip(label, axis=axis)
    return image, label


def image_randomcrop(image, label, crop_height, crop_width):
    """ Random Crop the src and label images
    :param image: numpy array
    :param label: numpy array
    :param crop_height: target.txt height
    :param crop_width: target.txt width
    :return: the processed numpy arrays
    """
    assert image.shape[1]>=crop_width and image.shape[0]>=crop_height
    image_width = image.shape[1]
    image_height = image.shape[0]

    x = np.random.randint(0, image_width-crop_width+1)
    y = np.random.randint(0, image_height-crop_height+1)

    return image[y:y+crop_height, x:x+crop_width], \
           label[y:y+crop_height, x:x+crop_width]


def apply_random_augmentation(x, y, args):
    if "flip_x" in args.augmentations:
        if np.random.random() > 0.5:
            x, y = image_flip(x, y, True)
    if "flip_y" in args.augmentations:
        if np.random.random() > 0.5:
            x, y = image_flip(x, y, False)
    if "random_crop" in args.augmentations:
        if np.random.random() > 0.5:
            x, y = image_randomcrop(x, y, args.image_height, args.image_width)

    return x, y


def plot_image_label(image, label):
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    grid_spec = gridspec.GridSpec(1, 2, width_ratios=[6, 6])

    plt.subplot(grid_spec[0])
    plt.imshow(image)
    plt.axis('off')
    plt.title('image', fontdict={"fontsize": 12})

    plt.subplot(grid_spec[1])
    plt.imshow(label)
    plt.axis('off')
    plt.title('image', fontdict={"fontsize": 12})

    plt.tight_layout()
    plt.show()


def segmentation_generator(base_urls, args, shuffle=True):
    while True:
        if shuffle:
            np.random.shuffle(base_urls)
        images = []
        labels = []
        batch_i = 0
        for i in range(len(base_urls)):
            base_url = base_urls[i]
            batch_i += 1
            # print(os.path.join(args.data_dir+"/"+args.image_dir, base_url+args.image_suffix))
            img = load_image(os.path.join(args.data_dir+"/"+args.image_dir, base_url+args.image_suffix),
                             mode=args.color_mode,
                             target_size=(args.image_width, args.image_height))

            label = load_image(os.path.join(args.data_dir+"/"+args.label_dir, base_url+args.label_suffix),
                               mode="gray",
                               target_size=(args.image_width, args.image_height))
            if args.augmentations is not None:
                img, label = apply_random_augmentation(img, label, args)

            #plot_image_label(img, label)
            label = to_categorical(label, args.n_class)
            images.append(img)
            labels.append(label)
            if batch_i % args.batch_size == 0:
                train_data = np.array(images)
                train_label = np.array(labels)
                yield (train_data, train_label)
                images = []
                labels = []
                batch_i = 0


def segmentation_val_generator(base_urls, args):
    while True:
        images = []
        labels = []
        batch_i = 0
        start_index = np.random.randint(0, len(base_urls) - args.batch_size)
        for i in range(start_index, len(base_urls)):
            base_url = base_urls[i]
            batch_i += 1
            img = load_image(os.path.join(args.data_dir+"/"+args.image_dir, base_url + args.image_suffix),
                             mode=args.color_mode,
                             target_size=(args.image_width, args.image_height))

            label = load_image(os.path.join(args.data_dir+"/"+args.label_dir, base_url + args.label_suffix),
                               mode="gray",
                               target_size=(args.image_width, args.image_height))

            #plot_image_label(img, label)
            label = to_categorical(label, args.n_class)
            images.append(img)
            labels.append(label)
            if batch_i % args.batch_size == 0:
                val_data = np.array(images)
                val_label = np.array(labels)
                yield (val_data, val_label)
                images = []
                labels = []
                batch_i = 0


def get_image_label(image_dir, label_dir, base_urls, args, shuffle=True):
    if shuffle:
        np.random.shuffle(base_urls)
    images = []
    labels = []
    batch_i = 0
    for i in range(len(base_urls)):
        base_url = base_urls[i]
        batch_i += 1
        # print(os.path.join(image_dir, base_url + args.image_suffix))
        img = load_image(os.path.join(image_dir, base_url + args.image_suffix),
                         mode=args.color_mode,
                         target_size=(args.image_width, args.image_height))

        label = load_image(os.path.join(label_dir, base_url + args.label_suffix),
                           mode="gray",
                           target_size=(args.image_width, args.image_height))
        if args.augmentations is not None:
            img, label = apply_random_augmentation(img, label, args)

        # plot_image_label(img, label)
        label = to_categorical(label, args.n_class)
        images.append(img)
        labels.append(label)
        if batch_i % args.batch_size == 0:
            train_data = np.array(images)
            train_label = np.array(labels)
            return train_data, train_label


def get_images(image_dir, base_urls, args, shuffle=True):
    if shuffle:
        np.random.shuffle(base_urls)
    images = []
    batch_i = 0
    for i in range(len(base_urls)):
        base_url = base_urls[i]
        batch_i += 1
        img = load_image(os.path.join(image_dir, base_url + args.image_suffix),
                         mode=args.color_mode,
                         target_size=(args.image_width, args.image_height))

        # plot_image_label(img, label)
        images.append(img)
        if batch_i % args.batch_size == 0:
            train_data = np.array(images)
            return train_data

