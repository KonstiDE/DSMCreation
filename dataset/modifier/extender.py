import numpy as np

from PIL import Image


def mirrow_extrapolate(image, thickness):
    left_image = get_side(image, thickness, 'l')
    right_image = get_side(image, thickness, 'r')

    horizontal = Image.new(image.mode, size=(image.width + 2 * thickness, image.height))
    horizontal.paste(left_image, (0, 0))
    horizontal.paste(image, (left_image.width, 0))
    horizontal.paste(right_image, (left_image.width + image.width, 0))

    tops_image = get_side(horizontal, thickness, 't')
    bottom_image = get_side(horizontal, thickness, 'b')

    full = Image.new(horizontal.mode, size=(horizontal.width, horizontal.height + 2 * thickness))
    full.paste(tops_image, (0, 0))
    full.paste(horizontal, (0, tops_image.height))
    full.paste(bottom_image, (0, tops_image.height + horizontal.height))

    return np.array(full)


def get_side(image, thickness, skyline):
    px = image.load()

    side = []

    if skyline == 'l':
        for i in range(0, image.height):
            for t in range(thickness):
                side.append(px[thickness - t, i])
    elif skyline == 'r':
        for i in range(0, image.height):
            for t in range(thickness):
                side.append(px[image.width - 1 - t, i])
    elif skyline == 't':
        for t in range(thickness):
            for i in range(0, image.width):
                side.append(px[i, thickness - t])
    elif skyline == 'b':
        for t in range(thickness):
            for i in range(0, image.width):
                side.append(px[i, image.height - 1 - t])
    else:
        raise AssertionError("skyline " + skyline + " not existing. Use l, r, t or b instead.")

    if skyline == 'r' or skyline == 'l':
        side_image = Image.new(image.mode, size=(thickness, image.height))
    else:
        side_image = Image.new(image.mode, size=(image.width, thickness))

    side_image.putdata(side)

    return side_image
