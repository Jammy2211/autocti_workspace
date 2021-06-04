import numpy as np
import os

import pickle
import autocti as ac

"""Load the HST ACS dataset"""
dataset_path = f"dataset/examples/acs/lines"
dataset_name = "jc0a01h8q"

lines = ac.LineCollection()
lines.load(filename=f"{dataset_path}/{dataset_name}.pickle")

ci_imagings = []

delta_function_index = 9

for i in range(lines.n_lines):
    ci_pattern = ac.ci.CIPatternUniform(
        regions=[(delta_function_index, delta_function_index + 1, 0, 1)],
        normalization=lines.lines[i].line[delta_function_index],
    )

    """
    The location attribute is the location of the warm pixel. Each cut-out line is 18 pixels, including 9 pixels before 
    the peak flux in the warm pixel itself and 8 pixels after which include its trail.

    We need to compute the readout_offset of this warm pixel image, which is the distance from the first
    pixel in the image (e.g. the pixel which is 9 pixel in front of the warm pixel itself) to the serial
    register. Thus, we subtract 9 from the warm pixel location.
    """

    readout_offsets = (
        lines.lines[i].location[0] - delta_function_index,
        lines.lines[i].location[1],
    )

    image = ac.ci.CIFrame.manual(
        array=lines.lines[i].line[:, None],
        pixel_scales=0.05,
        ci_pattern=ci_pattern,
        header=ac.Header(readout_offsets=readout_offsets),
    )

    noise_map = ac.ci.CIFrame.manual(
        # array=lines.lines[i].noise_map[:,None],
        array=np.ones(shape=image.shape_2d),
        pixel_scales=0.05,
        ci_pattern=ci_pattern,
        header=ac.Header(readout_offsets=readout_offsets),
    )

    ci_pre_cti = ci_pattern.ci_pre_cti_from(
        shape_2d=image.shape_2d, pixel_scales=image.pixel_scales
    )

    ci_imagings.append(
        ac.ci.CIImaging(
            image=image,
            noise_map=noise_map,
            ci_pre_cti=ci_pre_cti,
            name=lines.lines[i].name,
        )
    )

"""Serialize the `CIImaging`'s so we can load them in the CTI model fitting scripts."""

output_path = f"dataset/examples/acs/lines/"

if not os.path.exists(output_path):
    os.mkdir(output_path)

with open(f"{output_path}/{dataset_name}_ci_imagings.pickle", "wb") as f:
    pickle.dump(ci_imagings, f)
