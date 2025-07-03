import argparse
import os

import cv2
import numpy as np
import skimage
from PIL import Image


def main(image_dir, outdir, overlay_images=True):
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    for image_path in os.listdir(image_dir):
        if image_path.endswith("_cut.jpg") or image_path.endswith("_surfaceMask.jpg"):
            continue

        im_arr = cv2.imread(os.path.join(image_dir, image_path))
        im_arr = cv2.cvtColor(im_arr, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

        cut_path = image_path.replace(".jpg", "_cut.jpg")
        cut = cv2.imread(os.path.join(image_dir, cut_path))
        cut = cv2.cvtColor(cut, cv2.COLOR_BGR2RGB)
        car_mask = (cut < 255).astype(np.uint8)[:, :, 0]
        car_mask = cv2.dilate(car_mask, np.ones((3, 3), np.uint8), iterations=2)
        car_mask = cv2.erode(car_mask, np.ones((3, 3), np.uint8), iterations=3)

        bg_color = im_arr[10, 10]

        masked_im = im_arr.copy()
        masked_im[car_mask != 0] = bg_color

        # Convert the masked image to grayscale
        gray_masked_im_otsu = cv2.cvtColor(masked_im, cv2.COLOR_RGB2GRAY)

        # Apply Otsu's thresholding
        otsu_thresh = skimage.filters.threshold_multiotsu(
            gray_masked_im_otsu, classes=3
        )
        reflection_thresh = otsu_thresh[1]  # Get the second threshold for binary mask
        refl_mask = gray_masked_im_otsu < reflection_thresh
        refl_mask = refl_mask.astype(np.uint8) * 255  # Convert boolean mask

        if (masked_im * refl_mask[..., None])[:, :, 0][refl_mask != 0].mean() < 85:
            # If the mean value of the reflection area is too low, we use a different threshold
            reflection_thresh = otsu_thresh[
                0
            ]  # Get the first threshold for binary mask
            refl_mask = gray_masked_im_otsu < reflection_thresh
            refl_mask = refl_mask.astype(np.uint8) * 255  # Convert boolean mask

        opened_mask = cv2.morphologyEx(
            refl_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=8
        )
        closed_mask = cv2.morphologyEx(
            opened_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=8
        )
        dilated_mask = cv2.dilate(closed_mask, np.ones((3, 3), np.uint8), iterations=12)

        cv2.imwrite(
            os.path.join(outdir, image_path),
            (dilated_mask).astype(np.uint8),
        )

        if overlay_images:
            if not os.path.exists("overlay_images"):
                os.makedirs("overlay_images")
            original_image = Image.fromarray(im_arr).convert("RGBA")
            mask_overlay = np.full(
                (*dilated_mask.shape, 4), [255, 0, 0, 60], dtype=np.uint8
            )
            dilated_mask_bin = (dilated_mask / 255).astype(np.uint8)
            mask_overlay = dilated_mask_bin[..., None] * mask_overlay
            mask_image = Image.fromarray(mask_overlay, mode="RGBA")
            img = Image.alpha_composite(original_image, mask_image)

            comparison_image = np.hstack((im_arr, np.array(img.convert("RGB"))))
            comparison_image_path = os.path.join("overlay_images", image_path)
            cv2.imwrite(comparison_image_path, comparison_image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate reflection masks using Otsu thresholding."
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Directory containing input images.",
    )
    parser.add_argument(
        "--outdir", type=str, default="output", help="Directory to save output masks."
    )
    parser.add_argument(
        "--overlay_images", action="store_true", help="Save overlay images."
    )
    args = parser.parse_args()

    main(
        args.image_dir,
        args.outdir,
        overlay_images=args.overlay_images,
    )
