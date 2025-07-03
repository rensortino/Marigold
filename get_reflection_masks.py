import os

import cv2
import numpy as np
import skimage
from PIL import Image

image_dir = "data/shadow-gen/stepho_renderings/reflections_4/train/"
outdir = image_dir.replace("train", "reflection_masks_otsu")

stack_images = False
overlay_images = True

if not os.path.exists(outdir):
    os.makedirs(outdir)

for image_path in os.listdir(image_dir):
    if image_path.endswith("_cut.jpg") or image_path.endswith("_surfaceMask.jpg"):
        continue

    im_arr = cv2.imread(
        os.path.join(image_dir, image_path)
    )  # Ensure the image is loaded
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
    # We use THRESH_BINARY_INV because the reflections are darker than the background
    # ret, otsu_thresh = cv2.threshold(
    #     gray_masked_im_otsu, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    # )
    otsu_thresh = skimage.filters.threshold_multiotsu(gray_masked_im_otsu, classes=3)
    reflection_thresh = otsu_thresh[1]  # Get the second threshold for binary mask
    refl_mask = gray_masked_im_otsu < reflection_thresh
    refl_mask = refl_mask.astype(np.uint8) * 255  # Convert boolean mask

    if (masked_im * refl_mask[...,None])[:,:,0][refl_mask != 0].mean() < 85:
        # If the mean value of the reflection area is too low, we use a different threshold
        # This should avoid taking too much background in the reflection mask
        reflection_thresh = otsu_thresh[0]  # Get the first threshold for binary mask
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


    # Save the image and the mask for comparison
    # Combine the original image and the mask for comparison
    if stack_images:
        if not os.path.exists("stacked_images"):
            os.makedirs("stacked_images")
        comparison_image = np.hstack((im_arr, cv2.cvtColor(dilated_mask, cv2.COLOR_GRAY2RGB)))
        comparison_image_path = os.path.join("stacked_images", image_path)
        cv2.imwrite(comparison_image_path, comparison_image)

    if overlay_images:
        if not os.path.exists("overlay_images"):
            os.makedirs("overlay_images")
        original_image = Image.fromarray(im_arr).convert("RGBA")
        mask_overlay = np.full((*dilated_mask.shape, 4), [255,0,0,60], dtype=np.uint8)
        dilated_mask = (dilated_mask / 255).astype(np.uint8)
        mask_overlay = dilated_mask[...,None] * mask_overlay
        mask_image = Image.fromarray(mask_overlay, mode="RGBA")
        img = Image.alpha_composite(original_image, mask_image)
        overlay_image_path = os.path.join("overlay_images", image_path)

        comparison_image = np.hstack((im_arr, np.array(img.convert("RGB"))))
        comparison_image_path = os.path.join("overlay_images", image_path)
        cv2.imwrite(comparison_image_path, comparison_image)