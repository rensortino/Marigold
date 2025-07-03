BASE_DIR="data/stepho_renderings/reflections_4"
TRAIN_DIR="${BASE_DIR}/train"
MASK_DIR="${BASE_DIR}/reflection_masks_otsu"
DATA_SPLITS_PATH="data_split/reflection/train.txt"

MERGED_IMAGE_DIR="${BASE_DIR}/train_merged"
mkdir -p ${MERGED_IMAGE_DIR}

ls ${TRAIN_DIR}/*.jpg | grep -v _cut.jpg | grep -v _surfaceMask.jpg | awk -F / '{print $5}' | awk -F . '{print $1}' |  while read fname;  do echo $fname >> fnames.txt; done
cat fnames.txt | while read fname;  do printf "${fname}.jpg\t${fname}_mask.jpg\n" >> $DATA_SPLITS_PATH; done

# Copy the images and masks to the merged directory
echo "Copying images and masks to ${MERGED_IMAGE_DIR}"
cat fnames.txt | while read fname;  do cp ${TRAIN_DIR}/${fname}.jpg ${MERGED_IMAGE_DIR}/${fname}.jpg ; done
cat fnames.txt | while read fname;  do cp ${MASK_DIR}/${fname}.jpg ${MERGED_IMAGE_DIR}/${fname}_mask.jpg ; done

rm fnames.txt