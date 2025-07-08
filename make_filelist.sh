BASE_DIR="data/stepho_renderings/reflections_4"
DATA_SPLITS_DIR="data_split/reflection"

TRAIN_DIR="${BASE_DIR}/train"
TRAIN_MASK_DIR="${BASE_DIR}/train_masks"

VAL_DIR="${BASE_DIR}/val"
VAL_MASK_DIR="${BASE_DIR}/val_masks"

DST_TRAIN_DIR="${BASE_DIR}/train_merged"
DST_VAL_DIR="${BASE_DIR}/val_merged"

mkdir -p ${DST_TRAIN_DIR}
mkdir -p ${DST_VAL_DIR}

ls ${TRAIN_DIR}/*.jpg | grep -v _cut.jpg | grep -v _surfaceMask.jpg | awk -F / '{print $5}' | awk -F . '{print $1}' |  while read fname;  do echo $fname >> fnames.txt; done
cat fnames.txt | while read fname;  do printf "${fname}.jpg\t${fname}_mask.jpg\n" >> $DATA_SPLITS_DIR/train.txt; done

# Copy the images and masks to the merged directory
echo "Copying images and masks to ${DST_TRAIN_DIR}"
cat fnames.txt | while read fname;  do cp ${TRAIN_DIR}/${fname}.jpg ${DST_TRAIN_DIR}/${fname}_orig.jpg ; cp ${TRAIN_DIR}/${fname}_cut.jpg ${DST_TRAIN_DIR}/${fname}_cut.jpg; done
cat fnames.txt | while read fname;  do cp ${TRAIN_MASK_DIR}/${fname}_mask.jpg ${DST_TRAIN_DIR}/${fname}_mask.jpg ; cp ${TRAIN_MASK_DIR}/${fname}_reflection.jpg ${DST_TRAIN_DIR}/${fname}_reflection.jpg; done

rm fnames.txt

ls ${VAL_DIR}/*.jpg | grep -v _cut.jpg | grep -v _surfaceMask.jpg | awk -F / '{print $5}' | awk -F . '{print $1}' |  while read fname;  do echo $fname >> fnames.txt; done
cat fnames.txt | while read fname;  do printf "${fname}.jpg\t${fname}_mask.jpg\n" >> $DATA_SPLITS_DIR/val.txt; done

# Copy the images and masks to the merged directory
echo "Copying images and masks to ${DST_VAL_DIR}"
cat fnames.txt | while read fname;  do cp ${VAL_DIR}/${fname}.jpg ${DST_VAL_DIR}/${fname}_orig.jpg ; cp ${VAL_DIR}/${fname}_cut.jpg ${DST_VAL_DIR}/${fname}_cut.jpg; done
cat fnames.txt | while read fname;  do cp ${VAL_MASK_DIR}/${fname}_mask.jpg ${DST_VAL_DIR}/${fname}_mask.jpg ; cp ${VAL_MASK_DIR}/${fname}_reflection.jpg ${DST_VAL_DIR}/${fname}_reflection.jpg; done


rm fnames.txt
