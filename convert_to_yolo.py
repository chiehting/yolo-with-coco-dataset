import os
import shutil
from pycocotools.coco import COCO
import random
import skimage.io as io
import matplotlib.pyplot as plt

CURRENT_PAHT = os.path.abspath(os.getcwd())
DATA_NAME='train2017'
EXTRACTED_SAVING_SUB_PATH = "train"
# DATA_NAME='val2017'
# EXTRACTED_SAVING_SUB_PATH = "val"

DATASET='coco2017'
COCO_ANNOTATIONS_PATH = CURRENT_PAHT + "/" + DATASET +"/annotations/instances_{}.json".format(DATA_NAME)
COCO_IMAGES_DIRECTORY = CURRENT_PAHT + "/" + DATASET +"/{}/".format(DATA_NAME)
EXTRACTED_MAX_FILE = 50 # Definition -1 will not be restricted
EXTRACTED_SAVING_PATH = CURRENT_PAHT + "/extracted_dataset"
EXTRACTED_IMG_PATH = "{}/images/{}".format(EXTRACTED_SAVING_PATH, EXTRACTED_SAVING_SUB_PATH)
EXTRACTED_LABELS_PATH = "{}/labels/{}".format(EXTRACTED_SAVING_PATH, EXTRACTED_SAVING_SUB_PATH)
TARGET_CATEGORIES = ['bird']

os.makedirs(EXTRACTED_IMG_PATH, exist_ok=True)
os.makedirs(EXTRACTED_LABELS_PATH, exist_ok=True)

def create_classes_file(filePath, cats, customIds):
    with open(filePath, 'w') as f:
        for cat in cats:
            f.write("{}: {}\n".format(customIds[cat['id']], cat['name']))

def convert_to_yolo_format(customIds, imgs, anns):
    txt_count = 0
    anns_by_img = {}

    # 聚合照片的 labels
    for ann in anns:
        imgID = ann['image_id']
        if imgID not in anns_by_img:
            anns_by_img[imgID] = []
        anns_by_img[imgID].append(ann)

    # 寫 labels.txt 檔案
    for img in imgs:
        imgId = img['id']
        imgName = os.path.splitext(img['file_name'])[0]
        imgFileName = img['file_name']
        imgHeight = img['height']
        imgWidth = img['width']
        
        shutil.copyfile(COCO_IMAGES_DIRECTORY + '/' + imgFileName, EXTRACTED_IMG_PATH + '/' + imgFileName)
        labelFilePath = "{}/{}.txt".format(EXTRACTED_LABELS_PATH, imgName)
        
        # 沒有標記的照片只新增空檔案
        if imgId not in anns_by_img:
            with open(labelFilePath, 'w') as f:
                pass
            txt_count += 1
            continue

        # 處理圖片的標註
        with open(labelFilePath, 'w') as f:
            for ann in anns_by_img[imgId]:
                # 自定義 index
                customCatID = customIds[ann['category_id']]

                # 獲取邊界框座標 (COCO 格式: [x, y, width, height])
                bbox = ann['bbox']
                x, y, w, h = bbox

                # 轉換為 YOLO 格式 (中心點座標和寬高，歸一化到 0-1)
                x_center = (x + w / 2) / imgWidth
                y_center = (y + h / 2) / imgHeight
                width = w / imgWidth
                height = h / imgHeight
                
                # 寫入 YOLO 格式
                f.write(f"{customCatID} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        txt_count += 1
    print(f"總共轉換了 {txt_count} 個 txt 檔案，包含 {len(anns)} 個標註")

# read the data from coco dataset
coco = COCO(COCO_ANNOTATIONS_PATH)
cats = coco.loadCats(coco.getCatIds(TARGET_CATEGORIES))
catsIds = [cat['id'] for cat in cats]
imgs = coco.loadImgs(coco.getImgIds(catIds=catsIds))[:EXTRACTED_MAX_FILE if EXTRACTED_MAX_FILE > 0 else None]
imgIds = [img['id'] for img in imgs]
anns = coco.loadAnns(coco.getAnnIds(imgIds=imgIds, catIds=catsIds))

# show the first image
random_idx = random.randint(0, len(imgs) - 1)
I = io.imread(imgs[random_idx]['coco_url'])
plt.axis('off')
plt.imshow(I)
plt.show()

customIds = {cat['id']: idx for idx, cat in enumerate(cats)}
create_classes_file(EXTRACTED_SAVING_PATH + '/classes.txt', cats, customIds)
convert_to_yolo_format(customIds, imgs, anns)