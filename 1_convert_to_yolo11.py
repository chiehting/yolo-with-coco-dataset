import io
import os
import random
import shutil
from pycocotools.coco import COCO
import matplotlib.pyplot as plt

CURRENT_PAHT = os.path.abspath(os.getcwd())
DATASET='coco2017'
TARGET_CATEGORIES = ['bird','person']

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
        
        shutil.copyfile(coco_images_directory + '/' + imgFileName, extracted_img_path + '/' + imgFileName)
        labelFilePath = "{}/{}.txt".format(extracted_labels_path, imgName)
        
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


datesets = [
    { 
        "data_name": 'train2017', "extracted_daving_sub_path": "train", "extracted_max_file": 100 # Definition -1 will not be restricted
    },
    { 
        "data_name": 'val2017', "extracted_daving_sub_path": "val", "extracted_max_file": 30 # Definition -1 will not be restricted
    }
]

for dataset in datesets:
    print(dataset)
    coco_annotations_path = CURRENT_PAHT + "/" + DATASET +"/annotations/instances_{}.json".format(dataset['data_name'])
    coco_images_directory = CURRENT_PAHT + "/" + DATASET +"/{}/".format(dataset['data_name'])
    extracted_saving_path = CURRENT_PAHT + "/extracted_dataset"
    extracted_img_path = "{}/images/{}".format(extracted_saving_path, dataset['extracted_daving_sub_path'])
    extracted_labels_path = "{}/labels/{}".format(extracted_saving_path, dataset['extracted_daving_sub_path'])
    
    os.makedirs(extracted_img_path, exist_ok=True)
    os.makedirs(extracted_labels_path, exist_ok=True)

    # read the data from coco dataset
    coco = COCO(coco_annotations_path)
    cats = coco.loadCats(coco.getCatIds(TARGET_CATEGORIES))
    catsIds = [cat['id'] for cat in cats]
    imgs = coco.loadImgs(coco.getImgIds(catIds=catsIds))[:dataset['extracted_max_file'] if dataset['extracted_max_file'] > 0 else None]
    imgIds = [img['id'] for img in imgs]
    anns = coco.loadAnns(coco.getAnnIds(imgIds=imgIds, catIds=catsIds))

    # random_idx = random.randint(0, len(imgs) - 1)
    # I = io.imread(imgs[random_idx]['coco_url'])
    # plt.axis('off')
    # plt.imshow(I)
    # plt.show()

    customIds = {cat['id']: idx for idx, cat in enumerate(cats)}
    create_classes_file(extracted_saving_path + '/classes.txt', cats, customIds)
    convert_to_yolo_format(customIds, imgs, anns)
