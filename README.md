# COCO 到 YOLO 格式轉換工具

這個工具用於將 COCO 2017 資料集中的特定類別（預設為鳥類）轉換為 YOLO 格式，以便用於物件偵測訓練。

## 功能特點

- 從 COCO 資料集中提取特定類別的圖像和標註
- 將 COCO 格式的標註轉換為 YOLO 格式
- 支援限制提取的圖像數量
- 自動建立所需的目錄結構
- 生成類別映射文件

## 準備 COCO 2017 資料集

1. **下載 COCO 2017 資料集**
   - 訪問 [COCO 官方網站](https://cocodataset.org/#download) 或直接使用以下連結下載：
     - [2017 訓練集圖像 (18GB)](http://images.cocodataset.org/zips/train2017.zip)
     - [2017 驗證集圖像 (1GB)](http://images.cocodataset.org/zips/val2017.zip)
     - [2017 標註檔案 (241MB)](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)

2. **建立正確的目錄結構**
   - 在專案根目錄下建立 `coco2017` 資料夾
   - 解壓縮下載的檔案，並按照以下結構放置：

   ```txt
   專案根目錄/
   └── coco2017/
       ├── annotations/
       │   ├── instances_train2017.json
       │   └── instances_val2017.json
       │   └── (其他標註檔案)
       ├── train2017/
       │   └── (所有訓練集圖像)
       └── val2017/
           └── (所有驗證集圖像)
   ```

3. **解壓縮指令參考**

   ```bash
   # 建立目錄
   mkdir -p coco2017

   # 解壓縮標註檔案
   unzip annotations_trainval2017.zip -d coco2017/

   # 解壓縮訓練集圖像
   unzip train2017.zip -d coco2017/

   # 解壓縮驗證集圖像
   unzip val2017.zip -d coco2017/
   ```

## 目錄結構

轉換後的資料集將具有以下結構：

```txt
extracted_dataset/
├── images/
│   └── train/  (或 val/)
│       └── [圖像文件]
├── labels/
│   └── train/  (或 val/)
│       └── [標註文件.txt]
└── classes.txt
```

## 使用方法

1. 確保已安裝所需的依賴套件：

   ```bash
   pip install pycocotools matplotlib scikit-image
   ```

2. 下載並準備 COCO 2017 資料集（如上所述）

3. 設定參數：
   - `DATA_NAME`: 選擇 'train2017' 或 'val2017'
   - `EXTRACTED_SAVING_SUB_PATH`: 輸出子目錄名稱 ('train' 或 'val')
   - `EXTRACTED_MAX_FILE`: 限制提取的最大文件數量 (-1 表示不限制)
   - `TARGET_CATEGORIES`: 要提取的類別列表，例如 ['bird']

4. 執行腳本：

   ```bash
   python convert_to_yolo11.py
   ```

## 轉換格式說明

COCO 格式的標註：

```txt
[x, y, width, height]  # 左上角座標及寬高
```

YOLO 11 格式的標註：

```txt
<class_id> <x_center> <y_center> <width> <height>  # 中心點座標及寬高，所有值歸一化至 0-1
```

## 輸出文件

- **圖像文件**：複製到 `extracted_dataset/images/train/` 或 `extracted_dataset/images/val/`
- **標註文件**：轉換後的 YOLO 格式標註存儲在 `extracted_dataset/labels/train/` 或 `extracted_dataset/labels/val/`
- **類別文件**：`extracted_dataset/classes.txt` 包含類別 ID 和名稱的映射

## 常見問題

1. **無法讀取圖像或標註檔案**
   - 確認目錄結構是否正確
   - 檢查檔案權限是否足夠
   - 驗證標註檔案是否完整解壓

2. **記憶體不足**
   - 減少 `EXTRACTED_MAX_FILE` 的值
   - 分批處理資料集

3. **無法顯示圖像**
   - 確認 matplotlib 正確安裝
   - 在支援圖形界面的環境中執行

## 注意事項

- 如果圖像沒有對應的標註，會生成一個空的標註文件
- 類別 ID 會從 0 開始重新編號
- 確保有足夠的磁碟空間存儲提取的資料集
- 完整的 COCO 訓練集約需 18GB 空間，驗證集約需 1GB 空間

## 範例輸出

執行腳本後，會顯示一張隨機選擇的圖像，並輸出轉換的統計信息：

```txt
總共轉換了 X 個 txt 檔案，包含 Y 個標註
```
