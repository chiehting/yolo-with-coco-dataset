# 從頭開始訓練 YOLOv8
from ultralytics import YOLO

# 初始化一個新的模型（使用 yolo11n.yaml 架構，載 yolo11n.pt 權重）
# model = YOLO("yolo11n.yaml")
model = YOLO("yolo11s.yaml").load("yolo11s.pt")


# 開始訓練
# https://docs.ultralytics.com/modes/train/#train-settings
model.train(
    data='2_coco_slef_defined.yaml',
    epochs=30,  # 從零訓練通常需要更多的訓練週期
    imgsz=640,
    batch=16,
    save=True,
    name='train',
    # pretrained=False,  # 關鍵參數：不使用預訓練權重
    device="mps",
    # patience=20,            # 早停策略：20個週期無改善則停止
    cos_lr=True,            # 餘弦學習率調度
    lr0=0.01,               # 初始學習率
    lrf=0.01,               # 最終學習率因子
    # weight_decay=0.0005,    # 權重衰減
    # warmup_epochs=5,        # 預熱週期
    # close_mosaic=10,        # 最後10個週期關閉馬賽克增強
    # augment=True,           # 啟用數據增強
    cache=True,             # 緩存圖像以加速訓練
    # workers=4,              # 數據加載工作線程
)