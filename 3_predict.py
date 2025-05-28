from ultralytics import YOLO
import os
import random

# 取得上一個訓練的結果（模型權重）
# 假設你使用 YOLO 預設的訓練流程，模型權重通常會儲存在 runs/detect/train/weights/last.pt
model = YOLO("runs/detect/train/weights/last.pt")

# Run inference on 'bus.jpg' with arguments
image_dir = "extracted_dataset/images/train"
image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
image_path = os.path.join(image_dir, random.choice(image_files)) if image_files else None
results = model.predict(image_path, save=True, imgsz=320, conf=0.5)
if results and hasattr(results[0], 'show'):
    results[0].show()  # 直接顯示預測結果的圖片（需有GUI環境）
else:
    print("Prediction complete. Saved results to 'runs/detect/predict'.")

