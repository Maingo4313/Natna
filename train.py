
from ultralytics import YOLO

# Load a model
model = YOLO("License01 10.pt")

# Train the model
train_results = model.train(
    data="pap/data.yaml",  # path to dataset YAML
    epochs=10,  # number of training epochs
    lr0=0.001,           # ใช้ lr0 แทน lr
    imgsz=640,  # training image size
    device="0",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
    amp=False,
    workers=0
)

# Evaluate model performance on the validation set
metrics = model.val()