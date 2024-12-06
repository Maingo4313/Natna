from ultralytics import YOLO

model = YOLO("License01 09.pt")


results = model("test/004.jpg")
results[0].show()