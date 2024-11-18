from ultralytics import YOLO

# import the YOLO version 8 model for object detection
model = YOLO('models/best.pt')

results = model.predict('input_videos/bundesliga.mp4', save=True)
print(results[0])  # first frame
print('##############################')
for box in results[0].boxes:
    print(box)

