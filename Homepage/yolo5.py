def yolo(img):
    # YOLOv5
    import torch
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    # 추론
    results = model(img)
    # 결과
    # results.print()
    # results.show()
    results.save('exp/')
    # Save image to 'runs\detect\exp'
    # results.xyxy[0]  # 예측 (tensor)
    yolodf = results.pandas().xyxy[0]  # 예측 (pandas)
    yolo_result = list(yolodf['name'])
    return yolo_result

    # 결과 박스친 이미지로 보여줄 필요가 있을 경우 아래 코드 사용
    # Image.open('exp/zoo.jpg')