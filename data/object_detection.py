import os
from concurrent.futures import ProcessPoolExecutor

def detect_objects(path):
    from ultralytics import YOLO  # 각 프로세스 내에서 import
    model = YOLO("yolov8l.pt")    # 각 프로세스 내에서 로드
    fname = os.path.basename(path)
    result = model(path)
    labels = set([model.names[int(box.cls)] for box in result[0].boxes])
    return {"frame": fname, "objects": list(labels)}


def detect_all_objects(frame_dir):
    """
    Args:
        frame_dir (str): 프레임 이미지(.jpg)들이 저장된 폴더 경로

    Returns:
        List[Dict[str, Any]]: [{'frame': 파일명, 'objects': ['person', 'car', ...]}, ...]
    """
    img_paths = sorted([
        os.path.join(frame_dir, fname)
        for fname in os.listdir(frame_dir)
        if fname.endswith(".jpg")
    ])

    with ProcessPoolExecutor() as executor:
        results = list(executor.map(detect_objects, img_paths))

    return results
