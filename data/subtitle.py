import pytesseract
from PIL import Image
import os
from concurrent.futures import ProcessPoolExecutor

def ocr_frame(path):
    fname = os.path.basename(path)
    img = Image.open(path)
    text = pytesseract.image_to_string(img, lang='kor+eng')
    return {"frame": fname, "text": text.strip()}


def ocr_all_frames(frame_dir):
    """
    Args:
        frame_dir (str): 프레임 이미지(.jpg)들이 저장된 폴더 경로

    Returns:
        List[Dict[str, str]]: [{'frame': 파일명, 'text': OCR결과}, ...]
    """
    img_paths = sorted([
        os.path.join(frame_dir, fname)
        for fname in os.listdir(frame_dir)
        if fname.endswith(".jpg")
    ])

    with ProcessPoolExecutor() as executor:
        ocr_results = list(executor.map(ocr_frame, img_paths))

    return ocr_results
