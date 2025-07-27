from youtube_download import download_youtube_videos, extract_frames_from_videos
from youtube_list import youtube_list  # 유튜브 링크 리스트
from subtitle import ocr_all_frames
from object_detection import detect_all_objects

# 유튜브 영상 다운로드
download_youtube_videos(youtube_list, save_dir="my_data/mp4")

# 프레임 추출 (1초마다)
extract_frames_from_videos(video_dir="my_data/mp4", frame_root_dir="my_data/frames", every_n_seconds=1)


ocr_output = ocr_all_frames("/workspace/frames/frame_1")
for item in ocr_output:
    print(item["frame"], "→", item["text"])

object_info = detect_all_objects("/workspace/frames/frame_1")
for item in object_info:
    print(item["frame"], "→", item["objects"])