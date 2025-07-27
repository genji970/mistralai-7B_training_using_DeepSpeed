from pytube import YouTube
import os
import traceback
from youtube_list import youtube_list

"""
frames/
├── frame_1/
│   ├── frame_00000.jpg
│   ├── frame_00030.jpg
│   └── ...
├── frame_2/
│   ├── frame_00000.jpg
│   └── ...

"""

def download_youtube_videos(youtube_list, save_dir="content/sample_data/youtube_mp4"):
    os.makedirs(save_dir, exist_ok=True)

    for idx, youtube in enumerate(youtube_list, start=1):
        try:
            yt = YouTube(youtube)
            filename = f"frame_{idx}.mp4"

            stream = yt.streams.filter(progressive=True, file_extension='mp4')\
                               .order_by('resolution').desc().first()

            if stream:
                print(f"[✓] 다운로드: {filename}")
                stream.download(output_path=save_dir, filename=filename)
            else:
                print(f"[✗] 건너뜀 (progressive 스트림 없음): {filename}")

        except Exception as e:
            print(f"[!] 오류 발생: {youtube} → {e}")
            traceback.print_exc()


def extract_frames_from_videos(video_dir="content/sample_data/youtube_mp4",
                               frame_root_dir="frames",
                               every_n_seconds=1):
    os.makedirs(frame_root_dir, exist_ok=True)

    video_files = sorted([
        f for f in os.listdir(video_dir)
        if f.endswith(".mp4") and f.startswith("frame_")
    ])

    for video_file in video_files:
        video_path = os.path.join(video_dir, video_file)
        video_title = os.path.splitext(video_file)[0]
        out_dir = os.path.join(frame_root_dir, video_title)
        os.makedirs(out_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_interval = fps * every_n_seconds
        frame_num = 0

        print(f"[✓] 프레임 추출 중: {video_file} (FPS={fps})")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_num % frame_interval == 0:
                frame_filename = f"frame_{frame_num:05d}.jpg"
                frame_path = os.path.join(out_dir, frame_filename)
                cv2.imwrite(frame_path, frame)
            frame_num += 1

        cap.release()
        print(f"[✓] 완료: {out_dir}")
