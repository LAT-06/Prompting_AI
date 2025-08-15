import cv2
import os
import sys
from pathlib import Path

def extract_keyframes(video_path, output_dir, interval=30):
    # Tạo thư mục đầu ra dựa trên tên video nếu chưa tồn tại
    video_name = Path(video_path).stem
    video_output_dir = os.path.join(output_dir, video_name)
    os.makedirs(video_output_dir, exist_ok=True)
    
    # Mở video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return 0
    
    fps = cap.get(cv2.CAP_PROP_FPS)  # Lấy fps thực tế
    frame_count = 0
    saved_frame_count = 1
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % interval == 0:
            output_path = os.path.join(video_output_dir, f"{video_name}_frame{saved_frame_count}.jpg")
            cv2.imwrite(output_path, frame)
            print(f"Saved: {output_path}")
            saved_frame_count += 1
    
    cap.release()
    print(f"Extracted {saved_frame_count - 1} keyframes from {video_path} with FPS: {fps}")
    return saved_frame_count - 1, fps

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 frame.py <video_path_or_dir>")
        return
    
    input_path = sys.argv[1]
    output_dir = "./frames"
    os.makedirs(output_dir, exist_ok=True)
    interval = 30
    
    if os.path.isfile(input_path):
        keyframes_extracted, fps = extract_keyframes(input_path, output_dir, interval)
        total_keyframes = keyframes_extracted
    elif os.path.isdir(input_path):
        video_extensions = (".mp4", ".avi", ".mov", ".mkv")
        video_files = [f for f in os.listdir(input_path) if f.lower().endswith(video_extensions)]
        
        if not video_files:
            print(f"No video files found in {input_path}")
            return
        
        total_keyframes = 0
        for video_file in video_files:
            video_path = os.path.join(input_path, video_file)
            keyframes_extracted, fps = extract_keyframes(video_path, output_dir, interval)
            total_keyframes += keyframes_extracted
    else:
        print(f"Error: {input_path} is not a valid file or directory")
        return
    
    print(f"Total keyframes extracted: {total_keyframes}")

if __name__ == "__main__":
    main()