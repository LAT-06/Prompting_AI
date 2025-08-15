from pathlib import Path
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import sys
import json
import re
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import numpy as np
from keybert import KeyBERT

# Load với fast processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=True)
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
model.eval()

# Load model embedding
embedder = SentenceTransformer("all-MiniLM-L6-v2")
kw_model = KeyBERT(model=embedder)

BASE_SAVE_DIR = Path(__file__).resolve().parent / "json_saves"

STOPWORDS = {
    "a","an","the","in","on","at","with","of","and","or","to","for","from","by",
    "is","are","be","as","this","that","these","those","it","its","his","her",
    "their","our","your","you","we","i","my","me","he","she","they","them",
    "was","were","been","being","have","has","had","do","does","did","can",
    "could","will","would","should","may","might","not"
}

def extract_keywords(text: str, top_k: int = 5, use_keybert: bool = True):
    if use_keybert:
        keywords = kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 2),
            stop_words=list(STOPWORDS),
            top_n=top_k,
            use_mmr=True,
            diversity=0.7
        )
        return [kw[0] for kw in keywords]
    else:
        words = re.findall(r"[a-zA-Z]+", text.lower())
        filtered = [w for w in words if w not in STOPWORDS]
        seen, out = set(), []
        for w in filtered:
            if w not in seen:
                seen.add(w)
                out.append(w)
            if len(out) >= top_k:
                break
        return out

def caption_image(img_path: Path, fps=30.0, interval=30):
    image = Image.open(img_path).convert("RGB")
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)
    description = processor.decode(out[0], skip_special_tokens=True)
    embedding = embedder.encode(description).tolist()
    
    # Tính thời gian từ số frame trong tên file (giả định format: ..._frameN.jpg)
    match = re.search(r'_frame(\d+)', img_path.stem)
    if match:
        frame_number = int(match.group(1))
        # Tính thời gian dựa trên interval và fps
        total_seconds = (frame_number - 1) * (interval / fps)  # -1 vì frame 1 là frame đầu
        minutes = int(total_seconds // 60)
        seconds = int(total_seconds % 60)
        timestamp = f"{minutes:02d}:{seconds:02d}"
    else:
        timestamp = "00:00"  # Mặc định nếu không xác định được frame number
    
    return {
        "frame": str(img_path),
        "timestamp": timestamp,
        "description": description,
        "keywords": extract_keywords(description),
        "embedding": embedding
    }

def write_json(path: Path, items):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

# Hàm tìm frame dựa trên prompt và trả về kết quả nếu vượt ngưỡng
def search_frame_by_prompt(json_path: Path, prompt: str, confidence_threshold=50):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not data:
        return False

    # Lấy embedding cho prompt và chuyển sang float32
    prompt_embedding = np.array(embedder.encode(prompt), dtype=np.float32)
    # Tạo ma trận embeddings [N, D]
    embeddings = np.array([item["embedding"] for item in data], dtype=np.float32)

    # Chuẩn hóa vector để tính cosine nhanh bằng numpy
    prompt_norm = prompt_embedding / np.linalg.norm(prompt_embedding)
    embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    similarities = np.dot(embeddings_norm, prompt_norm)

    # Tìm các frame có confidence > threshold
    found = False
    for idx, score in enumerate(similarities):
        confidence = float(score) * 100
        if confidence >= confidence_threshold:
            best_match = data[idx]
            timestamp = best_match.get("timestamp", "00:00")
            print(f"\nSearching in {json_path}:")
            print(f"path to image: {best_match['frame']}, timestamp: {timestamp}, confidence: {confidence:.0f}%")
            found = True
    return found

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--search":
        prompt = sys.argv[2] if len(sys.argv) > 2 else input("Enter prompt: ")

        if len(sys.argv) > 3 and Path(sys.argv[3]).exists():
            json_path = Path(sys.argv[3])
            search_frame_by_prompt(json_path, prompt)
        else:
            json_files = list(BASE_SAVE_DIR.rglob("*.json"))
            if not json_files:
                print("No JSON files found in json_saves or its subdirectories")
                return
            
            found_match = False
            for json_file in json_files:
                if search_frame_by_prompt(json_file, prompt):
                    found_match = True
            
            if not found_match:
                print("No matching results found for the prompt.")
    else:
        target = Path(sys.argv[1] if len(sys.argv) > 1 else "~/Downloads/child.jpeg").expanduser()
        custom_name = Path(sys.argv[2]).name if len(sys.argv) > 2 else None

        BASE_SAVE_DIR.mkdir(parents=True, exist_ok=True)

        if target.is_dir():
            exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
            images = sorted(p for p in target.iterdir() if p.is_file() and p.suffix.lower() in exts)
            if not images:
                raise FileNotFoundError(f"No image files found in: {target}")

            folder_name = target.name
            # Lấy fps từ video gốc nếu có (cần đường dẫn video, ví dụ từ frame.py)
            video_path = None
            if len(sys.argv) > 2 and sys.argv[2].endswith((".mp4", ".avi", ".mov", ".mkv")):
                video_path = Path(sys.argv[2]).expanduser()
            fps = 30.0  # Mặc định
            interval = 30  # Mặc định từ frame.py
            if video_path and video_path.exists():
                import cv2
                cap = cv2.VideoCapture(str(video_path))
                fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()

            for img_path in images:
                item = caption_image(img_path, fps, interval)
                json_filename = img_path.stem + ".json"
                out_path = BASE_SAVE_DIR / folder_name / json_filename
                write_json(out_path, [item])
                print(f"Saved caption and embedding to {out_path}")
        else:
            if not target.is_file():
                raise FileNotFoundError(f"Image not found: {target}")
            item = caption_image(target)
            out_filename = custom_name if (custom_name and custom_name.endswith(".json")) else f"{target.stem}.json"
            out_path = BASE_SAVE_DIR / out_filename
            write_json(out_path, [item])
            print(f"Saved caption and embedding to {out_path}")

if __name__ == "__main__":
    main()