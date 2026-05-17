import os
import json
import cv2
import glob
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# 경로 설정
BASE_DIR = r"C:\Users\leegy\Desktop\CNN_Quant\AIhub\수어 영상\1.Training"
OUTPUT_DIR = r"C:\Users\leegy\Desktop\CNN_Quant\dataset\train"

def build_metadata_and_tasks(base_dir):
    print("JSON 파일들을 분석하여 메타데이터 맵을 생성합니다.")
    json_files = glob.glob(os.path.join(base_dir, "**", "*.json"), recursive=True)
    
    metadata_map = {}
    for j_path in json_files:
        try:
            with open(j_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # JSON 내부 metaData에서 비디오 파일명 식별
            video_name = data.get("metaData", {}).get("name", "")
            if not video_name: continue
                
            video_stem = Path(video_name).stem
            
            # 수어 동작의 시작/끝 시간 및 클래스 이름 분석
            if "data" in data and len(data["data"]) > 0:
                sign_data = data["data"][0]
                start_time = sign_data.get("start", 0.0)
                end_time = sign_data.get("end", 0.0)
                
                # "고민", "사과" 등 한글 라벨 추출
                class_name = sign_data.get("attributes", [{}])[0].get("name", "Unknown")
                class_name = class_name.strip().replace(" ", "_").replace("/", "_")
                
                if class_name != "Unknown":
                    metadata_map[video_stem] = {
                        "class_name": class_name,
                        "start_time": start_time,
                        "end_time": end_time
                    }
        except Exception:
            continue
            
    print(f"총 {len(metadata_map)}개의 유효한 JSON 메타데이터 파싱이 완료되었습니다.")
    
    print("비디오 파일을 검색 중입니다.")
    video_files = glob.glob(os.path.join(base_dir, "**", "*.mp4"), recursive=True)
    print(f"총 {len(video_files)}개의 비디오 파일을 찾았습니다.")
    
    # JSON 메타데이터와 매칭되는 비디오 파일만 작업 목록에 추가
    tasks = []
    for v_path in video_files:
        v_stem = Path(v_path).stem
        if v_stem in metadata_map:
            tasks.append((v_path, metadata_map[v_stem], OUTPUT_DIR))
            
    return tasks

def process_single_video(args):
    video_path, info, output_base_dir = args
    class_name = info["class_name"]
    start_time = info["start_time"]
    end_time = info["end_time"]
    
    video_stem = Path(video_path).stem
    class_dir = os.path.join(output_base_dir, class_name)
    os.makedirs(class_dir, exist_ok=True)
    
    #  하나의 영상에서 3개의 프레임 경로 생성 (25%, 50%, 75% 지점)
    out_paths = [
        os.path.join(class_dir, f"{video_stem}_1.jpg"), # 25% 지점
        os.path.join(class_dir, f"{video_stem}_2.jpg"), # 50% 지점
        os.path.join(class_dir, f"{video_stem}_3.jpg")  # 75% 지점
    ]
    
    # 이어하기 (Resume)
    # 3개의 이미지가 이미 모두 존재하면 비디오를 열지도 않고 바로 패스
    if all(os.path.exists(p) for p in out_paths):
        return True
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if fps == 0 or total_frames == 0:
        cap.release()
        return False
        
    # 동작 구간 프레임 계산
    if start_time == 0.0 and end_time == 0.0:
        start_f = 0
        end_f = total_frames
    else:
        start_f = int(start_time * fps)
        end_f = int(end_time * fps)
        
    start_f = max(0, start_f)
    end_f = min(total_frames, end_f)
    duration_f = end_f - start_f
    
    # 25%, 50%, 75% 프레임 번호 계산
    if duration_f <= 0:
        target_frames = [total_frames // 4, total_frames // 2, (total_frames * 3) // 4]
    else:
        target_frames = [
            start_f + int(duration_f * 0.25),
            start_f + int(duration_f * 0.50),
            start_f + int(duration_f * 0.75)
        ]
        
    success_count = 0
    # 3개의 프레임을 순회하며 추출 및 저장
    for i, target_frame in enumerate(target_frames):
        target_frame = min(max(target_frame, 0), total_frames - 1)
        
        # 만약 3개 중 일부만 저장되어 있다면, 저장된 건 건너뜀
        if os.path.exists(out_paths[i]):
            success_count += 1
            continue
            
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()
        
        if ret:
            # Numpy를 활용한 한글 경로 저장 (버그 해결)
            result, encoded_img = cv2.imencode('.jpg', frame)
            if result:
                with open(out_paths[i], mode='w+b') as f:
                    encoded_img.tofile(f)
                success_count += 1
                
    cap.release()
    # 3개의 이미지가 모두 성공적으로 저장되었으면 True 반환
    return success_count == 3

if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    tasks = build_metadata_and_tasks(BASE_DIR)
    print(f"총 {len(tasks)}개의 비디오에서 각각 3장의 프레임을 추출")
    
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(tqdm(executor.map(process_single_video, tasks), total=len(tasks)))
        
    success_videos = sum(1 for r in results if r)
    print(f"\n전처리 작업이 종료되었습니다. {success_videos}개의 비디오에서 이미지가 성공적으로 추출되었습니다.")
    print(f"총 추출된 이미지 수: 약 {success_videos * 3}장 예상")