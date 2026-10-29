
# Support direct script execution; relative data paths still use the project root.
if __package__ in (None, ""):
    import sys as _thesis_sys
    from pathlib import Path as _ThesisPath
    _thesis_sys.path.insert(0, str(_ThesisPath(__file__).resolve().parents[2]))

import os
import shutil

def reorganize_dataset():
    # 設定路徑
    raw_dir = os.path.join("dataset", "raw")
    part_dir = os.path.join(raw_dir, "part1")

    # 確保 part1 資料夾存在
    os.makedirs(part_dir, exist_ok=True)

    print("Starting directory reorganization...")
    
    # 掃描 raw 底下的所有東西
    moved_count = 0
    for item in os.listdir(raw_dir):
        item_path = os.path.join(raw_dir, item)
        
        # 只搬移名稱開頭是 "match" 且是資料夾的物件
        if os.path.isdir(item_path) and item.startswith("match"):
            target_path = os.path.join(part_dir, item)
            print(f"Moving {item} to part1...")
            
            try:
                shutil.move(item_path, target_path)
                moved_count += 1
            except Exception as e:
                print(f"Failed to move {item}. Error: {e}")

    print(f"Successfully moved {moved_count} match folders into part1.")

if __name__ == "__main__":
    reorganize_dataset()