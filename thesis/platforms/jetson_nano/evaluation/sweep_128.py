# -*- coding: utf-8 -*-
import subprocess
from pathlib import Path
import itertools
import csv
import re

def main():
    target_script = str(Path(__file__).resolve().with_name("batch_evaluate_128.py"))
    csv_filename = "hyperparameter_tuning_results_128_batch.csv"
    
    motion_list = [10, 15, 20]
    conf_list = [0.45, 0.55, 0.65]
    global_list = [0.55, 0.62, 0.70]
    miss_list = [1, 2, 3]
    
    combinations = list(itertools.product(motion_list, conf_list, global_list, miss_list))
    print("[INFO] 準備進行自動化參數搜索 (128x128 批次盲測)，總計 {} 組測試...".format(len(combinations)))

    with open(csv_filename, mode="w") as file:
        writer = csv.writer(file)
        writer.writerow(["Motion_Thresh", "Conf_Local", "Conf_Global", "Max_Miss", "Local_Ratio(%)", "Global_Ratio(%)", "Precision(%)", "Recall(%)", "Average_FPS"])
        
        for idx, (m, c, gc, ms) in enumerate(combinations):
            print("\n[{}/{}] 測試參數 -> Motion: {}, Local: {}, Global: {}, Miss: {}".format(idx+1, len(combinations), m, c, gc, ms))
            
            cmd = ["python3", target_script, "--motion", str(m), "--local_conf", str(c), "--global_conf", str(gc), "--miss", str(ms)]
            
            try:
                result = subprocess.run(
                    cmd, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE, 
                    universal_newlines=True
                )
                
                output = result.stdout
                err_output = result.stderr
                
                if result.returncode != 0:
                    print("       [嚴重錯誤] 程式當機，Return Code: {}".format(result.returncode))
                    print("       ========== 錯誤訊息 (Stderr) ==========")
                    print(err_output.strip())
                    print("       =======================================")
                    break
                    
                local_match = re.search(r"\[.*省電\] 佔比: ([\d\.]+)%", output)
                global_match = re.search(r"\[全域救援\] 佔比: ([\d\.]+)%", output)
                fps_match = re.search(r"平均 FPS: ([\d\.]+)", output)
                precision_match = re.search(r"準確率 \(Precision\): ([\d\.]+)%", output)
                recall_match = re.search(r"召回率 \(Recall\): ([\d\.]+)%", output)
                
                if local_match and global_match and fps_match and precision_match and recall_match:
                    local_ratio = float(local_match.group(1))
                    global_ratio = float(global_match.group(1))
                    fps = float(fps_match.group(1))
                    precision = float(precision_match.group(1))
                    recall = float(recall_match.group(1))
                    
                    writer.writerow([m, c, gc, ms, local_ratio, global_ratio, precision, recall, fps])
                    file.flush()
                    
                    print("       成功擷取 -> 省電: {:.1f}%, Precision: {:.1f}%, Recall: {:.1f}%, FPS: {:.1f}".format(
                        local_ratio, precision, recall, fps))
                else:
                    print("       [警告] 擷取失敗！主程式可能提早結束或遇到防呆檢查。")
                    print("       ========== 主程式真實輸出 (Stdout) ==========")
                    print(output.strip() if output.strip() else "(無任何標準輸出)")
                    if err_output.strip():
                        print("       ========== 主程式錯誤輸出 (Stderr) ==========")
                        print(err_output.strip())
                    print("       =============================================")
                    break
                    
            except Exception as e:
                print("       [錯誤] Runner 執行發生未預期錯誤: {}".format(e))
                break

if __name__ == "__main__":
    main()