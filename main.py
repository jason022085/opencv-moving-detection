import cv2
import glob
import os
import argparse
import numpy as np
from sympy import continued_fraction
import json
import AutoEncoderInference

def save_label(file_name, label, width, height):
    with open(file_name,'a') as f:
        # the format of yolo
        x_rate = (label[0] + label[2]//2)/width
        y_rate = (label[1] + label[3]//2)/height 
        w_rate = label[2]/width
        h_rate = label[3]/height
        f.write(f"0 {x_rate:.3f} {y_rate:.3f} {w_rate:.3f} {h_rate:.3f}\n")

def get_moving_in_video(video_name, saved_folder):
    # 檢查儲存目的地是否存在同名稱資料夾
    try:
        os.makedirs(saved_folder)
        print("make new folder")
    except FileExistsError:
        print("folder had exist")

    cap = cv2.VideoCapture(video_name) # 讀取影片
    width = cap.get(3)
    height = cap.get(4)
    fps = cap.get(5)
    frames = cap.get(7)
    resolution = min([width, height])
    print(f"RESOLUTION: {resolution}")
    print(f"FPS: {fps}")
    print(f"FRAMES: {frames}")

    if resolution == 0.0:
        print("video is broken")
        return -1

    if cap.isOpened(): # 判斷是否開啟影片
        rval, video_frame = cap.read()
        norm = cv2.normalize(video_frame, None, alpha=0,beta=200, norm_type=cv2.NORM_MINMAX)
        avg = cv2.GaussianBlur(norm, (5, 5), 0) # 模糊
        avg_float = np.float32(avg)
    else:
        rval = False

    count_motion = 0
    now_frame = 0
    AE = AutoEncoderInference.AutoencoderInference(cuda = False)
    while rval:  # 擷取視頻至關閉
        rval, video_frame = cap.read()
        if rval:
            norm = cv2.normalize(video_frame, None, alpha=0,beta=200, norm_type=cv2.NORM_MINMAX)
            blur = cv2.GaussianBlur(norm, (5, 5), 0)
            blur = np.float32(blur)
            diff = cv2.absdiff(avg_float, blur) # 相減
            gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            gray = gray.astype("uint8")
            ret, thresh = cv2.threshold(gray, 25, 255, cv2.THRESH_OTSU) # 二值化

            kernel = np.ones((3,3), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2) # 開運算就是先侵蝕後膨脹

            (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            has_motion = False
            for i, c in enumerate(cnts):
                if cv2.contourArea(c) < (height * width) * 0.01: # 面積太小就忽略不計
                    continue
                else:
                    (x, y ,w, h) = cv2.boundingRect(c)
                    rebuildable = AE.inference(video_frame) # 用AutoEncoder重建畫面，重建畫面與原始畫面相似的就是卸料畫面
                    if (now_frame >= frames * 0.50) & (y >= height * 0.40) & (h >= height * 0.1) & (h > w) & (rebuildable == True):
                        # 跳過前0.5影片 & 跳過上面0.4畫面 & 忽略框高小於0.1畫面高 & 框長大於框寬
                        has_motion = True           
                        # 劃出方框
                        cv2.rectangle(video_frame, (x,y), (x+w,y+h), (255,255,0), 2) # 最後兩個參數是顏色和邊寬的意思
                        cv2.imwrite(f"{saved_folder}/{i}_{now_frame}.jpg", video_frame) # 儲存圖片
                        # 儲存label
                        save_label(f"{folder_path}/saved_images/{args.video_name.split('.')[0]}/{i}_{now_frame}.txt", (x,y,w,h), width, height)
            count_motion = count_motion + 1 if has_motion else count_motion # 該偵數窗格有motion就+1
            # 更新平均影像
            cv2.accumulateWeighted(blur, avg_float, 0.1) # second = 0.1*first + (1 - 0.1)*second, 數值越大對速度越敏感
            avg = cv2.convertScaleAbs(avg_float)
            cv2.imshow("my video", video_frame)
            now_frame+=1
            if cv2.waitKey(1) == ord('q'):
                break
    total_motion_time = count_motion/fps if fps != 0 else 0
    if  total_motion_time >= 1: # 總秒數夠多才算有卸料
        print(f"有卸料共{total_motion_time:.2f}秒")
    else:
        print(f"無卸料共{total_motion_time:.2f}秒")
    cap.release()
    cv2.destroyAllWindows()
    return total_motion_time

    
if __name__ == "__main__":
    folder_path = "./where/the/videos/are" 
    video_names = os.listdir(folder_path)
    results = []
    for video_name in video_names:
        if video_name[-3:] == "mp4":
            parser = argparse.ArgumentParser()
            parser.add_argument("-video_name", help="video_name", type=str, default=f"{video_name}")
            args = parser.parse_args()
            print(video_name)
            total_motion_time = get_moving_in_video(f"{folder_path}/{args.video_name}", f"{folder_path}/saved_images/{args.video_name.split('.')[0]}")
            results.append(total_motion_time)
    
    f = open(f"{folder_path}/results.json", "w")
    json.dump({"results":results}, f)
    f.close()