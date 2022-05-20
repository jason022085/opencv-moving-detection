import cv2
import glob
import os
import argparse

def get_images_from_video(video_name, sec_interval, saved_folder):
    video_images = []
    cap = cv2.VideoCapture(video_name) # 讀取影片
    width = cap.get(3)
    height = cap.get(4)
    fps = cap.get(5)
    frames = cap.get(7)
    resolution = min([width, height])
    print(f"WIDTH: {width}")
    print(f"HEIGHT: {height}")
    print(f"FPS: {fps}")
    print(f"FRAMES: {frames}")
    
    if cap.isOpened(): #判斷是否開啟影片
        rval, video_frame = cap.read()
    else:
        rval = False
    
    c = 0
    while rval:  #擷取視頻至關閉
        rval, video_frame = cap.read()
        if (c % (sec_interval * fps) == 0) and (0 <= c <= frames): #每隔幾秒進行擷取
            if resolution != 480: # 儲存照片大小
                try:
                    ratio = 480/resolution
                    video_frame = cv2.resize(video_frame, (int(width * ratio), int(height * ratio)), interpolation=cv2.INTER_LINEAR)
                except Exception:
                    print("CAN NOT RESIZE")
            video_images.append(video_frame)
           # print("now frame: ", c)   
        c = c + 1
    cap.release()

    save_name = video_name.split('\\')[1][:27]
    for i in range(0, len(video_images) - 1): #顯示出所有擷取之圖片(最後一張可能儲存會error)
        label = str(i) if i > 9 else "0"+str(i)
        cv2.imwrite(f"./{saved_folder}/{save_name + '_' + label}.jpg", video_images[i]) #儲存圖片
        print(f"\rSaving: {i+1} / {len(video_images) - 1}", end = "")
    print("\n")
    return video_images
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-to_folder", help="where to save the images", type=str, default="saved_folder")
    parser.add_argument("-from_folder", help="where to find the videos", type=str, default="video_folder")
    parser.add_argument("-sec_interval", help="extract 1 images for every sec_interval seconds", type=int, default=5)
    args = parser.parse_args()
    # 檢查儲存目的地是否存在同名稱資料夾
    try:
        os.makedirs(f"./{args.to_folder}/")
        print("make new folder")
    except FileExistsError:
        print("folder had exist")

    # 獲取資料夾內所有影片
    for file_name in glob.glob(f'./{args.from_folder}/*.mp4'):
        print(f"Find a video: {file_name}")
        get_images_from_video(file_name, sec_interval = args.sec_interval, saved_folder=args.to_folder)