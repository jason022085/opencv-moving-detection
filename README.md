# opencv-moving-detection
偵測影片中的移動物體，輸出總移動秒數
1. AutoEncoder.py: 基於卷積的自動編碼器架構(權重沒有上傳)，其Encoder用於重建圖片，再根據重建圖片與原始圖片的差異判斷出是否為異常圖片。
2. AutoEncoderInference.py: 處理圖片並且應用AutoEncoder模型。
3. main.py: 輸入影片路徑，經過影像處理後，偵測移動部分的圖片，再經由AutoEncoderInference判斷是否為異常圖片。
4. video_to_image.py: 將影片逐偵輸出圖片並保存。
