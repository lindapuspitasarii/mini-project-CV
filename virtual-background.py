import cv2
import mediapipe as mp
import numpy as np
import time

mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation

bg_color = (0,255,196)
capture = cv2.VideoCapture(0)
# (video.mp4) -> untuk capture tayangan video .mp4
# (0) -> untuk capture kamera 1
# (1) -> untuk capture kamera 2
sebelumnya = 0
# untuk webcam input
with mp_selfie_segmentation.SelfieSegmentation(model_selection=0) as selfie_segmentation:
  bg_image = None
  while capture.isOpened():
    sukses, video = capture.read()
    if not sukses:
      print("Tidak ada tangkapan frame kamera")
      continue
    
    # preprocess masking video    
    video.flags.writeable = False
    results = selfie_segmentation.process(video)
    video.flags.writeable = True
    
    kondisi = np.stack((results.segmentation_mask,) *3, axis = -1) > 0.1
    
    # agar muncul gambar backgroundnya: pilih salah satu
    bg_video = cv2.imread("backgrounds/4.png") # buat backgroundnya
    #bg_video = cv2.GaussianBlur(video, (55,55), 0) # ada efek blur-nya
    
    if bg_video is None:
      bg_video = np.zeros(video.shape, dtype=np.uint8)
      bg_video[:] = bg_color
    output_video = np.where(kondisi, video, bg_video) # return elements chosen from x or y depending on condition
    
    # Frame per Second
    sekarang = time.time()
    fps = 1/(sekarang - sebelumnya)
    sebelumnya = sekarang
    cv2.putText(output_video, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_PLAIN, 3, (0,192,255), 2)
    
    # Header view-nya
    cv2.imshow("Kelompok 2 - Virtual Background", output_video)
    if cv2.waitKey(5) & 0xFF ==27:
      break
  capture.release()
