import numpy as np
import cv2
import onnx
import onnxruntime
import numpy as np
import matplotlib.pyplot as plt
# importing movie py libraries
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage
WIDTH = 512
HEIGHT = 256
NNX_FILENAME = "LDRN_KITTI_ResNext101_" + str(HEIGHT) + "_" + str(WIDTH) + ".onnx"
ONNX_SIMPLE_FILENAME = "LDRN_KITTI_ResNext101_" + str(HEIGHT) + "_" + str(WIDTH) + "_sim.onnx"



# img_original = cv2.imread("LapDepth-release/example/kitti_demo.jpg")

cam = cv2.VideoCapture('1.mp4')

# frame_width = int(cam.get(3))
# frame_height = int(cam.get(4))
# out = cv2.VideoWriter('out1.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 20, (frame_width,frame_height))


def preprocess(img_original):
  shape = (WIDTH, HEIGHT) 
  mean = np.array([0.485, 0.456, 0.406])
  std = np.array([0.229, 0.224, 0.225])
  img = img_original.copy()
  img = cv2.resize(img, shape)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img = img / 255.
  img = (img - mean) / std
  img = img.astype(np.float32)
  # tensor = cv2.dnn.blobFromImage(img)
  tensor = img.transpose(2, 0, 1).reshape(1, 3, HEIGHT, WIDTH)
  return tensor


''' Load model '''
sess = onnxruntime.InferenceSession('LDRN_KITTI_ResNext101_256_512.onnx')


while(True):
  ret, img_original = cam.read()
  ''' Run inference '''
  input = preprocess(img_original)
  input_name = sess.get_inputs()[0].name
  output = sess.run(None, {input_name: input})
  # print(type(output))


 

  ''' Show result '''
  out = np.array(output[5])
  out = np.squeeze(out, 0)
  out = np.squeeze(out, 0)
  out = out[int(out.shape[0] * 0.18) : , : ]
  out = out * 256.0

  # image_data = np.asarray(out)
  image_data = out

  # array = np.reshape(out, (800, 500))

  # data = im.fromarray(array)
  # image_data = np.asarray(data)
  cv2.imshow("output", image_data)
  cv2.waitKey(1)
  # cv2.imshow("output", oo)
  # cv2.waitKey(1)
  plt.imshow(out)
  plt.show()


      
  # # show the shape of the array
  # print(array.shape)

      
  # creating image object of
  # above array
 
  # print(type(image_data))
  # print(image_data.shape)

  # out.write(out)
  # creating animation
  # animation = VideoClip(out, duration = 10)
 
  # # # displaying animation with auto play and looping
  # animation.ipython_display(fps = 20, loop = True, autoplay = True)
  # print(out)


  # cv2.imshow("output", rgb)
  # cv2.waitKey(1)



  if 0xFF ==ord('q'):
    break











