import cv2
import time
if __name__ == '__main__':
    
    

    cam = cv2.VideoCapture(0)
    i = 0
    num_frames = 10000
    start = time.time()
        
    count = 0
    for i in range(num_frames):
        ret, frame = cam.read()
    	# Display the resulting frame
        cv2.imshow('frame',frame)
        cv2.waitKey(1)
        print(frame.shape)
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break
        #count = count + 1

    end = time.time()
    seconds = end - start
    fps  = num_frames / seconds
    print("Estimated frames per second : {0}".format(fps))
    cam.release()    
