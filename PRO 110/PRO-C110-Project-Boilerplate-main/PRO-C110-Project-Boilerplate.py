# import the opencv library
import cv2
import numpy as np 
import tensorflow as tf 

model = tf.keras.models.load_model("keras_model.h5")

# define a video capture object
vid = cv2.VideoCapture(0)
  
while(True):
      
    # Capture the video frame by frame
    ret, frame = vid.read()

    #1.Resizing the Image 
    img = cv2.resize(frame,(224,224))

    #2.Coverting the Image into Dimension 
    testImage = np.array(img,dtype=np.float32)
    testimage=np.expand_dims(test_image,axis=0)

    #3.Normalizing The Image 
    normalizedImage=test_image/255
    
    #4.Predict Result 
    prediction = model.predict(normalizedImage)
    print("Prediction : " , prediction)

    # Flip the Frame 
    frame = cv2.flip(frame,1)
  
    # Display the resulting frame
    cv2.imshow('frame', frame)
    
    # Quit window with spacebar
    key = cv2.waitKey(1)
    
    if key == 32:
        print("Closing")
        break
  
# After the loop release the cap object
vid.release()

# Destroy all the windows
cv2.destroyAllWindows()