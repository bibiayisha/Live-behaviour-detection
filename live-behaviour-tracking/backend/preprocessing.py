import cv2
# import numpy 

def preprocessing(img): 
    # print('preprocessing')
    resized_frame = cv2.resize(img, (128, 128))
    normalized_frame = resized_frame / 255
    # normalized_frame = numpy.resize(normalized_frame,(1,64,64,3))
    return normalized_frame
