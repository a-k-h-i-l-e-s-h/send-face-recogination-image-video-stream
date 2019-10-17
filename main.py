import numpy
import cv2
import sys
import time
import os
import threading
import json
import time
import base64
import requests
#import asyncio
cropimagePayload = []
fullimagePayload =[]


def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def readframe():
	cap = cv2.VideoCapture()
	cap.open("rtsp://aws:Cam@1234@62.150.34.143:554/Streaming/Channels/101") 
	loadprevstate = False	
	previous_frame_gray = None
	while(True):
		ret, frame = cap.read()
		if(ret):
			#cur_time =  int(round(time.time() * 1000))
			#cv2.imwrite("full-image/frame%d.jpg" % cur_time, frame)
			#print('r-e-a-d')
			if(loadprevstate == True):
				current_frame_gray = frame
				cv2.imwrite("current_frame_gray.jpg", current_frame_gray)				
				#retval, buffer1 = cv2.imencode('.jpg', current_frame_gray)
				#img1 = base64.b64encode(buffer1)
				#retval, buffer2 = cv2.imencode('.jpg', previous_frame_gray)
				#img2 = base64.b64encode(buffer2)
				#print('img2',img2)
				#frame_diff = cv2.absdiff(current_frame_gray,previous_frame_gray)
				#cv2.imshow('frame diff ',frame_diff)
				#frame_diff -= frame_diff.min()
				#diff = numpy.uint8(255.0*frame_diff/float(frame_diff.max()))
				#print('diff',diff)
				img1 = cv2.imread('previous_frame_gray.jpg', 0)
				img2 = cv2.imread('current_frame_gray.jpg', 0)
				#print(img1)
				res = cv2.absdiff(img1, img2)
				res = res.astype(numpy.uint8)
				percentage = (numpy.count_nonzero(res) * 100)/ res.size
				print('r-e-a-d', percentage) #print(percentage)
				if(percentage >= 10):		
					cur_time =  int(round(time.time() * 1000))			
					cv2.imwrite("full-image/%d.jpg" % cur_time, current_frame_gray)
					
				previous_frame_gray = current_frame_gray
				cv2.imwrite("previous_frame_gray.jpg", previous_frame_gray)
			else:
				previous_frame_gray = frame
				cv2.imwrite("previous_frame_gray.jpg", previous_frame_gray)
				loadprevstate = True				
				
		else:
			cap.open("rtsp://aws:Cam@1234@62.150.34.143:554/Streaming/Channels/101")
			time.sleep(2)


def rekoginize_face():
	global fullimagePayload
	frontalfaceXML = os.path.abspath("haarcascade_frontalface_default.xml")
	#frontalface.xml absolute path
	faceCascade = cv2.CascadeClassifier(frontalfaceXML)
	while(True):
		try:        
			if(len(fullimagePayload) > 0):
				image = fullimagePayload.pop(0)
				print('F-----R')
				frame = cv2.imread('full-image/' + image)
				os.remove('full-image' + '/' + image)
				epochtime = image.replace('.jpg','')
				#frame = increase_brightness(frame, value=30)
				#cv2.imwrite('cropped-image/img' + image, frame)
				gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
				faces = faces = faceCascade.detectMultiScale(
					gray,
					scaleFactor=1.1,
					minNeighbors=3,
					minSize=(40, 40),
					flags=cv2.CASCADE_SCALE_IMAGE
				) 			
				print(len(faces))
				#If faces are detected upload image
				if len(faces) > 0:
					upload(frame, faces, epochtime)
			else: 
				time.sleep(2)
		except:
			print('crpt-img')

	
def fetching_rawImage():
    global fullimagePayload
    try:  
        while(True):
            if(len(fullimagePayload) == 0):
              imgfiles = os.listdir('full-image')
              time.sleep(1)
              fullimagePayload = imgfiles
            else: 
              time.sleep(2)
    except:
        print('no-data')
			
def upload(frame, faces, epochtime):
	global cropimagePayload
	# print(frame)
	for (x, y, w, h) in faces:
		#Crop image so just face is in viewable
		cropped = crop(frame, x, y, w, h, 20, 20)
		cv2.imwrite("cropped-image/cropped.jpg", cropped)
		with open("cropped-image/cropped.jpg", "rb") as image_file:
			encoded_string = base64.b64encode(image_file.read())
			payload = '{"croppedFace":"'+str(encoded_string)+'","topic":"Ittihad/image","epochtime":"'+epochtime+'"}'	
			cropimagePayload.append(payload)
    	

def crop(img, x, y, w, h, scale_x, scale_y):
    return img[y - scale_y:y+h + scale_y, x - scale_x:x+w + scale_x]

def posttoLambda():
	global cropimagePayload
	while(True):
		print('crop',len(cropimagePayload))
		if(len(cropimagePayload) > 0):
			payload = cropimagePayload.pop(0)
			url = "https://i8i4qgmjcf.execute-api.eu-west-1.amazonaws.com/dev"
			headers = {
				'x-api-key': "QUOkRZdmvg9qHyH8sQQA3417yqIdPIfa26vn2iec",
				'Accept': "application/json",
				'Content-Type': "application/json",
				'accept-encoding': "gzip, deflate"
				}
			response = requests.request("POST", url, data=payload, headers=headers)
			print('datarespose', response.text)
		else:
			time.sleep(5)

def main():
    p = threading.Thread(target=posttoLambda, args=())
    q1 = threading.Thread(target=rekoginize_face, args=())
    q2 = threading.Thread(target=rekoginize_face, args=())
    r = threading.Thread(target=fetching_rawImage, args=())
    p.start()
    q1.start()
    q2.start()
    r.start()

main()

