from scipy.spatial import distance #scipy helps in calculation of EAR and MAR
from imutils import face_utils #give the landmarks of face, eye
from pygame import mixer # for music 
import imutils #imutils helps in image resize, rotate, detection edges etc
import dlib #help in frontal face detection by taking the 68 face landmarks
import cv2 #open cv helps in caturing the video




mixer.init()
mixer.music.load("music.wav")

def eye_aspect_ratio(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear
	
def mouth_aspect_ratio(mouth):
	A = distance.euclidean(mouth[1], mouth[7]) # 51, 59
	B = distance.euclidean(mouth[2], mouth[6]) # 53, 57
	C = distance.euclidean(mouth[3], mouth[5]) # 49, 55
	D = distance.euclidean(mouth[0], mouth[4])
	mar = (A + B + C) / (2.0 * D)
	return mar
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["mouth"]


earthresh = 0.25

marthresh = 2

frame_check = 20
detect = dlib.get_frontal_face_detector() #contain pretrained HOG and linear SVM face detector
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") #detects 68 landmarks on face

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
cap=cv2.VideoCapture(0) #capture the video from front camera
flag=0
while True:
	ret, frame=cap.read() #gives two arguments, first=if frame available or not(stored in "ret"), second =image array (in frame) 
	frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #convert frames to grayscale
	subjects = detect(gray, 0) #detect the image from the gray scaled frame by giving index as 0, takes the grayscaled image. detect function takes the grayscle image as input and returns a list of rectangles(subjects) that represent the detected faces.Each rectangle contains the x,y coordinates and height and width of rectangle
	for subject in subjects: # To detect all the landmarks on the face by taking the gray scale image from "subjects"
		shape = predict(gray, subject)
		shape = face_utils.shape_to_np(shape) #convert the facial landmarks (eye border) into Numpy array
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		
		centermouth=shape[mStart:mEnd]

		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)

		mouthmar= mouth_aspect_ratio(centermouth)
		
		ear = (leftEAR + rightEAR) / 2.0
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)

		mouthHull = cv2.convexHull(centermouth)

		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

		cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)

		if ear < earthresh or mouthmar > marthresh :
			flag += 1
			print (flag)
			if flag >= frame_check :
				cv2.putText(frame, "****************ALERT!****************", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				cv2.putText(frame, "****************ALERT!****************", (10,325),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				mixer.music.play()
		else:
			flag = 0

		
		#MAR


	cv2.imshow("Frame", frame) # display image on our window, takes 2 parameter, 1=window name, 2=image to be display (here image is stored in "frame" variable)
	key = cv2.waitKey(1) & 0xFF #allows you to wait for a specific time in millisecond until you press any button on the keyboard
	if key == ord("q"):
		break
cv2.destroyAllWindows()
cap.release() 


#MAR calculation


