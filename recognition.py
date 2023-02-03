import face_recognition as fr
import cv2
import os
import time
import pickle

NAME = ''
MODEL = 'cnn'
TOLERANCE = 0.5

def get_name(name):
    global NAME
    NAME = name

def capture():
    count_capture = 1
    if not os.path.exists(NAME):
        os.mkdir(NAME)

    cap = cv2.VideoCapture(0)
    # while True:
    while count_capture <= 10:
        ret, frame = cap.read()
        cv2.imshow(NAME, frame)
        cv2.imwrite(f'{NAME}/{NAME}_{count_capture}.jpg', frame)
        count_capture += 1
    
    cap.release()
    cv2.destroyAllWindows()

def training():
    count_training = 1
    path = f'{NAME}'
    
    faces_known = []
    names_known = []

    classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")    
    for img in os.listdir(path):
        image = cv2.imread(f'{path}/{img}')
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = classifier.detectMultiScale(grayscale, scaleFactor=1.2, minNeighbors=3)

        for (x, y, w, h) in faces:
            crop_face = image[y:y+h, x:x+w]
            cv2.imwrite(f'{NAME}/{NAME}_{count_training}.jpg', crop_face)

            encoding = fr.face_encodings(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))[0]

            names_known.append(path)
            faces_known.append(encoding)

        count_training += 1

    with open(f'{NAME}/{NAME}-val.pkl', 'wb') as f:
        pickle.dump(faces_known, f)
        pickle.dump(names_known, f)

def compare(known_face, known_name, lock_system, interval=0):
    scale = 0.25
    size = 1/scale
    chances_0 = 2
    if interval != 0:
        while True:
            cap_i = cv2.VideoCapture(0)
            ret, frame = cap_i.read()
            resized_img = cv2.resize(frame, (0, 0), None, scale, scale)
            resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)

            locations = fr.face_locations(resized_img, model = MODEL)
            encodings = fr.face_encodings(resized_img, locations)
            if locations == [] or encodings == []:
                chances_0 -= 1
                if chances_0 <= 0:
                    lock_system()
            for face_encoding, face_location in zip(encodings, locations):
                result = fr.compare_faces(known_face, face_encoding, TOLERANCE)
                match = None
                if True in result:
                    match = known_name[result.index(True)]
                    print(f'{result} from {match}')
                else:
                    print('none')
                    lock_system()

            cap_i.release()
            time.sleep(interval)
        
    else:
        t1 = time.time()
        cap_as = cv2.VideoCapture(0)
        t2 = time.time()
        print(t2-t1)
        ret, frame = cap_as.read()
        resized_img = cv2.resize(frame, (0, 0), None, scale, scale)
        # resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)

        locations = fr.face_locations(resized_img, model = MODEL)
        encodings = fr.face_encodings(resized_img, locations)
        print("Reached-2")
        if locations == [] or encodings == []:
            lock_system()
        for face_encoding, face_location in zip(encodings, locations):
            result = fr.compare_faces(known_face, face_encoding, TOLERANCE)
            print("Reached-3")
            match = None
            if True in result:
                match = known_name[result.index(True)]
                print(f'{result} from {match}')
            else:
                print("No")
                lock_system()
        cap_as.release()

if __name__ == "__main__":
    start = time.time()
    capture()
    known_face, known_name = training()
    print('Processing done')
    compare(known_face, known_name)
    print('Comparing done')
    end = time.time()
    print(end - start)
