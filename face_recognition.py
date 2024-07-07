import numpy as np
import cv2 as cv

haarcascade = cv.CascadeClassifier('haar_face.xml')

people = [
    'Irrfan_Khan', 'Jacqueline_Fernandez', 'John_Abraham', 'Juhi_Chawla', 'Kajal_Aggarwal', 
    'Kajol', 'Kangana_Ranaut', 'Kareena_Kapoor', 'Karisma_Kapoor', 'Kartik_Aaryan', 
    'Katrina_Kaif', 'Kiara_Advani', 'Kriti_Kharbanda', 'Kriti_Sanon', 'Kunal_Khemu', 
    'Lara_Dutta', 'Madhuri_Dixit', 'Manoj_Bajpayee', 'Mrunal_Thakur', 'Nana_Patekar', 
    'Nargis_Fakhri', 'Naseeruddin_Shah', 'Nushrat_Bharucha', 'Paresh_Rawal', 
    'Parineeti_Chopra', 'Pooja_Hegde', 'Prabhas', 'Prachi_Desai', 'Preity_Zinta', 
    'Priyanka_Chopra', 'R_Madhavan', 'Rajkummar_Rao', 'Ranbir_Kapoor', 'Aamir_Khan', 
    'Abhay_Deol', 'Abhishek_Bachchan', 'Aftab_Shivdasani', 'Aishwarya_Rai', 'Ajay_Devgn', 
    'Akshay_Kumar', 'Akshaye_Khanna', 'Alia_Bhatt', 'Ameesha_Patel', 'Amitabh_Bachchan', 
    'Amrita_Rao', 'Amy_Jackson', 'Anil_Kapoor', 'Anushka_Sharma', 'Anushka_Shetty', 
    'Arjun_Kapoor', 'Arjun_Rampal', 'Arshad_Warsi', 'Asin', 'Ayushmann_Khurrana', 
    'Bhumi_Pednekar', 'Bipasha_Basu', 'Bobby_Deol', 'Deepika_Padukone', 'Disha_Patani', 
    'Emraan_Hashmi', 'Esha_Gupta', 'Farhan_Akhtar', 'Govinda', 'Hrithik_Roshan', 
    'Huma_Qureshi', 'Ileana_DCruz', 'Randeep_Hooda', 'Rani_Mukerji', 'Ranveer_Singh', 
    'Richa_Chadda', 'Riteish_Deshmukh', 'Saif_Ali_Khan', 'Salman_Khan', 'Sanjay_Dutt', 
    'Sara_Ali_Khan', 'Shah_Rukh_Khan', 'Shahid_Kapoor', 'Shilpa_Shetty', 'Shraddha_Kapoor', 
    'Shreyas_Talpade', 'Shruti_Haasan', 'Sidharth_Malhotra', 'Sonakshi_Sinha', 
    'Sonam_Kapoor', 'Suniel_Shetty', 'Sunny_Deol', 'Sushant_Singh_Rajput', 'Taapsee_Pannu', 
    'Tabu', 'Tamannaah_Bhatia', 'Tiger_Shroff', 'Tusshar_Kapoor', 'Uday_Chopra', 
    'Vaani_Kapoor', 'Varun_Dhawan', 'Vicky_Kaushal', 'Vidya_Balan', 'Vivek_Oberoi', 
    'Yami_Gautam', 'Zareen_Khan'
]
# features = np.load('features.npy', allow_pickle=True)
# labels = np.load('labels.npy')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

img = cv.imread(r'validation/both3.jpeg')

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)



#  First Detect face

face_rect = haarcascade.detectMultiScale(gray,1.1,7,minSize=(50, 50))
for (x,y,w,h) in face_rect:
    faces_roi = gray[y:y+h,x:x+h]

    label,confidence = face_recognizer.predict(faces_roi)
    print(f'Label = {label} with a confidence {confidence}')
    cv.putText(img,str(people[label]),(20,20),cv.FONT_HERSHEY_COMPLEX,0.5,(0,255,0),thickness=1 )
    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),thickness=2)

cv.imshow("Detected Face",img)

cv.waitKey(0)


# import numpy as np
# import cv2 as cv

# haarcascade = cv.CascadeClassifier('haar_face.xml')
# # people = [ 'Shah rukh khan', 'Deepika Padukon', 'Johnny Depp', 'Angelina Jolie', 'Brad Pitt', 'Denzel Washington', 'Hugh Jackman', 'Jennifer Lawrence','Kate Winslet','Leonardo DiCaprio','Megan Fox','Natalie Portman','Nicole Kidman','Robert Downey Jr','Scarlett Johansson','Tom Cruise','Tom Hanks','Will Smith']
# face_recognizer = cv.face.LBPHFaceRecognizer_create()
# face_recognizer.read('face_trained.yml')

# img = cv.imread(r'validation/srkc.jpeg')
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)


# face_rect = haarcascade.detectMultiScale(gray, 1.2, 4, minSize=(30, 30))
# confidence_threshold = 200  # Define your confidence threshold here

# for (x, y, w, h) in face_rect:
#     faces_roi = gray[y:y+h, x:x+w]
#     label, confidence = face_recognizer.predict(faces_roi)
#     if confidence < confidence_threshold:
#         print(f'Label = {label} with a confidence {confidence}')
#         cv.putText(img, f'{people[label]} ({confidence:.2f})', (x, y - 10), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), thickness=2)
#         cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)
#     else:
#         print(f'Face detected with high confidence score {confidence}, not displaying.')

# if face_rect is not None and confidence < confidence_threshold:
#     cv.imshow("Detected Face", img)
#     cv.waitKey(0)
#     cv.destroyAllWindows()
# else:
#     print("No faces with good confidence found.")
