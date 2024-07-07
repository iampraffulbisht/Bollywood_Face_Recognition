import os
import cv2 as cv
import numpy as np

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


haarcascade = cv.CascadeClassifier('haar_face.xml')



DIR = r'Train'

features = []
labels = []

def create_train ():
    for person in people:
        path = os.path.join(DIR,person)
        label = people.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path,img)

            img_array = cv.imread(img_path)
            if img_array is None:
                print(f"Warning: Unable to read image {img_path}")
                continue
            
                
            gray = cv.cvtColor(img_array,cv.COLOR_BGR2GRAY)

            face_rect = haarcascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=4)

            for(x,y,w,h) in face_rect:
                faces_roi= gray[y:y+h,x:x+w]
                features.append(faces_roi)
                labels.append(label)

create_train()

print("----------Training done ---------")

# print(f'Length of the features =   {len(features)}')
# print(f'Length of the labels =  {len(labels)}')

features = np.array(features,dtype='object')
labels = np.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer_create()

# Training face recognizer on the features and labels we obtained 

face_recognizer.train(features,labels)

face_recognizer.save('face_trained.yml')
np.save('features.npy',features)
np.save('labels.npy',labels)
