import sys 
sys.path.append(r"C:\Users\91945\Downloads\Analysis\DL.ai\PIS\Starter Bundle")

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import img_to_array,load_img
from keras.utils import np_utils
from CONV import LeNet
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from sklearn.utils import shuffle

#ap=argparse.ArgumentParser()
#ap.add_argument("-i","--input",help="path to the input face images",required=True)
#ap.add_argument("-o","--output",help="path to output model",required=True)
#args=vars(ap.parse_args())

data=[]
labels=[]

for file in os.listdir(r"C:\Users\91945\Downloads\Analysis\DL.ai\PIS\Starter Bundle\Datasets\SMILEsmileD\SMILEs\negatives\negatives7"):
    img=cv2.imread(os.path.join(r"C:\Users\91945\Downloads\Analysis\DL.ai\PIS\Starter Bundle\Datasets\SMILEsmileD\SMILEs\negatives\negatives7",file))
    img=cv2.resize(img,(28,28))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img=img_to_array(img)
    data.append(img)
    label="not smiling"
    labels.append(label)
    
for file in os.listdir(r"C:\Users\91945\Downloads\Analysis\DL.ai\PIS\Starter Bundle\Datasets\SMILEsmileD\SMILEs\positives\positives7"):
    img=cv2.imread(os.path.join(r"C:\Users\91945\Downloads\Analysis\DL.ai\PIS\Starter Bundle\Datasets\SMILEsmileD\SMILEs\positives\positives7",file))
    img=cv2.resize(img,(28,28))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img=img_to_array(img)
    data.append(img)
    label="smiling"
    labels.append(label)
    
data=np.array(data,dtype="float")/255.0
labels=np.array(labels) 
idx = np.random.permutation(len(data))
x,y = data[idx], labels[idx]

le = LabelEncoder().fit(y)
y = np_utils.to_categorical(le.transform(y), 2)

classTotals=y.sum(axis=0)
classWeights=classTotals.max()/classTotals

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,stratify=y,random_state=42)

lenet=LeNet.LeNet(28,28,1,2)
model=lenet.build()

model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])
H=model.fit(x_train,y_train,validation_data=(x_test,y_test),class_weight=classWeights,batch_size=64,epochs=15,verbose=1)

pred=model.predict(x_test)
print(classification_report(y_test.argmax(axis=1),pred.argmax(axis=1),target_names=le.classes_))
model.save("model.hdf5")

plt.style.use("ggplot")
hist=H.history
plt.plot(hist['accuracy'],label='accuracy')
plt.plot(hist['val_accuracy'],label='val_accuracy')
plt.plot(hist['loss'],label='loss')
plt.plot(hist['val_loss'],label='val_loss')
plt.title("GRAPHS")
plt.xlabel("epochs")
plt.ylabel("acc/loss")
plt.legend()
plt.savefig("graph.png")
plt.show()
















