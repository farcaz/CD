# USAGE
# python train.py --dataset dataset --model fashion.model --labelbin mlb.pickle

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import InceptionV3
#from keras.applications import inception_v4
from tensorflow.keras.applications import Xception # TensorFlow ONLY
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications import MobileNet
# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from MyMiniVGG.smallervggnet import SmallerVGGNet
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
import argparse
import random
import pickle
import cv2
import os
import csv
from itertools import cycle
from keras_model import build_model
from tensorflow.keras.backend import clear_session
#from keras.backend import clear_session
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import Callback

from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, matthews_corrcoef, auc
from sklearn.metrics import balanced_accuracy_score, accuracy_score, average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import hamming_loss
from scipy import interp
from sklearn.metrics import classification_report
from sklearn.metrics import average_precision_score
import matplotlib.gridspec as gridspec
#import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()



def F1_score(y_true, y_pred):
    return f1_score(y_true, y_pred, average='samples')

class ComputeF1(Callback):
    
    def __init__(self):
    	self.best_f1 = -1
    	super(ComputeF1, self).__init__()
    	self.targets = []  # collect y_true batches
    	self.outputs = []  # collect y_pred batches

    	# the shape of these 2 variables will change according to batch shape
    	# to handle the "last batch", specify `validate_shape=False`
    	self.var_y_true = tf.Variable(0., validate_shape=False)
    	self.var_y_pred = tf.Variable(0., validate_shape=False)

    def on_batch_end(self, batch, logs=None):
        # evaluate the variables and save them into lists
        self.targets.append(K.eval(self.var_y_true))
        self.outputs.append(K.eval(self.var_y_pred))
'''
    def __init__(self):
        self.best_f1 = -1
        
    def on_epoch_end(self, epoch, logs={}):
        val_pred = np.round(self.model.predict(self.validation_data[0]))
        val_f1 = f1_score(self.validation_data[1], val_pred, average='samples')
        print('Validation Average F1 Score: ', val_f1)
        
        if val_f1 > self.best_f1:
            print('Better F1 Score, Saving model...')
            self.model.save('model.h5')
            self.best_f1 = val_f1
'''

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset (i.e., directory of images)")
#NewCode
#ap.add_argument("-mi", "--inputModel", type=str, default="vgg16",	help="name of pre-trained network to use")
ap.add_argument("--model", choices = ['ResNet50', 'Xception', 'DenseNet121', 'MobileNet','InceptionV3','SmallerVGGNet','VGG16','VGG19'], 
                        help = 'Select a model to train', required = True)
#NewCode end
ap.add_argument("-om", "--outModel",type=str, required=True,
	help="path to output model")
ap.add_argument("-l", "--labelbin", required=True,
	help="path to output label binarizer")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output accuracy/loss plot")

args = vars(ap.parse_args())

# define a dictionary that maps model names to their classes
# inside Keras
MODELS = {
	"VGG16": VGG16,
	"VGG19": VGG19,
	"inception": InceptionV3,
	"xception": Xception, # TensorFlow ONLY
	"resnet": ResNet50,
	"MiniVGG": SmallerVGGNet,
	"DenseNet121": DenseNet121,
	"MobileNet": MobileNet
}

# initialize the number of epochs to train for, initial learning rate,
# batch size, and image dimensions
EPOCHS = 2
INIT_LR = 1e-3
BS = 20
IMAGE_DIMS = (96, 96,3)
thresh =0.050
#Functions for Result Reportings
def plot_loss(N, v_loss, t_loss):
	fig,ax = plt.subplots()
	ax.set_xlabel('EPOCHS') ; ax.set_ylabel('Crossentropy Loss')
	ax.plot(np.arange(0,N), v_loss, 'b', label="Validation Loss")
	ax.plot(np.arange(0,N), t_loss, 'r', label="Train Loss")
	ax.set_title('Loss plot')
	plt.legend()
	plt.grid()
    # save plot in path
	plt.savefig("result/Xception/Loss_plot.png")

def plot_acc(N,v_acc,t_acc):
	fig,ax = plt.subplots()
	ax.set_xlabel('epoch') ; ax.set_ylabel('Accuracy')
	ax.plot(np.arange(0,N), v_acc,'b', label="Validation Accuracy")
	ax.plot(np.arange(0,N), t_acc, 'r', label="Train Accuracy")
	ax.set_title('Accuracy plot')
	plt.legend()
	plt.grid()
  # save plot in path
	plt.savefig("result/Xception/Accuracy_plot.png")

def get_y_pred(y_prob, thresh):
	return (y_prob>thresh)

# Compute ROC values fpr,tpr,threshold,fpr and tpr (micro, and macro) 
def Compute_Roc(n_classes,y_test,y_score):
	fpr = dict()
	tpr = dict()
	roc_auc = dict()
	for i in range(n_classes):
		print(i)
		#print(y_test[:,i],y_score[:,i])
		fpr[i], tpr[i], thresholds = roc_curve(y_test[ :,i], y_score[:,i])
		roc_auc[i] = auc(fpr[i], tpr[i])
	fpr["micro"], tpr["micro"], thresholdmacro= roc_curve(y_test.ravel(), y_score.ravel())
	roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
	all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
	mean_tpr = np.zeros_like(all_fpr)

	for i in range(n_classes):
		mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
	mean_tpr /= n_classes
	fpr["macro"] = all_fpr
	tpr["macro"] = mean_tpr
	roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
	return fpr,tpr,roc_auc,thresholds,thresholdmacro

#compute precision recall ,threshold, precision, and recall (micro)
def Compute_Precision_Recall(n_classes,Y_test,y_score):
	precision = dict()
	recall = dict()
	average_precision = dict()
	for i in range(n_classes):
		precision[i], recall[i],thresholds = precision_recall_curve(Y_test[ :,i],y_score[ :,i])
		average_precision[i] = average_precision_score(Y_test[:, i], y_score[:, i])
    	# A "micro-average": quantifying score on all classes jointly
	precision["micro"], recall["micro"],threshold_micro= precision_recall_curve(Y_test.ravel(),y_score.ravel())
	average_precision["micro"] = average_precision_score(Y_test, y_score,average="micro")
   
	#Added to claculate macro PR
	all_precision = np.unique(np.concatenate([precision[i] for i in range(n_classes)]))
	mean_recall = np.zeros_like(all_precision)
	
	for i in range(n_classes):
		mean_recall += interp(all_precision, precision[i], recall[i])
    	# Finally average it and compute AUC
	mean_recall /= n_classes
	precision["macro"] = all_precision
	recall["macro"] = mean_recall
	average_precision["macro"] = auc(precision["macro"], recall["macro"])
  	
	return precision,recall,average_precision,thresholds,threshold_micro

#plot Single Precision-Recall and Roc Curve in One graph for each model
def plot_single_rpr(fpr,tpr,n_classes,recall,precision):
	fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18,5))
  	#ROC Curve 
	ax[0].plot(fpr["micro"], tpr["micro"],color='deeppink', linestyle=':', linewidth=4,label="micro-average curve.")
	ax[0].plot(fpr["macro"], tpr["macro"],label='macro-average curve.',color='navy', linestyle=':', linewidth=4)
	ax[0].set_xlabel('False Positive Rate')
	ax[0].set_ylabel('True Positive Rate')
	ax[0].set_title('Roc curve')
	h,l=ax[0].get_legend_handles_labels()
  
	# precision recall curve
	ax[1].plot(recall["micro"], precision["micro"], color='deeppink', linewidth=4,linestyle=':')
	ax[1].plot(recall["macro"], precision["macro"], color='navy', linewidth=4,linestyle=':')
	ax[1].plot([1, 0], [0, 1], 'k--', lw=2)
	ax[1].set_xlabel('Recall')
	ax[1].set_ylabel('Precision')
	ax[1].set_title('Precision-Recall')
	plt.subplots_adjust(right=0.73)
	fig.legend(h,l,loc="center right",borderaxespad=0.1,title="Legend Title", ncol=2)
	fig.savefig("result/Xception/rpr_curve.png",format='png',dpi=500)

#Plot ROC for model
def plot_roc(fpr,tpr):
	fig,ax = plt.subplots()
	ax.plot(fpr["micro"], tpr["micro"],color='deeppink', linestyle=':', linewidth=4,label="micro-average curve.")
	ax.plot(fpr["macro"], tpr["macro"],label='macro-average curve.',color='navy', linestyle=':', linewidth=4)
	ax.set_xlabel('False Positive Rate')
	ax.set_ylabel('True Positive Rate')
	ax.set_title('Roc curve')
	plt.legend()
	plt.grid()
	plt.savefig("result/Xception/ROC-AUC.png")

def plot_roc_multiLable(fpr,tpr,n_classes,roc_auc):
	# First aggregate all false positive rates
	all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

	# Then interpolate all ROC curves at this points
	mean_tpr = np.zeros_like(all_fpr)
	for i in range(n_classes):
	    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

	# Finally average it and compute AUC
	mean_tpr /= n_classes
	lw=2
	#roc_auc = dict()

	fpr["macro"] = all_fpr
	tpr["macro"] = mean_tpr
	roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
	roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


	# Plot all ROC curves
	plt.figure()
	plt.plot(fpr["micro"], tpr["micro"],
	         label='micro-average ROC curve (area = {0:0.2f})'
	               ''.format(roc_auc["micro"]),
	         color='deeppink', linestyle=':', linewidth=4)

	plt.plot(fpr["macro"], tpr["macro"],
	         label='macro-average ROC curve (area = {0:0.2f})'
	               ''.format(roc_auc["macro"]),
	         color='navy', linestyle=':', linewidth=4)
	ROCs=0
	for i in range(n_classes):
		ROCs+= roc_auc[i]
	plt.plot(label='Average ROC of labels = {0})'
	             ''.format(ROCs/n_classes))
	
	colors = cycle(['aqua', 'darkorange', 'cornflowerblue','b','g','r','c','m','y','k','w'])
	for i, color in zip(range(n_classes), colors):
	    plt.plot(fpr[i], tpr[i], color=color, lw=lw)
	             


	plt.plot([0, 1], [0, 1], 'k--', lw=lw)
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC for multi-lable classification of medical Image Concepts')
	plt.legend(loc="lower right")
	plt.grid()
	plt.savefig("result/Xception/ROC-AUC-MLabel.png")
#Plot Precision Recall Curve (PRC) for each model
def plot_prc(precision,recall):
	fig,ax = plt.subplots()
	ax.plot(recall["micro"], precision["micro"], color='deeppink', linewidth=4,linestyle=':')
	ax.plot(recall["macro"], precision["macro"], color='navy', linewidth=4,linestyle=':')
	ax.plot([1, 0], [0, 1], 'k--', lw=2)
	ax.set_xlabel('Recall')
	ax.set_ylabel('Precision')
	ax.set_title('Precision-Recall curve')
	plt.legend()
	plt.grid()
	plt.savefig("result/Xception/PRC-AUC.png")

clear_session()
# grab the image paths and randomly shuffle them
print("[INFO] loading images...")
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)

# initialize the data and labels
data = []
labels = []

# loop over the input images
for imagePath in imagePaths:
	# load the image, pre-process it, and store it in the data list
	image = cv2.imread(imagePath)
	#print(imagePath)
	image = cv2.resize(image, (IMAGE_DIMS[0], IMAGE_DIMS[1]))
	image = img_to_array(image)
	data.append(image)

	# extract set of class labels from the image path and update the
	# labels list
	l = label = imagePath.split(os.path.sep)[-2].split("_")
	labels.append(l)

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="int") / 255.0
labels = np.array(labels)
print("[INFO] data matrix: {} images ({:.2f}MB)".format(
	len(imagePaths), data.nbytes / (1024 * 1000.0)))

# binarize the labels using scikit-learn's special multi-label
# binarizer implementation
print("[INFO] class labels:")
mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(labels)

# loop over each of the possible class labels and show them
for (i, label) in enumerate(mlb.classes_):
	print("{}. {}".format(i + 1, label))

# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.25, random_state=42)
#print(testX,testY)

if args["model"]=="SmallerVGGNet":
	print("Enterting MyMiniVGG Block:")
	# construct the image generator for data augmentation
	#aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1, 	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2, 	horizontal_flip=True, fill_mode="nearest")
	# construct the image generator for data augmentation
	aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,horizontal_flip=True, fill_mode="nearest")
	# initialize the model using a sigmoid activation as the final layer
	# in the network so we can perform multi-label classification
	print("[INFO] compiling model...")
	model = SmallerVGGNet.build(width=IMAGE_DIMS[1], height=IMAGE_DIMS[0], depth=IMAGE_DIMS[2], classes=len(mlb.classes_), 	finalAct="sigmoid")
	
	
	# initialize the optimizer (SGD is sufficient)
	opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

	# compile the model using binary cross-entropy rather than
	# categorical cross-entropy -- this may seem counterintuitive for
	# multi-label classification, but keep in mind that the goal here
	# is to treat each output label as an independent Bernoulli
	# distribution

	model.compile(loss="binary_crossentropy", optimizer=opt,	metrics=["accuracy"])
else:
	model = build_model('train', model_name = args["model"])

	

f1_score_callback = ComputeF1()
# train the network
print("[INFO] training network...")
H = model.fit(trainX, trainY, batch_size=BS,
	validation_data=(testX, testY),callbacks = [f1_score_callback],
	epochs=EPOCHS, verbose=2)

# save the model to disk
print("[INFO] serializing network...")
model.save(args["outModel"])

## Compute test F1 Score
#model = load_model(args["model"])

score = F1_score(testY, model.predict(testX).round())
print('F1 Score =', score)

# save the multi-label binarizer to disk
print("[INFO] serializing label binarizer...")
f = open(args["labelbin"], "wb")
f.write(pickle.dumps(mlb))
f.close()

# plot the training loss and accuracy
plt.style.use("bmh")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_accuracy")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_accuracy")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.savefig(args["plot"])

#Result Reporting
N = EPOCHS
v_loss = H.history["val_loss"]
t_loss = H.history["loss"]
plot_loss(N, v_loss, t_loss)
t_acc = H.history["accuracy"]
v_acc = H.history["val_accuracy"]
plot_acc(N, v_acc, t_acc)

prediction=model.predict(testX)
np.savetxt("result/Xception/prediction.csv", prediction, delimiter=",")
	#os.path.join("result/Xception/prediction.csv")
	#f=open(("result/Xception/prediction.csv"),"w")
	#f.write(str(prediction))
	#f.close()

print ("Prediction is :", prediction)
print ("Prediction shape :", prediction.shape)
print ("Prediction type:", type(prediction))
print ("%%%%%%%%%%%%%%%%%")
print ("Test_y:", testY)
print ("Test shape:", testY.shape)
print ("Test-y type:", type(testY))
print ("Labels :", labels)
	
predict = get_y_pred(prediction,thresh)
print ('prediction after comparing with threshold for first:',predict[0])
predict = predict.astype(int)
print ('prediction for first:',predict[0])
np.savetxt("result/Xception/prediction-threshold.csv", predict, delimiter=",")
	#os.path.join("result/Xception/prediction-thresh.csv")
	#f=open(("result/Xception/prediction-thresh.csv"),"w")
	#f.write(str(predict))
	#f.close()
	#print("Chacha Nehru",predict)
	#predicted_tags=[]
	#for i in range(len(predict)):
	#	predicted_tags.append(mlb.inverse_transform(predict)
	#print ('prediction after inverse mapping for first:',predicted_tags[0])	
	#os.path.join("result/Xception/predicted_tags.txt")
	#f=open(("result/Xception/predicted_tags.txt"),"w")
	#f.write(str(predicted_tags))
	#f.close()
	
	# Calculating various metrics
fpr,tpr,roc_auc,r_thresholds,r_micro_threshold=Compute_Roc(4,testY,predict)
precision,recall,average_precision,pr_thresholds,pr_micro_threshold=Compute_Precision_Recall(2,testY,predict)

	#np.savetxt("result/Xception/fpr.csv", fpr, delimiter=",")
	#np.savetxt("result/Xception/tpr.csv", tpr, delimiter=",")
	#np.savetxt("result/Xception/precision.csv", precision, delimiter=",")
	#np.savetxt("result/Xception/recall.csv", recall, delimiter=",")
	#np.savetxt("result/Xception/av-precision.csv", average_precision, delimiter=",")




plot_single_rpr(fpr,tpr,4,recall,precision)
plot_roc(fpr,tpr)
plot_roc_multiLable(fpr,tpr,4,roc_auc)
plot_prc(precision,recall)

# Write train and validation accuracy and loss in CSV file
w=csv.writer(open("result/Xception/history-model.csv",'w'))
for key, val in H.history.items():
	w.writerow([key,val])
            
#Write AUC for each class in text file
os.path.join("result/Xception/roc-auc.txt")        
f=open(("result/Xception/roc-auc.txt"),"w")
f.write(str(roc_auc))
f.close()
print ("ROC is :",roc_auc)
    
#Write PRC-AUC for each class in text file
os.path.join("result/Xception/prc-auc.txt")        
f=open(("result/Xception/prc-auc.txt"),"w")
f.write(str(average_precision))
f.close()  
      
# Calculate Hamming distance       
hl=hamming_loss(testY, predict)
os.path.join("result/Xception/Hamming-distance.txt")
f=open(("result/Xception/Hamming-distance.txt"), "w")
f.write(str(hl))
f.close()
print ('Hamming distance is : ', hl)

plt.style.use("bmh")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_accuracy")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.savefig(args["plot"])

# Calculate Classification Report
c_report=classification_report(testY, predict)
os.path.join("result/Xception/Classification-report.txt")
f=open(("result/Xception/Classification-report.txt"),"w")
f.write(str(c_report))
f.close()

