# from flask import Flask, render_template

# app = Flask(__name__)

# @app.route('/')
# def index():
#     return render_template('index.html')


# if __name__ == '__main__':
#     app.run(debug=True)





# import jason5 as jason
from flask import Flask,request,Response
# from flask_restful import Api,Resource
import tensorflow as tf
from PIL import Image
# import requests
import numpy as np
from flask_cors import CORS
import joblib


app = Flask(__name__)
# api = Api(app)
CORS(app)



"""
PRE-maligant phases

Cancer evolution is a step-wise non-linear process that may start early in life or later in adulthood,
and includes pre-malignant (indolent) and malignant phases. Early somatic changes may not be detectable
or are found by chance in apparently healthy individuals. The same lesions may be detected in pre-malignant clonal conditions. 
In some patients, these lesions may never become relevant clinically whereas in others, 
they act together with additional pro-oncogenic hits and thereby contribute to the formation of an overt malignancy. 
Although some pre-malignant stages of a malignancy have been characterized, no global system to define and to classify these conditions is available. 
To discuss open issues related to pre-malignant phases of neoplastic disorders, a working conference was organized in Vienna in August 2015. 
The outcomes of this conference are summarized herein and include a basic proposal for a nomenclature and classification of pre-malignant conditions. 
This proposal should assist in the communication among patients, physicians and scientists, which is critical as genome-sequencing will soon be offered 
widely for early cancer-detection.






"""


@app.route('/blood', methods=['GET', 'POST'])
def blood_cancer():
	if request.method == 'POST':
		class_names =["[Malignant] early Pre-B","[Malignant] Pre-B","[Malignant] Pro-B","Benign"]
		model = tf.keras.models.load_model("./Blood_Cancer.h5")
		image = request.files["file"]
		image = np.array(Image.open(image).convert("RGB").resize((256,256)))
		image = image/255
		img_array = tf.expand_dims(image,0)
		predictions = model.predict(img_array)
		print("=================================")
		print(predictions[0])
		print("=================================")
		prediction = np.argmax(predictions[0])
		confidence = round(100*(np.max(predictions[0])),2)
		print(f"Predicted Class is {class_names[prediction]} and its Confidence is {confidence}")
		return {"class":class_names[prediction],"Confidence":confidence}
	else:
		return "Found the response !!"

@app.route('/brain', methods=['GET', 'POST'])
def brain_tumor():
	if request.method == 'POST':
		print(request.form)
		class_names =['Effected', 'Healthy']
		model = tf.keras.models.load_model("./Brain_Model.h5")
		image = request.files["file"]
		image = np.array(Image.open(image).resize((256,256)))
		img_array = tf.expand_dims(image,0)
		print(img_array)
		predictions = model.predict(img_array)
		print("=================================")
		print(predictions[0])
		print("=================================")
		print(class_names[predictions])
		print("=================================")

		confidence = round(100*(np.max(predictions[0])),2)
		return {"class":class_names[predictions],"Confidence":confidence}
	else:
		return "Found the Response !!!"
	

@app.route('/breast',methods=['GET','POST'])
def breast_cancer():
	if request.method == 'POST':
		class_names =['Effected', 'Healthy','Less_Effected']
		model = tf.keras.models.load_model("./Custom_Resnet_1.h5")
		image = request.files["file"]
		image = np.array(Image.open(image).convert("RGB").resize((224,224)))
		img_array = tf.expand_dims(image,0)
		predictions = model.predict(img_array)
		prediction = np.argmax(predictions[0])
		confidence = round(100*(np.max(predictions[0])),2)
		return {"class":class_names[prediction],"confidence":confidence}
	else:
		return "Found the Response !!!"	

	
@app.route('/fatal',methods=['GET','POST'])
def fatal_disease():
	return "Under Construction !!!!!!!!"
	# if request.method == 'POST':
	# 	model = joblib.load('trained_model_1.joblib')
	# 	data = np.array(request.get_json())
	# 	reshaped_data = data.reshape(1, -1)  # Reshape to match the expected format
	# 	pred = model.predict(reshaped_data)
	# 	return {"Prediction":pred[0][0]}
	# else:
	# 	return "Found the Response !!!!"

@app.route('/mental',methods=['GET','POST'])
def mental_disease():
	if request.method == 'POST':
		model = joblib.load('trained_model_1.joblib')
		data = np.array(request.get_json()) # accepting data as json
		reshaped_data = data.reshape(1, -1)  # Reshape to match the expected format
		pred = model.predict(reshaped_data)
		return {"Prediction":pred[0][0]}
	else:
		return "Found the Response !!!!"
		
		
	

if __name__ == '__main__':
   app.run(debug=True)