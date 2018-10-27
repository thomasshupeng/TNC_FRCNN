"""
File Name: TNC_RestSvc.py
v 1.0

This program provides the RESTful API for TNC project

6/30/2018
Shu Peng
"""
from flask import Flask, jsonify, request
import os
import requests
import datetime
import shutil
import tempfile
import BU_ModelLoader

'''
1.	PredictImageUrl
https://southcentralus.dev.cognitive.microsoft.com/docs/services/57982f59b5964e36841e22dfbfe78fc1/operations/5a3044f608fa5e06b890f163
                
This one is predicting image by a given a url, which is very similar to what you have with some changes:
a.	Instead pass pic URL as url parameter, it passes URL in http body, which doesn’t need any encoding.  
b.	We can make projectid, iterationid, applicationid all as optional, those id can be useful for different 
projects(云南老君山，北大生命科学院， etc.)

2.	PredictImage
https://southcentralus.dev.cognitive.microsoft.com/docs/services/57982f59b5964e36841e22dfbfe78fc1/operations/5a3044f608fa5e06b890f164

'''

project_name_to_id = {'TNC': '11111111',
                      'BU': '22222222',
                      'YUNNAN': '33333333'}

SERVICE_NAME = 'tncapi'
API_VERSION = 'v1.0'
END_POINT_NAME = 'Prediction'
MODEL_NAME = '21CResNet18'

# Clean up temp image folder
temp_folder = os.path.join(os.getcwd(), 'temp')
if os.path.exists(temp_folder):
    shutil.rmtree(temp_folder)
    os.makedirs(temp_folder)

model = BU_ModelLoader.FRCNN_Model('14CFRCNNAlexNet')
model.load()

app = Flask(__name__)
app.config["DEBUG"] = True

# 1.	PredictImageUrl
# https://southcentralus.api.cognitive.microsoft.com/customvision/v1.1/Prediction/{projectId}/url[?iterationId][&application]
predict_image_url_endpoint = "/" + SERVICE_NAME + \
                             "/" + API_VERSION + \
                             "/" + END_POINT_NAME + \
                             "/" + project_name_to_id['BU'] + \
                             "/url"
print("PredictImageUrl API = ", predict_image_url_endpoint)


@app.route(predict_image_url_endpoint, methods=['POST'])
def post_prediction_img_url():
    # Print received request for debugging purpose
    # TODO: create a logger for basic information

    print("=== Arguments ===")
    iteration_id = request.args.get('iterationId')
    print("iteration_id = {!s}".format(iteration_id))
    application = request.args.get('application')
    print("application = {!s}".format(application))
    print("=== Headers ===")
    content_type = request.headers.get('Content-Type')
    print("Content-Type = {!s}".format(content_type))

    # TODO: we can use Prediction-Key as model chooser
    prediction_key = request.headers.get('Prediction-Key')
    print("Prediction-Key = {!s}".format(prediction_key))
    print("=== Body ===")
    img_url = request.json
    print("Url = {!s}".format(img_url))

    # Download image file from given Url, if failed to downloading the image simply return error code    # TODO: check if the file is really an image file before downloading
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)
    filename = img_url[img_url.rfind('/')+1:]
    img_path_file = os.path.join(temp_folder, filename)

    r = requests.get(img_url, allow_redirects=True, verify=False)
    if r.status_code == requests.codes.ok:
        with open(img_path_file, 'wb') as f:
            f.write(r.content)
    else:
        return jsonify({"Url": img_url, "Code": r.status_code, "Error": r.reason})

    if not os.path.exists(img_path_file):
        res_error = {"Url": img_url, "Error": "Can not open image file."}
        return jsonify(res_error)

    predictions = model.predict(img_path_file)
    os.remove(img_path_file)

    res_prediction_img_url = {
        "Id": "string",
        "Project": project_name_to_id['BU'],
        "Iteration": iteration_id,
        "Created": datetime.datetime.now().isoformat(),
        "Predictions": predictions}
    return jsonify(res_prediction_img_url)


# 2.	PredictImage
# https://southcentralus.api.cognitive.microsoft.com/customvision/v1.1/Prediction/{projectId}/image[?iterationId][&application]
predict_image_endpoint = "/" + SERVICE_NAME + \
                             "/" + API_VERSION + \
                             "/" + END_POINT_NAME + \
                             "/" + project_name_to_id['BU'] + \
                             "/image"
print("PredictImage API = ", predict_image_endpoint)


@app.route(predict_image_endpoint, methods=['POST'])
def post_prediction_image():
    # Print received request for debugging purpose
    # TODO: create a logger for basic information

    print("=== Arguments ===")
    iteration_id = request.args.get('iterationId')
    print("iteration_id = {!s}".format(iteration_id))
    application = request.args.get('application')
    print("application = {!s}".format(application))
    print("=== Headers ===")
    content_type = request.headers.get('Content-Type')
    print("Content-Type = {!s}".format(content_type))
    content_length = request.headers.get('Content-Length')
    print("Content-Length = {!s}".format(content_length))

    # TODO: we can use Prediction-Key as model chooser
    prediction_key = request.headers.get('Prediction-Key')
    print("Prediction-Key = {!s}".format(prediction_key))
    print("=== Saving content to temp file ===")
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)
    temp_file = tempfile.NamedTemporaryFile(suffix='.JPG', dir=temp_folder, delete=False)
    img_path_file = temp_file.name
    temp_file.write(request.get_data())
    temp_file.close()

    print("Image is saved as {!s}".format(img_path_file))
    predictions = model.predict(img_path_file)
    os.remove(img_path_file)
    print("Image file {!s} is removed.".format(img_path_file))

    res_prediction_img_url = {
        "Id": "string",
        "Project": project_name_to_id['TNC'],
        "Iteration": iteration_id,
        "Created": datetime.datetime.now().isoformat(),
        "Predictions": predictions}

    return jsonify(res_prediction_img_url)


@app.errorhandler(404)
def page_not_found(e):
    return "<h1>404</h1><p>The resource could not be found.</p>", 404


app.run(ssl_context='adhoc')
