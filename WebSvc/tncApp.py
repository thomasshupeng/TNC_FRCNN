"""
File Name: tncApp.py
v 1.1

This program provides the RESTful API for TNC project

10/27/2018
Shu Peng
"""

from flask import Flask, jsonify, request, __version__ as fv
import os
import sys
import requests
import datetime
import BU_ModelLoader
import socket

USING_HTTPS = False
DEBUG_MODE = False

PROJECT_NAME = 'BU'

# This is a trick to get IP address for current server/machine
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.connect(("8.8.8.8", 80))
IPAddr = s.getsockname()[0]
s.close()

HOST_NAME = 'tncimg.westus2.azurecontainer.io'

project_name_to_id = {'TNC': '11111111',
                      'BU': '22222222',
                      'YUNNAN': '33333333'}

SERVICE_NAME = 'tncapi'
API_VERSION = 'v1.0'
END_POINT_NAME = 'Prediction'
MODEL_NAME = '14CFRCNNAlexNet'

app = Flask(__name__)
app.config["DEBUG"] = DEBUG_MODE

full_predict_image_url_endpoint = ''
full_predict_image_endpoint = ''
service_start_time = datetime.datetime.now()
model_load_time = service_start_time

# https://southcentralus.api.cognitive.microsoft.com/customvision/v1.1/Prediction/{projectId}/url[?iterationId][&application]
predict_image_url_endpoint = "/" + SERVICE_NAME + \
                             "/" + API_VERSION + \
                             "/" + END_POINT_NAME + \
                             "/" + project_name_to_id[PROJECT_NAME] + \
                             "/url"
# https://southcentralus.api.cognitive.microsoft.com/customvision/v1.1/Prediction/{projectId}/image[?iterationId][&application]
predict_image_endpoint = "/" + SERVICE_NAME + \
                         "/" + API_VERSION + \
                         "/" + END_POINT_NAME + \
                         "/" + project_name_to_id[PROJECT_NAME] + \
                         "/image"

predict_model_endpoint = "/" + SERVICE_NAME + \
                         "/" + API_VERSION + \
                         "/" + END_POINT_NAME + \
                         "/" + project_name_to_id[PROJECT_NAME] + \
                         "/model"

PORT_NUMBER = 8080
if USING_HTTPS:
    PORT_NUMBER = 443

if USING_HTTPS:
    full_predict_image_url_endpoint = 'https://' + HOST_NAME + predict_image_url_endpoint
    full_predict_image_endpoint = 'https://' + HOST_NAME + predict_image_endpoint
else:
    full_predict_image_url_endpoint = 'http://' + HOST_NAME + ":" + str(PORT_NUMBER) + predict_image_url_endpoint
    full_predict_image_endpoint = 'http://' + HOST_NAME + ":" + str(PORT_NUMBER) + predict_image_endpoint

print("Full PredictImageUrl API = ", full_predict_image_url_endpoint)
print("Full PredictImage API = ", full_predict_image_endpoint)

model = BU_ModelLoader.FRCNN_Model(MODEL_NAME)
model.load()


# 1.	PredictImageUrl
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
    language = request.headers.get('Language')
    print("Language = {!s}".format(content_type))

    # TODO: we can use Prediction-Key as model chooser
    prediction_key = request.headers.get('Prediction-Key')
    print("Prediction-Key = {!s}".format(prediction_key))
    print("=== Body ===")
    img_url = request.json
    print("Url = {!s}".format(img_url))

    # Download image file from given Url, if failed to downloading the image simply return error code
    #  TODO: check if the file is really an image file before downloading
    r = requests.get(img_url, allow_redirects=True, verify=False)
    if r.status_code == requests.codes.ok:
        predictions = model.predict(r.content, lang=language)
        res_prediction_img_url = {
            "Id": "string",
            "Project": project_name_to_id[PROJECT_NAME],
            "Iteration": iteration_id,
            "Created": datetime.datetime.now().isoformat(),
            "Predictions": predictions}
        return jsonify(res_prediction_img_url)
    else:
        return jsonify({"Error": r.reason, "Url": img_url, "Code": r.status_code})


# 2.	PredictImage
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
    language = request.headers.get('Language')
    print("Language = {!s}".format(content_type))

    # TODO: we can use Prediction-Key as model chooser
    prediction_key = request.headers.get('Prediction-Key')
    print("Prediction-Key = {!s}".format(prediction_key))
    predictions = model.predict(request.get_data(), lang=language)
    res_prediction_img_url = {
        "Id": "string",
        "Project": project_name_to_id['TNC'],
        "Iteration": iteration_id,
        "Created": datetime.datetime.now().isoformat(),
        "Predictions": predictions}
    return jsonify(res_prediction_img_url)


# 3. root
@app.route('/', methods=['GET', 'POST'])
def get_root():
    current_time = datetime.datetime.now()
    print("==== root ====")
    msg = "<h1>Welcome to TNC wildlife RESTful API</h1>"
    msg = msg + "<p>version 1.30 F-RCNN</p>"
    msg = msg + "<p>Service started from {:d}/{:d}/{:d}</p>".format(service_start_time.month, service_start_time.day,
                                                                    service_start_time.year)
    msg = msg + "<p>Service has been running for {:d} days.</p>".format((current_time - service_start_time).days)
    msg = msg + "<p>Recent model update time: {:d}/{:d}/{:d}</p>".format(model_load_time.month, model_load_time.day,
                                                                         model_load_time.year)
    msg = msg + "<p>Model in use for {:d} days</p>".format((current_time - model_load_time).days)
    msg = msg + "<p>-----------------------------------------</p>"
    msg = msg + "<p>Python " + sys.version + "</p>"
    msg = msg + "<p>CNTK " + model.get_cntk_version() + "</p>"
    msg = msg + "<p>Flask " + fv + "</p>"
    msg = msg + "<p>-----------------------------------------</p>"
    msg = msg + "<p>Host IP address is : " + IPAddr + "</p>"
    msg = msg + "<p>PredictImageUrl</p>"
    msg = msg + "<p>POST to: " + full_predict_image_url_endpoint + "</p>"
    msg = msg + "<p>PredictImage</p>"
    msg = msg + "<p>POST to: " + full_predict_image_endpoint + "</p>"
    msg = msg + "<p>-----------------------------------------</p>"
    msg = msg + "<p>Please report issue to shpeng@microsoft.com</p>"
    return msg


@app.route(predict_model_endpoint, methods=['GET', 'POST'])
def get_model():
    global model_load_time
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

    model_url = request.headers.get('Model-URL')
    print("Model-URL = {!s}".format(model_url))
    model_name = request.headers.get('Model-Name')
    print("Model-Name = {!s}".format(model_name))
    class_map_name = request.headers.get('Class-Map')
    print("Class-Map = {!s}".format(class_map_name))
    en_zh_map_name = request.headers.get('En-Zh-Map')
    print("En-Zh-Map = {!s}".format(en_zh_map_name))

    if model_url[-1] != '/':
        model_url += '/'

    class_map_file_url = model_url + class_map_name
    r = requests.get(class_map_file_url, allow_redirects=True, verify=False)
    if r.status_code == requests.codes.ok:
        with open(model.get_class_map_file(), 'wb') as cf:
            cf.write(r.content)
    else:
        return jsonify({"Error": r.reason, "Url": class_map_file_url, "Code": r.status_code})

    model_file_url = model_url + model_name
    r = requests.get(model_file_url, allow_redirects=True, verify=False)
    if r.status_code == requests.codes.ok:
        with open(model.get_model_file(), 'wb') as mf:
            mf.write(r.content)
    else:
        return jsonify({"Error": r.reason, "Url": class_map_file_url, "Code": r.status_code})

    en_zh_map_file_url = model_url + en_zh_map_name
    r = requests.get(en_zh_map_file_url, allow_redirects=True, verify=False)
    if r.status_code == requests.codes.ok:
        with open(model.get_en_zh_map_file(), 'wb', encoding='utf-8') as ef:
            ef.write(r.content)
    else:
        return jsonify({"Error": r.reason, "Url": class_map_file_url, "Code": r.status_code})

    model.load()
    model_load_time = datetime.datetime.now()

    return jsonify({"Updated": model_name})


@app.errorhandler(404)
def page_not_found(e):
    return "<h1>404</h1><p>The resource could not be found.</p>", 404


if __name__ == '__main__':
    HOST = os.environ.get('SERVER_HOST', '0.0.0.0')

    PORT_NUMBER = int(os.environ.get('SERVER_PORT', PORT_NUMBER))

    if USING_HTTPS:
        app.run(host=HOST, port=PORT_NUMBER, ssl_context='adhoc')
    else:
        app.run(host=HOST, port=PORT_NUMBER)
