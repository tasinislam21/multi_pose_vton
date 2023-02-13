from flask import Flask, render_template, request, send_file
from PIL import Image
import io
from os import listdir
from os.path import isfile, join

app = Flask(__name__)
path1 = "data/train/pose1/"
path2 = "data/train/pose2/"
path3 = "data/train/pose3/"
path4 = "data/train/pose4/"
onlyfiles1 = [f for f in listdir(path1) if isfile(join(path1, f))]
onlyfiles2 = [f for f in listdir(path2) if isfile(join(path2, f))]
onlyfiles3 = [f for f in listdir(path3) if isfile(join(path3, f))]
onlyfiles4 = [f for f in listdir(path4) if isfile(join(path4, f))]
index1 = 1
index2 = 1
index3 = 1
index4 = 1

def provide_image1():
    image_path = path1 + onlyfiles1[index1]
    img = Image.open(image_path).convert('RGB')
    return img

def provide_image2():
    image_path = path2 + onlyfiles2[index2]
    img = Image.open(image_path).convert('RGB')
    return img

def provide_image3():
    image_path = path3 + onlyfiles3[index3]
    img = Image.open(image_path).convert('RGB')
    return img

def provide_image4():
    image_path = path4 + onlyfiles4[index4]
    img = Image.open(image_path).convert('RGB')
    return img

@app.route("/")
def pose_website():
    return render_template('home.html', indexes=[str(index1), str(index2), str(index3), str(index4)])

@app.route("/image1")
def get_pose_1():
    image_io = io.BytesIO()
    image = provide_image1()
    image.save(image_io, format='PNG')
    image_io.seek(0)
    return send_file(
        image_io,
        as_attachment=False,
        mimetype='image/png'
    )

@app.route("/image2")
def get_pose_2():
    image_io = io.BytesIO()
    image = provide_image2()
    image.save(image_io, format='PNG')
    image_io.seek(0)
    return send_file(
        image_io,
        as_attachment=False,
        mimetype='image/png'
    )

@app.route("/image3")
def get_pose_3():
    image_io = io.BytesIO()
    image = provide_image3()
    image.save(image_io, format='PNG')
    image_io.seek(0)
    return send_file(
        image_io,
        as_attachment=False,
        mimetype='image/png'
    )

@app.route("/image4")
def get_pose_4():
    image_io = io.BytesIO()
    image = provide_image4()
    image.save(image_io, format='PNG')
    image_io.seek(0)
    return send_file(
        image_io,
        as_attachment=False,
        mimetype='image/png'
    )

@app.route('/set_index1', methods=['POST'])
def set_index1():
    response = app.response_class(status=200)
    jsonData = request.get_json()
    value = jsonData['slide_value']
    global index1
    index1 = int(value)
    return response

@app.route('/set_index2', methods=['POST'])
def set_index2():
    response = app.response_class(status=200)
    jsonData = request.get_json()
    value = jsonData['slide_value']
    global index2
    index2 = int(value)
    return response

@app.route('/set_index3', methods=['POST'])
def set_index3():
    response = app.response_class(status=200)
    jsonData = request.get_json()
    value = jsonData['slide_value']
    global index3
    index3 = int(value)
    return response

@app.route('/set_index4', methods=['POST'])
def set_index4():
    response = app.response_class(status=200)
    jsonData = request.get_json()
    value = jsonData['slide_value']
    global index4
    index4 = int(value)
    return response

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
    return response

if __name__ == '__main__':
   app.run(debug=True)