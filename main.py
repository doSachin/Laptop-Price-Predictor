import os
import pickle
import xgboost

import numpy as np
from flask import Flask, render_template, request

# import the model
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

# top models
top_brand = ['Apple', 'Microsoft', 'Dell', 'HP', 'Lenovo']

app = Flask(__name__)

IMG_FOLDER = os.path.join('static', 'img')
app.config['UPLOAD_FOLDER'] = IMG_FOLDER


@app.route('/')
def index():
    images = os.path.join(app.config['UPLOAD_FOLDER'], 'top-5-laptop.jpg')
    return render_template("index.html", user_image=images)


@app.route('/recommend')
def recommend_ui():
    return render_template('recommend.html')


@app.route('/recommend_laptop', methods=['post'])
def recommend():
    # brand
    brand = request.form.get('input_brand')
    brand_1 = np.where(df['Company'].unique())
    print(f"brand name is: {brand}")

    # type
    model = request.form.get('input_model')
    #model_1 = np.where(df['TypeName']== )
    print(f"model type is: {model}")

    # ops
    ops = request.form.get('input_os')
    #ops_1 = np.where(df['os'].unique())
    print(f"operating sys name is: {ops}")

    # cupbrand
    cpu = request.form.get('input_cpu')
    # cpu = np.where(df['CpuBrand'].unique())
    print(f"processor name is: {cpu}")

    # ram
    ram = request.form.get('input_ram')
    #ram_1 = [2, 4, 6, 8, 12, 16, 24, 32, 64]
    print(f"ram size is: {ram}")

    # gupbrand
    gpu = request.form.get('input_gpu')
    # gpu = np.where(df['GpuBrand'].unique())
    print(f"graphics name is: {gpu}")

    # hdd
    hdd = request.form.get('input_hdd')
    # hdd = [0, 128, 256, 512, 1024, 2048]
    print(f"hdd size is: {hdd}")

    # ssd
    ssd = request.form.get('input_ssd')
    # ssd = [0, 8, 128, 256, 512, 1024]
    print(f"ssd size is: {ssd}")

    # touchscreen
    touchscreen = request.form.get('input_touchscreen')
    # touchscreen = ['Yes', 'No']
    print(f"touchscreen is: {touchscreen}")

    # ips
    ips = request.form.get('input_ips')
    # ips = ['Yes', 'No']
    print(f"ips is: {ips}")

    # resolution
    resolution = request.form.get('input_resolution')
    # resolution = ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800', '2560x1600',
    # '2560x1440','2304x1440']
    print(f"resolution is: {resolution}")

    # screensize
    screensize = request.form.get('input_screensize')
    screen = float(screensize)
    print(f"screensize is: {screensize}")

    # # weight removed from pickel file
    # weight = request.form.get('input_weight')
    # # weight = df['Weight']
    # print(f"weight is : {weight}")

    # ips/touchscreen

    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0

    # ppi
    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res ** 2) + (Y_res ** 2)) ** 0.5 / screen

    query = np.array([brand, model, ram, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, ops])

    query = query.reshape(1, 11)

    data = str(int(np.exp(pipe.predict(query)[0])))
    print(f"The approximate price of Laptop is: {data}")

    return render_template('recommend.html', data=data)


if __name__ == '__main__':
    #app.run(debug=True)
    app.run(host='0.0.0.0', port =8080)
