from flask import Flask, render_template, request
import numpy as np
import base64
import random
from scipy.misc import imread, imresize
from skimage import io
import tensorflow as tf
from keras.models import load_model
from prepare_data import normalize
from tensorflow.python.keras.backend import set_session
import json
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform

config = tf.ConfigProto(
    device_count={'GPU': 1},
    intra_op_parallelism_threads=1,
    allow_soft_placement=True
)

config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.6

session = tf.Session(config=config)
set_session(session)

app = Flask(__name__)

mlp = load_model("./models/mlp7.h5")
# with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
#     conv = load_model("./models/knn_93.h5")
conv = load_model("./models/objects_3.h5")
graph = tf.get_default_graph()
OBJECTS = {0: "Airplane", 1: "Wine Bottle", 2: "Butterfly", 3: "Banana",4:"T-Shirt",5:"Umbrella",6:"Grapes"}


@app.route("/", methods=["GET", "POST"])
def ready():
    with session.as_default():
        with session.graph.as_default():
            if request.method == "GET":
                return render_template("index1.html")
            if request.method == "POST":
                data = request.form["payload"].split(",")[1]
                type = request.form["type"]
                gan = request.form["gan"]
                print(gan)
                print(type)
                # net = "ConvNet"
                if type == "CanvasCNN" or type == "CanvasMLP":
                    img = base64.b64decode(data)
                    with open('temp.png', 'wb') as output:
                        output.write(img)
                    x = imread('temp.png', mode='L')
                if type == "GAN":
                    x = imread('./static/{}.png'.format(gan), mode='L')


                x = imresize(x, (28, 28))
                #io.imshow(x)
                #io.show()

                if type == "CanvasMLP":
                    model = mlp
                    print("Selected MLP model")
                    # invert the colors
                    x = np.invert(x)
                    # flatten the matrix
                    x = x.flatten()
                    print(x.shape)
                    # brighten the image a bit (by 60%)
                    for i in range(len(x)):
                        if x[i] > 50:
                            x[i] = min(255, x[i] + x[i] * 0.60)
                    x = normalize(x)

                else:
                    model = conv
                    print("Selected CNN model")
                    x = np.expand_dims(x, axis=0)
                    x = np.reshape(x, (28, 28, 1))
                    # invert the colors
                    x = np.invert(x)

                    # brighten the image by 60%
                    for i in range(len(x)):
                        for j in range(len(x)):
                            if x[i][j] > 50:
                                x[i][j] = min(255, x[i][j] + x[i][j] * 0.60)
                    x = normalize(x)
                    x = x.reshape(28, 28, 1)


                # normalize the values between -1 and 1

                val = model.predict(np.array([x]))
                pred = OBJECTS[np.argmax(val)]
                classes = ['airplane', 'wine bottle', 'butterfly', 'banana', 't-shirt', 'umbrella', 'grapes']
                print(pred)
                print(list(val[0]))
                return render_template("index1.html", predicted = pred, preds=list(val[0]), classes=json.dumps(classes), chart=True, putback=request.form["payload"])


app.run()