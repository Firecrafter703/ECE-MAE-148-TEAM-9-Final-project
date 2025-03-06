from depthai_sdk import OakCamera

# Download & deploy a model from Roboflow universe:
# # https://universe.roboflow.com/david-lee-d0rhs/american-sign-language-letters/dataset/6

with OakCamera(usb_speed='usb2') as oak:
    color = oak.create_camera('color')
    model_config = {
        #'american-sign-language-letters/6'
        #american-sign-language-letters-prnzs/1
        #detect-count-and-visualize-object-detection-yuq5l/3
        #ggQjV8UgC2yOO6wzm1eg
        #stTXBOgAPxeZGqLnoGda
        'source': 'roboflow', # Specify that we are downloading the model from Roboflow
        'model':'detect-count-and-visualize-object-detection-yuq5l/3',
        'key':'ggQjV8UgC2yOO6wzm1eg' 
    }
    nn = oak.create_nn(model_config, color)
    oak.visualize(nn, fps=True)
    oak.start(blocking=True)