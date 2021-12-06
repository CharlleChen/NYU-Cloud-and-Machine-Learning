import torch
import torchvision.models as models
from torchvision import datasets, transforms as T
import torchvision
# from IPython.display import Image, display
from PIL import Image
from flask import Flask, json, request


transform = T.Compose([T.Resize(256),
                      T.CenterCrop(224),
                      T.ToTensor(),
                      T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
def preprocess(image):
    image = transform(image)
    image = image.unsqueeze(0)
    return image

idx_to_label = eval(open('imagenet1000_clsidx_to_labels.txt', 'r').read())

device = torch.device('cpu')

def inference(inputs):

    model = models.mobilenet_v2(pretrained=False)
    model.load_state_dict(torch.load('/mnt/mobilenetv2.pth'))
    model.eval()

    inputs = inputs.to(device)
    output = model(inputs)
    index = output.data.numpy().argmax()

    return idx_to_label[index]

api = Flask(__name__)

@api.route('/inference', methods=['POST'])
def get_result():
    res = {}
    file = request.files['image']
    if not file:
        res['status'] = 'missing image'
    else:
        res['status'] = 'success'
        image = Image.open(file.stream)
        ans = inference(preprocess(image))
        res['ret'] = ans

    return json.dumps(res)

api.run(host='0.0.0.0')

