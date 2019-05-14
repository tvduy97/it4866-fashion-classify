from django.shortcuts import render
from .forms import UploadFileForm
from django.http import JsonResponse
from .predict import handle_uploaded_file
import tensorflow as tf
import string
import random
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import calendar
import time

data_folder = "myapp/"

file_to_open = data_folder + "model"
model = tf.keras.models.load_model(file_to_open)


def index(request):
    form = UploadFileForm()
    return render(request, 'index.html', {'form': form})


def classify(request):
    file = request.FILES['file']
    filename = str(calendar.timegm(time.gmtime())) + ''.join(
        random.choice(string.ascii_letters) for x in range(8))
    path = default_storage.save(
        filename + '.png', ContentFile(file.read()))
    predict = handle_uploaded_file('media/' + path, model)
    return JsonResponse(predict)
