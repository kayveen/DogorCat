import os
from flask import Flask, render_template, request, send_from_directory
from keras.models import load_model
from keras_preprocessing import image
import numpy as np
import tensorflow as tf


app = Flask(__name__)

dir_path = os.path.dirname(os.path.realpath(__file__))
