import numpy as np
import pandas as pd
import cv2
import mahotas
import matplotlib.pyplot as plt
import seaborn as sns
from skimage import io, feature, measure
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

print("✅ All required libraries are installed correctly!")
