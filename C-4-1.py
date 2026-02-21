import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
df = pd.read_csv("shopping_trends.csv")
features = df[['Age', 'Purchase Amount (USD)']]