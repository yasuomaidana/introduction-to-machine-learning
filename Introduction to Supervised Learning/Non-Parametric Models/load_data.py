import pandas as pd
from scipy.io import arff
import requests
import os

mnist = "mnist_784"
csv_path = "data/{}.csv".format(mnist)
if not os.path.exists(csv_path):
    URL = "https://www.openml.org/data/download/52667/mnist_784.arff"
    response = requests.get(URL)
    arff_path = "data/{}.arff".format(mnist)
    open(arff_path, "wb").write(response.content)
    data = arff.loadarff(arff_path)
    data = pd.DataFrame(data[0])
    data = data.astype(int)
    data.to_csv(csv_path,index=False)
    os.remove(arff_path)

