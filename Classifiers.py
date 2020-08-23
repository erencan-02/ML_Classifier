import pandas as pd 
import numpy as np
from sklearn import preprocessing, neighbors
from sklearn.model_selection import train_test_split


#Plotting
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
#import seaborn as sns



import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis





class Classifier:
    def __init__(self, classifier, data_set, lable_name, *, test_size=0.5):
        self.classifier = classifier
        self.data_set = data_set
        self.lable_name = lable_name
        self.test_size = test_size

        self.X = np.array(data_set)
        self.Y = np.array(data[lable_name])

        self.X_train, self.X_test, self.Y_train, self.Y_test = 4*[None] #train_test_split(X,y,test_size=0.1)


    def split(self):
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size=self.test_size)


    def train(self):
        self.split()
        self.classifier.fit(self.X_train, self.Y_train)


    def accuracy(self):
        return round(self.classifier.score(self.X_test, self.Y_test)*100, 2)


    def split_results(self):
        splits = list(map(lambda x: float(x/10), list(range(1, 10))))
        results = {}
        old_test_size = self.test_size
        for _ in splits:
            print(f'Splitting data with test size: {_*100}%')
            self.test_size = _
            self.split()
            self.train()
            acc = self.accuracy()
            results[int(_*100)] = acc

        max_result = max(results, key=lambda key: results[key])
        results['MAX'] = f'{max_result}%'
        self.test_size = old_test_size
        return results


    def classifier_info(self):
        return 'You are using the {}'.format(type(self.classifier).__name__)


    def plot(self):
        self.data_set.hist(bins=15, color='steelblue', edgecolor='black', linewidth=1.0, xlabelsize=8, ylabelsize=8, grid=False)    
        plt.show()


    
classifiers = {
    '1': KNeighborsClassifier(3),
    '2': SVC(kernel="linear", C=0.025),
    '3': SVC(gamma=2, C=1),
    '4': GaussianProcessClassifier(1.0 * RBF(1.0)),
    '5': DecisionTreeClassifier(max_depth=5),
    '6': RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    '7': MLPClassifier(alpha=1, max_iter=1000),
    '8': AdaBoostClassifier(),
    '9': GaussianNB(),
    '10': QuadraticDiscriminantAnalysis()
}



data = pd.read_csv("heart_failure_clinical_records_dataset.csv") 
clf = Classifier(classifiers['1'], data, 'DEATH_EVENT', test_size=0.001)
clf.train()
accuracy = clf.accuracy()
print(accuracy)
clf.plot()


