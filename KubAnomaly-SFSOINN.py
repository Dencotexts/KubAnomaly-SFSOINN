import tensorflow as tf
from sklearn.svm import SVC
from tensorflow.keras.optimizers import Adam
import numpy as np
from sklearn.model_selection import train_test_split
import csv
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support, accuracy_score
import matplotlib.pyplot as plt
from SF_SOINN import SF_SOINN  # Ensure you have the SF_SOINN implementation
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg', depending on your installation


class Dataset():
    def __init__(self, path=None, Normalize=False, Normalize_method='std'):
        if path is None:
            print('Path cannot be None')
            exit()
        print(path)
        self.X = []
        self.Y = []
        with open(path, 'rt') as csvfile:
            csv_reader = csv.reader(csvfile)
            for idx, row in enumerate(csv_reader):
                if idx == 0:
                    continue
                self.X.append([float(v) for v in row[1:37]])
                self.Y.append([int(v) for v in row[37:]].index(1))
        if Normalize:
            self.X = np.array(self.X)
            if Normalize_method == 'std':
                normalizer = StandardScaler()
            elif Normalize_method == 'l2':
                normalizer = Normalizer()
            self.X = normalizer.fit_transform(self.X)
            self.Y = np.array(self.Y)
        else:
            self.X = np.array(self.X)
            self.Y = np.array(self.Y)

class KubAnomaly_Model():
    def __init__(self):
        self.train_X = []
        self.train_Y = []
        self.cnn_flag = False
        # Complex data set
        data_path = ['./Data/cmdinjection.csv', './Data/DVWA_Normal.csv', './Data/DVWA_SQLInjection1.csv', './Data/DVWA_SQLInjection2.csv',
                     './Data/DVWA_SQLInjection3.csv', './Data/sqlinject.csv', './Data/wordPressNormalandAttack/NormalV1.1.csv',
                     './Data/SqlandCommand/AttackV1.1.csv', './Data/SqlandCommand/InsiderSql.csv', './Data/SqlandCommand/NormalV1.2.csv',
                     './Data/brutforce/AttackV1.1.csv', './Data/brutforce/InsiderV1.1.csv']
        
        for path in data_path:
            ds = Dataset(path, Normalize=True, Normalize_method='l2')
            self.train_X.append(ds.X)
            self.train_Y.append(ds.Y)
        
        self.train_X = np.concatenate(self.train_X)
        self.train_Y = np.concatenate(self.train_Y)
        
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.train_X, self.train_Y, test_size=0.2, random_state=1)
        self.y_GT = self.Y_test

    def linear_svm(self):
        print("Linear SVM Testing...")
        clf = SVC(kernel='linear', C=1, random_state=1)
        clf.fit(self.X_train, self.Y_train)
        y_pred = clf.predict(self.X_test)
        self.evaluate_results(self.Y_test, y_pred, "Linear SVM")

    def rbf_svm(self):
        print("RBF SVM Testing...")
        clf = SVC(kernel='rbf', C=1, random_state=1)
        clf.fit(self.X_train, self.Y_train)
        y_pred = clf.predict(self.X_test)
        self.evaluate_results(self.Y_test, y_pred, "RBF SVM")

    def kubanomaly(self):
        print("KubAnomaly Testing...")
        # One-hot encoding for TensorFlow model
        self.Y_train_one_hot = tf.keras.utils.to_categorical(self.Y_train, 2)
        self.Y_test_one_hot = tf.keras.utils.to_categorical(self.Y_test, 2)
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(64, activation='elu', input_shape=(self.X_train.shape[1],)),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(32, activation='elu'),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(16, activation='elu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(2, activation='softmax')
        ])
        self.compile_evaluate()

    def cnn(self):
        print("CNN Testing...")
        self.X_train = np.expand_dims(self.X_train, axis=2)
        self.X_test = np.expand_dims(self.X_test, axis=2)
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Conv1D(16, kernel_size=5, input_shape=(self.X_train.shape[1], 1), padding='same', activation='elu'),
            tf.keras.layers.Conv1D(32, kernel_size=5, padding='same', activation='elu'),
            tf.keras.layers.MaxPooling1D(2),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(32, activation='elu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(16, activation='elu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(2, activation='softmax')
        ])
        self.cnn_flag = True
        self.compile_evaluate()

    def sf_soinn(self):
        print("SF-SOINN Testing...")
        
        # Initialize SF-SOINN with first 3 random samples from X_train
        model = SF_SOINN(x1=self.X_train[0], x2=self.X_train[1], x3=self.X_train[2], max_edge_age=50)
        
        # Use integer labels for SF-SOINN, not one-hot encoded labels
        for i, x in enumerate(self.X_train):
            label = 'Normal' if self.Y_train[i] == 0 else 'Attack'
            model.input_signal(x, y=label)
        
        # Testing phase - predicting for X_test
        y_pred = []
        for x in self.X_test:
            # Predict without learning (input_signal(x, learning=False))
            pred, _ = model.input_signal(x, learning=False)
            # Convert prediction back to integer format: 'Normal' -> 0, 'Attack' -> 1
            y_pred.append(0 if pred == 'Normal' else 1)
        
        # Evaluate the results with metrics (accuracy, precision, recall, F1, AUC)
        self.evaluate_results(self.Y_test, y_pred, "SF-SOINN")

    def compile_evaluate(self):
        self.model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=Adam(), metrics=['accuracy'])
        self.model.fit(self.X_train, self.Y_train_one_hot, batch_size=128, epochs=30, verbose=1, validation_data=(self.X_test, self.Y_test_one_hot))
        y_pred = np.argmax(self.model.predict(self.X_test), axis=1)
        self.evaluate_results(np.argmax(self.Y_test_one_hot, axis=1), y_pred, "CNN" if self.cnn_flag else "KubAnomaly")

    def evaluate_results(self, y_true, y_pred, model_name):
        acc = accuracy_score(y_true, y_pred)
        prec, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='micro')
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        auc_s = auc(fpr, tpr)
        print(f"{model_name} - ACC: {acc}, Precision: {prec}, Recall: {recall}, F1: {f1}, AUC: {auc_s}")
        # Save values for plotting
        setattr(self, f"{model_name}_fpr", fpr)
        setattr(self, f"{model_name}_tpr", tpr)
        setattr(self, f"{model_name}_auc", auc_s)

    def generate_plot_figure(self):
        plt.figure()
        lw = 1
        for model in ['Linear SVM', 'RBF SVM', 'KubAnomaly', 'CNN', 'SF-SOINN']:
            fpr = getattr(self, f"{model}_fpr", None)
            tpr = getattr(self, f"{model}_tpr", None)
            auc_val = getattr(self, f"{model}_auc", None)
            if fpr is not None and tpr is not None:
                plt.plot(fpr, tpr, lw=2, label=f'{model} ROC curve (AUC = {auc_val:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])  # Correctly closing the bracket
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.show()
        input("Press Enter to continue...")  # This line keeps the window open
if __name__ == '__main__':
    try:
        print("Starting model tests...")
        test = KubAnomaly_Model()
        test.linear_svm()
        test.rbf_svm()
        test.kubanomaly()
        test.cnn()
        test.sf_soinn()
        test.generate_plot_figure()
    except Exception as e:
        print(f"An error occurred: {e}")
