# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 00:00:14 2018

@author: Manisha
"""

import os
import glob
import librosa
import matplotlib.pyplot as plt
import scipy
import numpy as np
import wave
from sklearn.ensemble import GradientBoostingClassifier as GBC




def main():
	path= r'C:\Users\Manisha\Documents\free-spoken-digit-dataset-master\recordings'
	path_test=r'C:\Users\Manisha\Documents\free-spoken-digit-dataset-master\test recording'
	zero=[]
	X=[]
	Y=[]
	X_test=[]
	for filename in glob.glob(os.path.join(path, '*.wav')):
		label=filename[71]
		Y.append(label)
		w=wave.open(filename, 'r')
		d=w.readframes(w.getnframes())
		zero.append(d)
		w.close()
		X_, sample_rate = librosa.load(filename, res_type='kaiser_fast')
		mfccs = np.mean(librosa.feature.mfcc(y=X_, sr=sample_rate, n_mfcc=40).T,axis=0)
		X.append(mfccs)
	for filename in glob.glob(os.path.join(path_test, '*.wav')):
           
		print(filename[75])
		w=wave.open(filename, 'r')
		d=w.readframes(w.getnframes())
		zero.append(d)
		w.close()
		X_, sample_rate = librosa.load(filename, res_type='kaiser_fast')
		mfccs = np.mean(librosa.feature.mfcc(y=X_, sr=sample_rate, n_mfcc=40).T,axis=0)
		X_test.append(mfccs)
	gbc = GBC(n_estimators=50, max_depth=5,min_samples_leaf=200,max_features=10,verbose=1)
	gbc.fit(X,Y)
	result=gbc.predict(X_test)
	print(result)	
	print('We are done')

main()
