# -*- coding: utf-8 -*-
"""
Created on Sun Feb  1 00:13:13 2026

@author: Dhruv P Trial
"""
import numpy as np
Features = np.load(r"C:\Users\User\Downloads\RavDess_features.npy")
Labels = np.load(r"C:\Users\User\Downloads\RavDess_labels.npy")
TempFeatures = Features.transpose(0,3,1,2)
Indices = np.argsort(Labels)
New_Labels = Labels[Indices]
New_Features = TempFeatures[Indices]
Final_Features = (New_Features - np.mean(New_Features))/np.std(New_Features)
ChangeElements = []
Prev = -1
for i in range(4320):
    if Prev != New_Labels[i]:
        ChangeElements.append(i)
        Prev = New_Labels[i]
ChangeElements.append(4320)
ValFeatures = np.zeros((432,1,128,130))
TestFeatures = np.zeros((432,1,128,130))
TrainFeatures = np.zeros((3500,1,128,130))
ValLabels = np.zeros(432) - 1
TestLabels = np.zeros(432) - 1
TrainLabels = np.zeros(3500) - 1
ValNo = 0
TestNo = 0
TrainNo = 0
for idx,element in enumerate(ChangeElements[:-1]):
    Diff = ChangeElements[idx+1] - element
    ValFeatures[ValNo:ValNo+Diff//10] = Final_Features[element:element+Diff//10]
    ValLabels[ValNo:ValNo+Diff//10] = New_Labels[element:element+Diff//10]
    TestFeatures[TestNo:TestNo + Diff//10] = Final_Features[element+Diff//10:element+2*(Diff//10)]
    TestLabels[TestNo:TestNo+Diff//10] = New_Labels[element+Diff//10:element+2*(Diff//10)]
    TrainFeatures[TrainNo:TrainNo+Diff - 2*(Diff//10)] = Final_Features[element+2*(Diff//10):element+Diff]
    TrainLabels[TrainNo:TrainNo+Diff - 2*(Diff//10)] = New_Labels[element+2*(Diff//10):element+Diff]
    ValNo += Diff//10
    TestNo += Diff//10
    TrainNo += Diff - 2*(Diff//10)
breakpoint()
np.save(r"D:\AI_Club\Train\Features.npy",TrainFeatures)
np.save(r"D:\AI_Club\Train\Labels.npy",TrainLabels)
np.save(r"D:\AI_Club\Val\Features.npy",ValFeatures)
np.save(r"D:\AI_Club\Val\Labels.npy",ValLabels)
np.save(r"D:\AI_Club\Test\Features.npy",TestFeatures)
np.save(r"D:\AI_Club\Test\Labels.npy",TestLabels)
