# -*- coding: utf-8 -*-
"""
Created on Fri Dec 12 23:46:54 2025

@author: Dhruv P Trial
"""

import cupy as cp
import numpy as np
def TestAccuracy(DeepLearning, Parameter_Dict, ConvolutionalLayers,Layer_Lengths,
                 MidLayerActivation,LastLayerActivation,
           TestInputFilePath = r"D:\AI_Club\Test\Features.npy" ,
           TestOutputFilePath = r"D:\AI_Club\Test\Labels.npy",
           batch_len  = 256,section = 1,Dataset=None,DGiven = False):
    OutDataset = np.load(TestOutputFilePath)
    if DGiven:
        InpDataset = Dataset
        records = len(InpDataset)
    else:
        InpDataset = np.load(TestInputFilePath)
    records = len(InpDataset)
    InpDataset = InpDataset[0:int(section*records//1)]
    OutDataset = OutDataset[0:int(section*records//1)]
    records = len(InpDataset)
    if not DGiven: Dataset = cp.array(InpDataset,dtype=cp.float32)
    else: Dataset = InpDataset
    OutputDataset = cp.array(OutDataset,dtype=cp.int32)
    TotalOut = cp.zeros((records,))
    for counter in range(records//batch_len):
        Input = Dataset[counter*batch_len:min((counter+1)*batch_len,records)]
        PooledLayers = [Input]
        Layers = [Input.reshape((-1,batch_len))]
        ConvChecker = False
        #Convolutional Layers
        for ind,key in enumerate(ConvolutionalLayers):
            ConvChecker = True
            Kernel = Parameter_Dict[key[0]]
            Biases = Parameter_Dict[key[0] + "Bias"]
            Padding = int(key[3])
            Convolved = DeepLearning.Convolve4D(PooledLayers[-1], Kernel,Padding,Bias=Biases)
            Inter = DeepLearning.BatchNorm(Convolved)
            Convolved = DeepLearning.LeakyRelu(Inter)
            PSize = int(key[4])
            PType = key[5]
            PStride = int(key[6])
            Pooled,mask = DeepLearning.Pool(Convolved,PSize,Stride = PStride,PoolType=PType)
            PooledLayers.append(Pooled)
            Layers[0] = Pooled
        if ConvChecker:
            Layers[0] = Layers[0].reshape((Layers[0].shape[0],-1)).T
        #Feedforward Layers
        Buffer = Layers[0]
        for idx,Key in enumerate(Layer_Lengths):    
            if idx == 0: continue
            Wn = "w" + str(idx)
            Bn = "b" + str(idx)
            if idx != DeepLearning.Length:
                Buffer =  DeepLearning.Forward(Buffer, DeepLearning.Parameter_Dict[Wn], DeepLearning.Parameter_Dict[Bn], MidLayerActivation)
            else :
                Output_Layer =  DeepLearning.Forward(Buffer, DeepLearning.Parameter_Dict[Wn], DeepLearning.Parameter_Dict[Bn], LastLayerActivation)
        Output = cp.argmax(Output_Layer,axis=0,keepdims=True)
        TotalOut[counter*batch_len:(counter+1)*batch_len] = Output
    Equality = OutputDataset == TotalOut
    Correct = cp.sum(Equality)
    Accuracy = Correct/len(TotalOut)*100
    return Accuracy
