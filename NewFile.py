import cupy as cp
import cupyx as cpx
from cupy.lib.stride_tricks import as_strided
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle as pkl
import copy
import Convolutional_Testing as Tester
import time
class DeepLearning():
    def __init__(self):
        self.Layer_Lengths = Layer_Lengths
        self.Optimizer = Optimizer
        self.LossFunc = LossFunc
        self.Parameter_Dict = Parameter_Dict
        if self.Optimizer == "ADAM" or self.Optimizer == "Momentum":
            self.Nu = Nu
            self.Velocity_Dict = Velocity_Dict
        if self.Optimizer == "ADAM":    
            self.Delta = Delta
            self.Confidence_Dict = Confidence_Dict
        self.t = 0
        self.Length = len(Layer_Lengths) - 1
        self.MidLayerActivation = MidLayerActivation
        self.LastLayerActivation = LastLayerActivation
        self.alpha = alpha
        self.Beta = Beta
        self.ConvolutionalLayers = ConvolutionalLayers
        self.VerticalInp = [0 for i in range(len(self.ConvolutionalLayers))]
        self.HorizontalInp = [0 for i in range(len(self.ConvolutionalLayers))]
        self.Batches = [0 for i in range(len(self.ConvolutionalLayers))]
        self.Channels = [0 for i in range(len(self.ConvolutionalLayers))]
    def Softmax(self,matrix):
        max_val = cp.max(matrix,axis=0,keepdims=True)
        exp_values = cp.exp(matrix - max_val)
        Sum = cp.sum(exp_values,axis=0,keepdims=True)
        softmax_output = exp_values/Sum
        return softmax_output
    def Sigmoid(self,Matrix):
        return 1/(1+cp.exp(-Matrix))
    def ReLU(self,Matrix):
        Output = cp.fmax(Matrix,cp.zeros(Matrix.shape))
        return Output
    def LeakyRelu(self,Matrix):
        Output = cp.fmax(Matrix,0)
        Output += cp.fmin(Matrix,0)*0.1
        return Output
    def Convolve4D(self,Input,Kernels,PadWidth,Type = "Correlate",Bias = cp.zeros(1)):
        """ Currently Type = "Convolve" is set for backpropagation. 
        Any other use nescessitates chaning the function"""
        if len(Kernels.shape) != 4 or len(Input.shape) != 4:
            raise ValueError("Wrong Input Shapes")
        if Type== "Convolve":
            Len = len(Kernels.shape)
            Kernels = cp.flip(Kernels,axis=(Len-2,Len-1))
            Kernels = Kernels.swapaxes(0,1)
        KSize = Kernels.shape[-1]
        Padded = cp.pad(Input,((0,0), (0,0), (PadWidth, PadWidth), (PadWidth, PadWidth)))
        Bs,Cs,Hs,Vs = Padded.strides
        Mid1 = Padded.shape[2] - KSize + 1
        Mid2 = Padded.shape[3] - KSize + 1
        if Mid1 <=  0 or Mid2 <= 0:
            raise IndexError("Kernel is bigger than Input")
        shape = (Input.shape[0],Mid1,Mid2,Input.shape[1],KSize,KSize)
        Wind = as_strided(Padded,shape=shape,strides=(Bs,Hs,Vs,Cs,Hs,Vs))
        Output = cp.einsum('b h w c p q, o c p q -> b o h w', Wind, Kernels)#/Kernels.shape[1]
        return Output + Bias
    def ConvolveBackProp(self,Input,Kernels,PadWidth,Type = "Correlate",Bias = cp.zeros(1)):
        if len(Kernels.shape) != 4 or len(Input.shape) != 4:
            raise ValueError("Wrong Input Shapes")
        if Type== "Convolve":
            Len = len(Kernels.shape)
            Kernels = cp.flip(Kernels,axis=(Len-2,Len-1))
        Padded = cp.pad(Input,((0,0), (0,0), (PadWidth, PadWidth), (PadWidth, PadWidth)))
        B, C_out, H_out, W_out = Kernels.shape
        K_h = Padded.shape[2] - H_out + 1
        K_w = Padded.shape[3] - W_out + 1
        if K_h <=0 ^ K_w<=0:
            raise ValueError("One dimension can only be convolved")
        elif K_h <=  0 and K_w <= 0:
            return self.ConvolveBackProp(Kernels,Input,PadWidth,Type=Type,Bias=Bias) 
        shape = (B, Input.shape[1], K_h, K_w, H_out, W_out)        
        Bs,Cs,Hs,Ws = Padded.strides
        Window = as_strided(Padded, shape=shape, strides=(Bs, Cs, Hs, Ws, Hs, Ws))        
        Output = cp.einsum('b c h w x y, b d x y-> d c h w',Window,Kernels)/batch_len
        return Output + Bias
    def Pool(self,Input,Size,Stride = 1,PoolType = 'Max'):
        HKernels = (Input.shape[-2] - Size)//(Stride) + 1
        VKernels = (Input.shape[-1] - Size)//(Stride) + 1
        Batches,CStride,Horizontal,Vertical = Input.strides
        Window = as_strided(Input,
                            shape=(Input.shape[0],Input.shape[1],HKernels,VKernels,Size,Size),
                            strides=(Batches,CStride,Horizontal*Stride,Vertical*Stride,Horizontal,Vertical))
        if PoolType == 'Max':
            AlephNol = Window.max(axis=(4,5))
            mask = Window == AlephNol[:,:,:,:,None,None]
        elif PoolType == 'Avg':
            AlephNol = Window.mean(axis=(4,5))
            mask = cp.ones(Window.shape)
        return AlephNol,mask
    def Forward(self,Input,Weights,Bias,Activation):
        z = Weights @ Input + Bias
        if Activation == "Relu": return self.ReLU(z)
        elif Activation == "Leaky": 
            z = self.BatchNorm(z)
            return self.LeakyRelu(z)
        elif Activation == "Sigmoid": return self.Sigmoid(z)
        elif Activation == "Softmax": return self.Softmax(z)
        elif Activation == "None": return z
        else: raise KeyError("Activation was incorrect")
    def CSE_Loss(self,Outputs,Actual):
        loss = -cp.mean(cp.sum(Actual*cp.log(Outputs + 1e-9),axis=0))
        return loss
    def MSE_Loss(self,Outputs,Actual):
        loss = cp.mean((Outputs - Actual)**2,axis=0)
        return loss
    def Momentum(self,Momentum,Matrix):
        return Momentum*self.Nu + Matrix*(1-self.Nu)
    def ADAM(self,Momentum,Variance,Matrix,i=0):
        Velocity = Momentum*self.Nu + Matrix*(1-self.Nu)
        Confidence = Variance*self.Delta + cp.square(Matrix)*(1-self.Delta)
        V_Hat = Velocity/(1-self.Nu**self.t)
        C_Hat = Confidence/(1-self.Delta**self.t)
        return Velocity,Confidence,V_Hat,C_Hat
    def FF_Layer_Backprop(self,Layer_No,PreviousB_Adj_raw,ThisLayer,PreviousLayer,NextLayer,dropout=True):
        Keyb = "b" + str(self.Length - Layer_No)
        Keyw = "w" + str(self.Length - Layer_No)
        KeyPrevw = "w" + str(self.Length - Layer_No + 1)
        if Layer_No == 0: # Here 0 means output layer
            B_Mid = cp.array([])
            if self.LossFunc == "CSE" and (self.LastLayerActivation == "Softmax" or 
                                           self.LastLayerActivation == "Sigmoid"):
                b_adj_raw = (ThisLayer - Actual) * ClassWeights[:,None]
            else:
                if self.LossFunc == "CSE":
                    B_Mid = (ThisLayer - Actual)/(ThisLayer*(1-ThisLayer))
                elif self.LossFunc == "MSE":
                    B_Mid = 2*(ThisLayer - Actual)
                if self.LastLayerActivation == "Softmax":
                    dot = cp.sum(B_Mid * ThisLayer, axis=0, keepdims=True)
                    b_adj_raw = ThisLayer * (B_Mid - dot)
                elif self.LastLayerActivation == "Sigmoid":
                    b_adj_raw = B_Mid*ThisLayer*(1-ThisLayer)
                elif self.LastLayerActivation == "Relu":
                    b_adj_raw =B_Mid*(ThisLayer> 0).astype(cp.float32)
        else:
            Sigmas_Mid = self.Parameter_Dict[KeyPrevw].T @ PreviousB_Adj_raw
            if self.MidLayerActivation == "Relu":
                b_adj_raw =Sigmas_Mid*(ThisLayer> 0).astype(cp.float32)
            elif self.MidLayerActivation == "Sigmoid":
                b_adj_raw = Sigmas_Mid*ThisLayer*(1-ThisLayer)
            elif self.MidLayerActivation == "Leaky":
                b_adj_raw = Sigmas_Mid*(ThisLayer> 0).astype(cp.float32)*0.9 + 0.1*Sigmas_Mid
        b_adj = cp.mean(b_adj_raw,axis=1,keepdims=True)
        w_adj = b_adj_raw @ PreviousLayer.T/batch_len
        if self.Optimizer == "ADAM":
            self.Velocity_Dict[Keyb + "_vel"],self.Confidence_Dict[Keyb + "_Sq"],b_m_hat, b_c_hat = self.ADAM(
                self.Velocity_Dict[Keyb + "_vel"],self.Confidence_Dict[Keyb + "_Sq"],b_adj)
            self.Velocity_Dict[Keyw + "_vel"],self.Confidence_Dict[Keyw + "_Sq"],w_m_hat, w_c_hat = self.ADAM(
                self.Velocity_Dict[Keyw + "_vel"],self.Confidence_Dict[Keyw + "_Sq"],w_adj)
            return w_m_hat,b_m_hat,w_c_hat,b_c_hat,b_adj_raw
        elif self.Optimizer == "Momentum":
            self.Velocity_Dict[Keyb + "_vel"] = b_vel = self.Momentum(self.Velocity_Dict[Keyb + "_vel"],b_adj)
            self.Velocity_Dict[Keyw + "_vel"] = w_vel = self.Momentum(self.Velocity_Dict[Keyw + "_vel"],b_adj)
            return w_vel,b_vel,b_adj_raw
        elif self.Optimizer == "SGD":
            return w_adj,b_adj
    def BackProp(self,Actual,Nu,Delta):
        self.t += 1
        self.PreviousB_Adj_raw = cp.array([])
        ReturnDict = {}
        #Feedforward Backprop
        for Inverted_Layer_No,Layer_Length in enumerate(Layer_Lengths[::-1]):
            if Inverted_Layer_No != self.Length : 
                Layer_Outs = self.FF_Layer_Backprop(Inverted_Layer_No, self.PreviousB_Adj_raw, 
                            Layers[-1-Inverted_Layer_No], Layers[-2-Inverted_Layer_No], Layers[-Inverted_Layer_No])
                basicB = "b" + str(self.Length-Inverted_Layer_No)
                basicW = "w" + str(self.Length-Inverted_Layer_No)
                if self.Optimizer == "ADAM":
                    KeyList = [basicW+"_m_hat",basicB+"_m_hat",basicW+"_c_hat",basicB+"_c_hat"]
                    AppendDict = dict(zip(KeyList,Layer_Outs[0:4]))
                    self.PreviousB_Adj_raw = Layer_Outs[4]
                elif self.Optimizer == "Momentum":
                    KeyList = [basicW+"_vel",basicB+"_vel"]
                    AppendDict = dict(zip(KeyList,Layer_Outs[0:2]))
                    self.PreviousB_Adj_raw = Layer_Outs[2]
                elif self.Optimizer == "SGD":
                    KeyList = [basicW+"_adj",basicB+"_adj"]
                    AppendDict = dict(zip(KeyList,Layer_Outs[0:2]))
                    self.PreviousB_Adj_raw = Layer_Outs[1]
                ReturnDict.update(AppendDict)
        #Convolutional BackProp
        self.PassInput = self.PreviousB_Adj_raw.T @ self.Parameter_Dict["w1"]
        for f_index in range(len(self.ConvolutionalLayers)):
            Shape  = PooledLayers[-(1+f_index)].shape
            if f_index == 0: self.PoolGrad = self.PassInput.reshape(Shape)
            ExpandedGrad = self.PoolGrad[:,:,:,:,None,None]
            BackedGrads = ExpandedGrad*masks[-(f_index + 1)]
            if self.t == 1: #This is to generate the indices for scatter add
                #They stay the same for every iteration so they are generated only once
                PoolKernelSize = int(self.ConvolutionalLayers[-(f_index + 1)][4])
                Stride = int(self.ConvolutionalLayers[(-1-f_index)][6])
                ValuesPerRecord = Shape[1]*Shape[2]*Shape[3]*PoolKernelSize*PoolKernelSize
                Batch = cp.repeat(cp.arange(Shape[0],dtype=cp.int32),ValuesPerRecord)
                ValuesPerChannel = Shape[2]*Shape[3]*PoolKernelSize*PoolKernelSize
                ChannelOne = cp.repeat(cp.arange(Shape[1],dtype=cp.int32),ValuesPerChannel)
                Channels = cp.tile(ChannelOne,Shape[0])
                MainArr = cp.array([],dtype=cp.int32)
                WeirdArr = cp.array([],dtype=cp.int32)
                KernelArr = cp.tile(cp.arange(PoolKernelSize,dtype=cp.int32),PoolKernelSize)
                Mid = cp.repeat(cp.arange(0,PoolKernelSize,dtype=cp.int32),PoolKernelSize)
                TilerArr = cp.tile(Mid,BackedGrads.shape[-3])
                for h in range(BackedGrads.shape[-4]): # Vertical Size
                    temp = cp.add(TilerArr,h*Stride,dtype=cp.int32)
                    MainArr = cp.concatenate((MainArr,temp))
                for g in range(BackedGrads.shape[-3]): # Horizontal Size
                    PreTemp = cp.add(KernelArr,g*Stride,dtype=cp.int32)
                    WeirdArr = cp.concatenate((WeirdArr,PreTemp))
                VerticalInp = cp.tile(MainArr,Shape[0]*Shape[1])
                HorizontalInp = cp.tile(WeirdArr,Shape[0]*Shape[1]*Shape[2])
                self.Batches[f_index]  = Batch
                self.Channels[f_index] = Channels
                self.VerticalInp[f_index] = VerticalInp
                self.HorizontalInp[f_index] = HorizontalInp
            PrevLayer = PooledLayers[-2-f_index]
            self.PoolGrad = cp.zeros(ConvolvedLayers[-1-f_index].shape)
            ScShape = (self.Batches[f_index],self.Channels[f_index],self.VerticalInp[f_index],self.HorizontalInp[f_index])
            cpx.scatter_add(self.PoolGrad, ScShape, BackedGrads.ravel())
            Layer_Name = self.ConvolutionalLayers[-1-f_index][0]
            Kernels = self.Parameter_Dict[Layer_Name]
            ForwardPadding =int(self.ConvolutionalLayers[-(1+f_index)][3])
            ConvolutionalKernelSize = int(self.ConvolutionalLayers[-(f_index + 1)][2])
            self.KernelBackProp = self.ConvolveBackProp(PrevLayer.astype(cp.float32),self.PoolGrad.astype(cp.float32),ForwardPadding)#/(self.PoolGrad.shape[2]*self.PoolGrad.shape[3])
            self.InputBackProp = self.Convolve4D(self.PoolGrad,Kernels,ConvolutionalKernelSize -ForwardPadding-1,Type='Convolve')#/(int(self.ConvolutionalLayers[-(1+f_index)][2])**2)
            self.BiasBackProp = cp.sum(self.PoolGrad,axis=(0,2,3),keepdims=True)[0]/(self.PoolGrad.shape[0]*self.PoolGrad.shape[2]*self.PoolGrad.shape[3])
            self.PoolGrad = self.InputBackProp
            if self.Optimizer == "ADAM":
                self.Velocity_Dict[Layer_Name],self.Confidence_Dict[Layer_Name],K_m_hat, K_c_hat = self.ADAM(self.Velocity_Dict[Layer_Name],self.Confidence_Dict[Layer_Name],self.KernelBackProp,i=1)
                self.Velocity_Dict[Layer_Name + "Bias"],self.Confidence_Dict[Layer_Name + "Bias"],B_m_hat, B_c_hat = self.ADAM(self.Velocity_Dict[Layer_Name + "Bias"],self.Confidence_Dict[Layer_Name + "Bias"],self.BiasBackProp)
                ReturnDict[Layer_Name] = [K_m_hat,K_c_hat]
                ReturnDict[Layer_Name + "Bias"] = [B_m_hat,B_c_hat]
            elif self.Optimizer == "Momentum":
                self.Velocity_Dict[Layer_Name] = K_vel = self.Momentum(self.Velocity_Dict[Layer_Name],self.KernelBackProp)
                self.Velocity_Dict[Layer_Name + "Bias"] = B_vel = self.Momentum(self.Velocity_Dict[Layer_Name + "Bias"],self.BiasBackProp)
                ReturnDict[Layer_Name] = K_vel
                ReturnDict[Layer_Name + "Bias"] = B_vel
            elif self.Optimizer == "SGD":
                ReturnDict[Layer_Name] = self.KernelBackProp
                ReturnDict[Layer_Name + "Bias"] = self.BiasBackProp
            ReturnDict["Main_Input"] = self.InputBackProp
        return ReturnDict
    def Correct(self,ReturnDict,CorrectInput = False):
        """ Todo: Input Correction Code to be written here
                Dict Key is "Main_Input"""
        #Convolutional BackProp
        for index,Layer in enumerate(self.ConvolutionalLayers):
            Name = Layer[0]
            self.Parameter_Dict
            if self.Optimizer == "ADAM":
                self.Parameter_Dict[Name + "Bias"] -= self.Beta*ReturnDict[Name + "Bias"][0]/cp.sqrt(ReturnDict[Name + "Bias"][1] + 1e-18)
                self.Parameter_Dict[Name]-= self.Beta*ReturnDict[Name][0]/cp.sqrt(ReturnDict[Name][1] + 1e-18)
            elif self.Optimizer == "Momentum":
                self.Parameter_Dict[Name + "Bias"] -= self.alpha*ReturnDict[Name]
                self.Parameter_Dict[Name] -= self.alpha*ReturnDict[Name + "Bias"]
            elif self.Optimizer == "SGD":
                self.Parameter_Dict[Name+ "Bias"] -= self.alpha*ReturnDict[Name + "Bias"]
                self.Parameter_Dict[Name] -= self.alpha*ReturnDict[Name]
        #FeedForward BackProp
        for Layer_No,Layer in enumerate(Layer_Lengths):
            if Layer_No != 0:
                Keyb = "b" + str(Layer_No)
                Keyw = "w" + str(Layer_No)
                if self.Optimizer == "ADAM":
                    self.Parameter_Dict[Keyb] -= self.alpha*ReturnDict[Keyb + "_m_hat"]/cp.sqrt(ReturnDict[Keyb + "_c_hat"] + 1e-8)
                    self.Parameter_Dict[Keyw] -= self.alpha*ReturnDict[Keyw + "_m_hat"]/cp.sqrt(ReturnDict[Keyw + "_c_hat"] + 1e-8)
                elif self.Optimizer == "Momentum":
                    self.Parameter_Dict[Keyb] -= self.alpha*ReturnDict[Keyb + "_vel"]
                    self.Parameter_Dict[Keyw] -= self.alpha*ReturnDict[Keyw + "_vel"]
                elif self.Optimizer == "SGD":
                    self.Parameter_Dict[Keyb] -= self.alpha*ReturnDict[Keyb + "_adj"]
                    self.Parameter_Dict[Keyw] -= self.alpha*ReturnDict[Keyw + "_adj"]
        
    def OneHots(self,indices,Matrix_Size):
        HotsIntermediate = cp.zeros((Matrix_Size,len(indices)))
        a = cp.arange(len(indices))
        HotsIntermediate[indices.flatten(),a.flatten()] = 1
        return HotsIntermediate
    def BatchNorm(self,Matrix,axis=0):
        mean = cp.mean(Matrix,axis=axis,keepdims = True)
        std = cp.std(Matrix,axis=axis,keepdims = True)
        return(Matrix - mean)/(std + 1e-8)
    def Dropout(self,Output,rate):
        mask = (cp.random.random(Output.shape) < rate).astype(cp.float32)
        NewOutput = Output*mask/rate
        return mask,NewOutput
TestCutoff = 10*1000 #1000 is there to convert millis into secs
"""                     Editable Values                   """
InputsFilePath = r"D:\AI_Club\RevisedData\emotion_data_X_train.npy"
ActualFilePath = r"D:\AI_Club\RevisedData\emotion_data_Y_train.npy"
OutputFolderPath = r"D:\ConvolutionData\OutputParametersGPAI\HelloWorld\\"
epochs = 100 #int(input("Enter the number of epochs for which the code should run:"))
ConvLayers = 5 #int(input("Please enter the number of Convolutional Layers: ")) #3
ConvolutionalLayers = [['0','1','2','1','2','Avg','1'],
                       ['1','4','2','1','1','Avg','1'],
                       ['2','8','3','1','2','Avg','1'],
                       ['3','16','3','0','2','Avg','2'],
                       ['4','32','5','0','2','Avg','2'],
                       ['5','64','5','0','2','Avg','2']]
#print("The format for entering is 'Output Channels 'Kernel size'  'Padding'  'PoolSize' 'PoolType' 'PoolStride'")
#print("Max Pool --> Max")
#print("Avg Pool --> Avg")
#for CLayer in range(ConvLayers):
#    Layer = list (input (f"Convolutional Layer {CLayer + 1}: ") . split ())
#    Layer.insert(0,str(CLayer))
#    if (Layer[0][0] == "w" or Layer[0][0] == "b") and len(Layer) == 2:
#        raise KeyError("May be a Keyword")
#    if (int(Layer[3])>int(Layer[2])):
#        raise ValueError("Kernel Size smaller than padding")
#    ConvolutionalLayers.append(Layer)
Layer_Lengths = [128,32] #list (map (int,input("Enter the Layers of the Neural Network Separated By a Space: ").split()))
alpha = 0.001 #float(input("Enter the Alpha Value:")) #0.002
if ConvLayers: Beta = 0.001 #float(input("Enter Beta Value: ")) #0.2
else: Beta = 0.2 #To avoid errors... does a bad job of it actually
batch_len  =  16 #int(input("Enter the Batch Length: ")) #256
Optimizer = "ADAM" #input("Enter the Optimzer: ") #"ADAM"
if Optimizer == "ADAM" or Optimizer == "Momentum":
    Nu = 0.9 #float(input("Enter the Nu Value:")) #0.9 
    Velocity_Dict = {}
if Optimizer == "ADAM":
    Delta = 0.999 #float(input("Enter the Delta Value: ")) #0.999
    Confidence_Dict = {}
LossFunc = "CSE" #input("Enter the Loss Function: ").upper() #"CSE"
MidLayerActivation = "Leaky" #input("Enter the Activation for all but the last layer: ").capitalize() #"Relu"
LastLayerActivation = "Softmax" #input("Enter the Activation for the last layer: ").capitalize() #"Softmax"
"""                     Editable Values                   """

""" Warning: If you're running without any convolutional layers, ensure that all
zero values become positive post normalization ensuring the neurons don't die"""

PreDataset = np.load(InputsFilePath)
ActualsDataset = np.load(ActualFilePath).astype(np.int16)
PreDataset = PreDataset.transpose(0,3,1,2)
Indices = np.arange(len(ActualsDataset))
counts = np.bincount(ActualsDataset)
weights = counts.sum() / (len(counts) * counts)
ClassWeights = cp.array(weights,dtype=cp.float32)
if len(PreDataset.shape) == 3:
    records,Height,Width = PreDataset.shape
    Channels = 1
else: records,Channels,Height,Width = PreDataset.shape
InpHeight = Height
InpWidth = Width
Input_Len = Height*Width
if len(ConvolutionalLayers):
    for Layer in ConvolutionalLayers:
        KernelSize = int(Layer[2])
        Padding = int(Layer[3])
        InpWidth -= KernelSize -1 - 2*Padding
        InpHeight -= KernelSize - 1 - 2*Padding
        InpWidth -= int(Layer[4]) 
        InpWidth /= int(Layer[6])
        InpWidth += 1
        InpHeight -= int(Layer[4]) 
        InpHeight /= int(Layer[6])
        InpHeight += 1
        Input_Len = int(InpHeight)*int(InpWidth)*int(ConvolutionalLayers[-1][1])
Output_Len = len(np.unique(ActualsDataset))
Layer_Lengths.append(Output_Len)
Layer_Lengths.insert(0,Input_Len)
# !!! Change this later
Parameter_Dict = {}
#Convolutional Part
KK = [0 for i in range(len(ConvolutionalLayers))]
Changes = []
for idx,element in enumerate(ConvolutionalLayers):
    numb = int(element[1]) 
    s = int(element[2])
    if idx == 0:
        Parameter_Dict[element[0]] = cp.random.randn(numb,Channels,s,s)/(s*cp.sqrt(Channels))
        Parameter_Dict[element[0] + "Bias"] = cp.random.randn(numb,1,1)/cp.sqrt(numb)
        Shpe = (numb,Channels,s,s)
    else:
        Parameter_Dict[element[0]] = cp.random.randn(numb,int(ConvolutionalLayers[idx-1][1]),s,s)/(s*cp.sqrt(int(ConvolutionalLayers[idx-1][1])))
        Parameter_Dict[element[0] + "Bias"] = cp.random.randn(numb,1,1)/cp.sqrt(numb)
        Shpe = (numb,int(ConvolutionalLayers[idx-1][1]),s,s)
    #Optimize Dict Storage. Separate Bias Store is short term patch
    if Optimizer == "ADAM":
        Velocity_Dict[element[0]] = cp.zeros(Shpe)
        Confidence_Dict[element[0]] = cp.zeros(Shpe)
        Velocity_Dict[element[0] + "Bias"] = cp.zeros((numb,1,1))
        Confidence_Dict[element[0] + "Bias"] = cp.zeros((numb,1,1))
    elif Optimizer == "Momentum":
        Velocity_Dict[element[0]] = cp.zeros(Shpe)
        Velocity_Dict[element[0] + "Bias"] = cp.zeros((numb,1,1))
    elif Optimizer == "SGD":pass
    else: raise NotImplementedError("Unknown Optimizer")
#FeedForward Part
for index,Length in enumerate(Layer_Lengths):
    if index!=0:
        Wn = "w" + str(index)
        Parameter_Dict[Wn] = 1.414*cp.random.randn(Length,Layer_Lengths[index-1])/cp.sqrt(Layer_Lengths[index-1])
        Bn = "b" + str(index)
        Parameter_Dict[Bn] = cp.random.randn(Length,1)
        if Optimizer == "ADAM":
            VWn = "w" + str(index) + "_vel"
            Velocity_Dict[VWn] = cp.zeros((Length,Layer_Lengths[index-1]))
            VBn = "b" + str(index) + "_vel"
            Velocity_Dict[VBn] = cp.zeros((Length,1))
            CWn = "w" + str(index) + "_Sq"
            Confidence_Dict[CWn] = cp.zeros((Length,Layer_Lengths[index-1]))
            CBn = "b" + str(index) + "_Sq"
            Confidence_Dict[CBn] = cp.zeros((Length,1))
        elif Optimizer == "Momentum":
            VWn = "w" + str(index) + "_vel"
            Velocity_Dict[VWn] = cp.zeros((Length,Layer_Lengths[index-1]))
            VBn = "b" + str(index) + "_vel"
            Velocity_Dict[VBn] = cp.zeros((Length,1))
        elif Optimizer == "SGD": pass
DeepLearning = DeepLearning()
Dataset = cp.array(PreDataset,dtype=cp.float32)
OutputDataset = cp.array(ActualsDataset,dtype=cp.int32)
LossList = cp.array([])
Current_Layer = cp.array([])
print("Starting Main Loop")
MaxAccuracy = 0
FinalParams = {}
TestTime = 0
Section = 1
LogChanges = []
for ep in range(epochs):
    TotalOut = cp.zeros((records,))
    for counter in range(records//batch_len - 1):
        Input = Dataset[counter*batch_len:(counter+1)*batch_len]
        Actual_Raw = OutputDataset[counter*batch_len:(counter+1)*batch_len]
        Actual = DeepLearning.OneHots(Actual_Raw, Output_Len)
        ConvolvedLayers = [Input]
        PooledLayers = [Input]
        Layers = [Input.reshape((-1,batch_len))]
        masks= []
        DropoutMasks = []
        ConvChecker = False
        #Convolutional Layers
        for ind,item in enumerate(ConvolutionalLayers):
            ConvChecker = True
            Kernel = DeepLearning.Parameter_Dict[item[0]]
            Biases = DeepLearning.Parameter_Dict[item[0] + "Bias"]
            Convolved = DeepLearning.Convolve4D(PooledLayers[-1], Kernel,int(item[3]),Bias=Biases)
            Convolved = DeepLearning.LeakyRelu(Convolved)
            Pooled,mask = DeepLearning.Pool(Convolved,int(item[4]),Stride = int(item[6]),PoolType=item[5])
            ConvolvedLayers.append(Convolved)
            PooledLayers.append(Pooled)
            masks.append(mask)
            Layers[0] = Pooled
        if ConvChecker:
            Layers[0] = Layers[0].reshape((Layers[0].shape[0],-1)).T
            Layers[0] = DeepLearning.BatchNorm(Layers[0])
        #Feedforward Layers
        for index,Layer in enumerate(Layer_Lengths):
            Wn = "w" + str(index)
            Bn = "b" + str(index)
            if index !=0 :
                if index != DeepLearning.Length:
                    Current_Layer =  DeepLearning.Forward(Layers[-1], DeepLearning.Parameter_Dict[Wn], DeepLearning.Parameter_Dict[Bn], MidLayerActivation)
                    Dmask,Current_Layer = DeepLearning.Dropout(Current_Layer,0.8)
                    DropoutMasks.append(Dmask)
                else :
                    Current_Layer =  DeepLearning.Forward(Layers[-1], DeepLearning.Parameter_Dict[Wn], DeepLearning.Parameter_Dict[Bn], LastLayerActivation)
                Layers.append(Current_Layer)
        if LossFunc == "CSE":
            Loss = DeepLearning.CSE_Loss(Layers[-1],Actual)
        elif LossFunc == "MSE":
            Loss = DeepLearning.MSE_Loss(Layers[-1],Actual)
        else:
            raise NotImplementedError("Unkown Loss Function")
        Adjustments = DeepLearning.BackProp(Actual,Nu,Delta)
        DeepLearning.Correct(Adjustments)
        LossList = cp.append(LossList,Loss)
        TotalOut[counter*batch_len:(counter+1)*batch_len] = cp.argmax(Layers[-1],axis=0)
        print(f"{counter*batch_len} records done")
    Equality = OutputDataset == TotalOut
    Correct = cp.sum(Equality)
    Train_Accuracy = Correct/len(TotalOut)*100
    print(f"{ep + 1} epochs done")
    LogChanges.append(copy.deepcopy(Changes))
    if LossList[-2].item()>2 and LossList[-3].item()>2 and ep != 0:
        y = np.array(LossList.get()).flatten()
        x = np.arange(len(y))*batch_len
        plt.plot(x,y)
        plt.show()
        #raise RuntimeError("Loss is not going down")
    TestStart = time.perf_counter()
    Accuracy = Tester.TestAccuracy(DeepLearning, Parameter_Dict, ConvolutionalLayers, 
                        Layer_Lengths, MidLayerActivation, LastLayerActivation,
                        batch_len = 2*batch_len,section = Section)
    #Train_Accuracy = Tester.TestAccuracy(DeepLearning, Parameter_Dict, ConvolutionalLayers, 
    #                    Layer_Lengths, MidLayerActivation, LastLayerActivation,
    #                    batch_len = 2*batch_len,section = Section,
    #                    TestInputFilePath = r"D:\ConvolutionData\Train\InputData.npy" ,
    #                    TestOutputFilePath = r"D:\ConvolutionData\Train\OutputData.npy",)
    print(f"Train accuracy is {Train_Accuracy}%")
    TestEnd = time.perf_counter()
    TestTime = TestEnd - TestStart
    if TestTime > TestCutoff:
        Section *= TestCutoff/TestTime
    print(f"Test accuracy is {Accuracy}%")
    if(Accuracy>MaxAccuracy):
        MaxAccuracy = Accuracy
        FinalParams = copy.deepcopy(DeepLearning.Parameter_Dict) 
y = np.array(LossList.get()).flatten()
x = np.arange(len(y))*batch_len
#plt.yscale('log')
plt.plot(x,y)
plt.show()
try: os.mkdir(OutputFolderPath)
except: pass
finally:
        PDict = FinalParams
        NewDict = {"FF":{},"Convolution":{},"Pooling":{}}
        for Layer in range(len(Layer_Lengths[1:])):
            NewDict["FF"][Layer+1] = [PDict["w" + str(Layer+1)],PDict["b" + str(Layer+1)]]
        for item in ConvolutionalLayers:
            NewDict["Convolution"][item[0]] = PDict[item[0]]
            NewDict["Convolution"][item[0]+"Bias"] = PDict[item[0]+"Bias"]
            NewDict["Convolution"][item[0]+"Padding"] = int(item[3])
            NewDict["Pooling"][item[0]] = [int(item[4]),item[5],int(item[6])]
        NewDict["Data"] = [MidLayerActivation,LastLayerActivation]
        Path = OutputFolderPath + "Params.bin"
        with open(Path,"wb") as WriteFile:
            pkl.dump(NewDict,WriteFile)
