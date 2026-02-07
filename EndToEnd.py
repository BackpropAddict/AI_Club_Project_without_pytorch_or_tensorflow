# -*- coding: utf-8 -*-
"""
Created on Sat Feb  7 23:53:56 2026

@author: Dhruv P Trial
"""
import librosa 
import cupy as cp
from cupy.lib.stride_tricks import as_strided
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

def wav_to_mel_spectrogram(
    wav_path,
    sr=22050,
    n_mels=128,
    n_fft=1024,
    hop_length=512,
    fmin=0.0,
    fmax=None,
    noise_db_threshold=-20.0,
    target_frames=130,
    normalize=True,
):
    # 1) Compute required samples ONLY
    samples_needed = (target_frames - 1) * hop_length + n_fft
    duration = samples_needed / sr

    # 2) Fast audio load (no overkill resampling)
    y, native_sr = librosa.load(
    wav_path,
    sr=None,   # <-- do NOT resample
    mono=True,
    duration=duration
)

    # Pad waveform if too short
    if y.size < samples_needed:
        y = np.pad(y, (0, samples_needed - y.size))

    # 3) Mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=2.0
    )

    # 4) dB + noise gate
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = np.maximum(mel_db, noise_db_threshold)

    # 5) Crop to exact frame count (guaranteed)
    mel_db = mel_db[:, :target_frames]

    # 6) Normalize
    if normalize:
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)

    return mel_db.astype(np.float32)

class DeepLearning():
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
    def Convolve4D(self,Input,Kernels,PadWidth,Type = "Correlate",Bias = cp.zeros(1)):
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
            raise IndexError("Kernel is smaller than Input")
        shape = (Input.shape[0],Mid1,Mid2,Input.shape[1],KSize,KSize)
        Wind = as_strided(Padded,shape=shape,strides=(Bs,Hs,Vs,Cs,Hs,Vs))
        Output = cp.einsum('b h w c p q, o c p q -> b o h w', Wind, Kernels)
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
            mask = cp.ones(Window.shape)/(Size*Size)
        return AlephNol,mask
    def LeakyRelu(self,Matrix):
        Output = cp.fmax(Matrix,0)
        Output += cp.fmin(Matrix,0)*0.01
        return Output
    def Forward(self,Input,Weights,Bias,Activation):
        z = cp.dot(Weights,Input) + Bias
        if Activation == "Relu": return self.ReLU(z)
        elif Activation == "Sigmoid": return self.Sigmoid(z)
        elif Activation == "Softmax": return self.Softmax(z)
        elif Activation == "Leaky": 
            z = self.BatchNorm(z)
            return self.LeakyRelu(z)
        elif Activation == "None": return z
        else: raise KeyError("Activation was incorrect")
    def OneHots(self,indices,Matrix_Size):
        HotsIntermediate = cp.zeros((Matrix_Size,len(indices)))
        a = cp.arange(len(indices))
        HotsIntermediate[indices.flatten(),a.flatten()] = 1
        return HotsIntermediate
    def BatchNorm(self,Matrix):
        mean = cp.mean(Matrix,keepdims = True)
        std = cp.std(Matrix,keepdims = True)
        return(Matrix - mean)/(std + 1e-8)
    def ConfusionMatrix(self,y_true, y_pred, num_classes=8):
        y_true = y_true.astype(cp.int32)
        y_pred = y_pred.astype(cp.int32)
        cm = cp.zeros((num_classes, num_classes), dtype=cp.int32)
        for i in range(y_true.shape[0]):
            cm[y_true[i], y_pred[i]] += 1
        return cm
    def plot_confusion_matrix(self,cm, class_names=None, normalize=False):
        if normalize:
            cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(cm)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        if class_names is not None:
            ax.set_xticks(np.arange(len(class_names)))
            ax.set_yticks(np.arange(len(class_names)))
            ax.set_xticklabels(class_names, rotation=45, ha="right")
            ax.set_yticklabels(class_names)
        # Write numbers inside cells
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                value = cm[i, j]
                text = f"{value:.2f}" if normalize else f"{int(value)}"
                ax.text(j, i, text,
                        ha="center", va="center",
                        color="white" if value > cm.max() * 0.6 else "black")
        title = "Normalized Confusion Matrix" if normalize else "Confusion Matrix"
        ax.set_title(title)
        plt.tight_layout()
        plt.show()
    def MacroF1(self,cm):
        cm = cm.astype(cp.float32)
        TP = cp.diag(cm)
        FP = cm.sum(axis=0) - TP
        FN = cm.sum(axis=1) - TP
        precision = cp.where(TP + FP > 0,TP / (TP + FP),0.0)
        recall = cp.where(TP + FN > 0,TP / (TP + FN),0.0)
        f1 = cp.where(precision + recall > 0,2 * precision * recall / (precision + recall),0.0)
        return f1.mean(), f1, precision, recall
"""                     Editable Values                   """
ParameterFilePath = r"C:\Users\User\OneDrive\Desktop\256Layer-LetterRecog\49%Accuracy.bin"
WavPath = r"C:\Users\User\OneDrive\Documents\Arduino\libraries\Adafruit_LED_Backpack_Library\examples\wavface\wavs\ihunger.wav"
batch_len  = 32 #int(input("Enter the Batch Length: "))
"""                     Editable Values                   """
InpDataset = wav_to_mel_spectrogram(WavPath)[None,None,:,:]
with open(ParameterFilePath,"rb") as ReadFile:
    Parameter_Dict = pkl.load(ReadFile)
ConvolutionalDict = Parameter_Dict["Convolution"]
PoolDict = Parameter_Dict["Pooling"]
PKeys = PoolDict.keys()
FeedForward = Parameter_Dict["FF"]
FFKeys = list(FeedForward.keys())
MidLayerActivation = Parameter_Dict["Data"][0]
LastLayerActivation = Parameter_Dict["Data"][1]
DeepLearning = DeepLearning()
Input = cp.array(InpDataset,dtype=cp.float32)
Current_Layer = cp.array([])
PooledLayers = [Input]
Layers = [Input]
ConvChecker = False
#Convolutional Layers
for ind,key in enumerate(PKeys):
    ConvChecker = True
    Kernel = ConvolutionalDict[key]
    Biases = ConvolutionalDict[key + "Bias"]
    Padding = ConvolutionalDict[key+"Padding"]
    Convolved = DeepLearning.Convolve4D(PooledLayers[-1], Kernel,Padding,Bias=Biases)
    Inter = DeepLearning.BatchNorm(Convolved)
    Convolved = DeepLearning.LeakyRelu(Inter)
    DbConvolved = Convolved.get()
    KSize = (Kernel.shape[0]*Kernel.shape[1],Kernel.shape[2],Kernel.shape[3])
    DbKernel = Kernel.reshape(KSize).get()
    PSize = PoolDict[key][0]
    PType = PoolDict[key][1]
    PStride = PoolDict[key][2]
    Pooled,mask = DeepLearning.Pool(Convolved,PSize,Stride = PStride,PoolType=PType)
    PooledLayers.append(Pooled)
    Layers[0] = Pooled
if ConvChecker:
    Layers[0] = Layers[0].reshape((Layers[0].shape[0],-1)).transpose()
#Feedforward Layers
Buffer = Layers[0]
for Key in FFKeys[:-1]:    
    Middle_Layer = DeepLearning.Forward(Buffer, FeedForward[Key][0], FeedForward[Key][1], MidLayerActivation)
    Buffer = Middle_Layer #DeepLearning.BatchNorm(Middle_Layer)
Output_Layer = DeepLearning.Forward(Buffer, FeedForward[FFKeys[-1]][0], FeedForward[FFKeys[-1]][1], LastLayerActivation)
Output = cp.argmax(Output_Layer,axis=0,keepdims=True)
print(f"The predicted class was class {int(cp.argmax(Output))}")