#######################################################################################################
# Script: 2/26/24
#      Extracts test DIL from predicted label map
#         Removes small lesions (<30 mm3)
#         Removes far lesions (>3 mm from prostate)
#         Writes to ./DILOnly subdirectory
#      Extracts reference DIL from reference label map
#         Writes to ./RefDILOnly subdirectory
# Inputs:
#      imgDir: Base directory with ./imagesTr, ./labelsTr subdirectories
#      outSegDir: Directory of nnUNet predictions.
# Outputs:
#      Writes extracted test DIL to ./DILOnly subdirectory
#      Writes extracted ref DIL to to ./RefDILOnly subdirectory
#      Writes lesionElim.csv file with 3 columns: ID, number of eliminated small lesions, number of eliminated far lesions. 
######################################################################################################

import matplotlib.pyplot as plt
import numpy as np
import math
from scipy import ndimage
import SimpleITK as sitk
import sys
import os, subprocess, shutil
import pandas as pd

class maskProperties:
  def __init__(self, imgsitk, masksitk):
    self.vol = 0
    self.mean = -1
    self.conRatio = -1
    self._imgsitk = imgsitk
    self._masksitk = masksitk
    self.img = sitk.GetArrayFromImage(imgsitk)
    self.mask = sitk.GetArrayFromImage(masksitk)
    self.masksdmsitk = -1
    
  def getMasksitk(self):
    return(self._masksitk)

  def getVol(self):
    return(self.vol)

  def getMean(self):
    return(self.mean)

  def getConRatio(self):
    return(self.conRatio)

  def calcMean(self):
    self.mean = np.mean(self.img[self.mask == 1])
    return(self.mean)

  def calcMax(self):
    self.mean = np.max(self.img[self.mask == 1])
    return(self.mean)

  def calcVol(self):
    self.vol = self.mask.sum()*self._masksitk.GetSpacing()[0]*self._masksitk.GetSpacing()[1]*self._masksitk.GetSpacing()[2]
    return(self.vol)

  def calcCOMpix(self):
    COMpix = ndimage.measurements.center_of_mass(self.mask)
    return(COMpix)

  def calcSDM(self):
    sdmFilter = sitk.SignedMaurerDistanceMapImageFilter()
    sdmFilter.SetUseImageSpacing(True)  # Utilize imaging spacing to calculate Euclidean distance
    sdmFilter.SquaredDistanceOff()  # No signed square distance needed.
    masksdmsitk = sdmFilter.Execute(self._masksitk)
    return(masksdmsitk)

  def getSDM(self):
    if (self.masksdmsitk == -1):
      self.masksdmsitk = self.calcSDM()
    return(self.masksdmsitk)

  def calcCon(self):
    masksdmsitk = self.getSDM()
    masksdm = sitk.GetArrayFromImage(masksdmsitk)

    binaryThresholdImageFilter = sitk.BinaryThresholdImageFilter()
    binaryThresholdImageFilter.SetLowerThreshold(0.1)
    binaryThresholdImageFilter.SetUpperThreshold(2)

    ringsitk = binaryThresholdImageFilter.Execute(masksdmsitk)
    ring = sitk.GetArrayFromImage(ringsitk)
    img = sitk.GetArrayFromImage(self._imgsitk)

    self.conRatio = np.mean(img[self.mask == 1])/np.mean(img[ring == 1])
    return (self.conRatio)

  def calcConInPros(self, prossitk):
    masksdmsitk = self.getSDM()
    masksdm = sitk.GetArrayFromImage(masksdmsitk)

    binaryThresholdImageFilter = sitk.BinaryThresholdImageFilter()
    binaryThresholdImageFilter.SetLowerThreshold(0.1)
    binaryThresholdImageFilter.SetUpperThreshold(2)

    ringsitk = binaryThresholdImageFilter.Execute(masksdmsitk)
    ringprossitk = ringsitk*prossitk
    ringpros = sitk.GetArrayFromImage(ringprossitk)
    img = sitk.GetArrayFromImage(self._imgsitk)

    self.conRatio = np.mean(img[self.mask == 1])/np.mean(img[ringpros == 1])
    return (self.conRatio)

def readImageFcn(name):
  readerFilter = sitk.ImageFileReader()
  readerFilter.SetFileName(name)
  imgsitk = readerFilter.Execute()
  img = sitk.GetArrayFromImage(imgsitk)
  return(imgsitk, img)

readerFilter = sitk.ImageFileReader()
writerFilter = sitk.ImageFileWriter()
labelProsFilter = sitk.LabelShapeStatisticsImageFilter()
ccFilter = sitk.ConnectedComponentImageFilter()
ccFilter.SetFullyConnected(False)
sdmFilter = sitk.SignedMaurerDistanceMapImageFilter()
sdmFilter.SetUseImageSpacing(True)  
sdmFilter.SquaredDistanceOff()  

imgDir = ''
outSegDir = 'predict'

# list to store files
patients = [f for f in os.listdir(outSegDir) if f.endswith('.nii.gz')]

DILOnlyDir = os.path.join(outSegDir, 'DILOnly')
if (not(os.path.isdir(DILOnlyDir))):
  os.mkdir(DILOnlyDir)

RefDILOnlyDir = os.path.join(outSegDir, 'RefDILOnly')
if (not(os.path.isdir(RefDILOnlyDir))):
  os.mkdir(RefDILOnlyDir)

elimDILSmall = []
elimDILFar = []

for i in patients:
  print('============')
  adcName = os.path.join(imgDir, 'imagesTr', i.split('.')[0] + '_0001.nii.gz')
  adcsitk, adc = readImageFcn(adcName)

  # Open predicted mask. Extract out test DIL. Remove small components and components far from the prostate. 
  testMaskFullName = os.path.join(outSegDir, i)
  print('Loading: ', testMaskFullName)
  testMasksitk, testMask = readImageFcn(testMaskFullName)

  testTZsitk = testMasksitk == 1
  testPZsitk = testMasksitk == 2
  testDILsitk = testMasksitk == 3

  testProssitk = testTZsitk + testPZsitk + testDILsitk #Add all 3 together to get entire prostate. Max value 1. 
  testDIL = sitk.GetArrayFromImage(testDILsitk)

  #Check if there are two separate prostate components. If so, then keep the larger one. 
  ccTestProssitk = ccFilter.Execute(testProssitk)
  ccTestProssitkMaxLabel = np.max(ccTestProssitk)

  labelProsFilter.Execute(ccTestProssitk)
  ccTestProsVolarr = np.zeros(ccTestProssitkMaxLabel)

  for lct in range(1, ccTestProssitkMaxLabel + 1):
    ccTestProsVolarr[lct - 1] = labelProsFilter.GetPhysicalSize(lct)

  if (ccTestProssitkMaxLabel > 1):
    keepind = (np.where(ccTestProsVolarr == max( ccTestProsVolarr)))[0]
    testProssitk = (ccTestProssitk == (keepind + 1))

  #Extract pros array. 
  testPros = sitk.GetArrayFromImage(testProssitk)

  # Connected component labeling for DIL
  ccTestDILsitk = ccFilter.Execute(testDILsitk)
  ccTestDIL = sitk.GetArrayFromImage(ccTestDILsitk)
  ccTestDILMaxLabel = np.max(ccTestDIL)

  elimDILSmallVal = 0
  elimDILFarVal = 0

  for j in range(0, ccTestDILMaxLabel):
    ccTestDILLabelsitk = (ccTestDILsitk == (j+1))
    ccTestDILLabel = sitk.GetArrayFromImage(ccTestDILLabelsitk)
    jcoord = np.where(ccTestDILLabel == 1)

    ccTestDILLabelADCMask = maskProperties(adcsitk, ccTestDILLabelsitk)
    ccTestDILLabelVol = ccTestDILLabelADCMask.calcVol()
 
    print('Lesion: ', (j + 1), '  Volume [mm3]: ', ccTestDILLabelVol)

    # Set small pixels to zero
    if (ccTestDILLabelVol<30):
      print('---Setting these pixels to zero since <30 mm3')
      testDIL[jcoord[0], jcoord[1], jcoord[2]] = 0
      elimDILSmallVal = elimDILSmallVal + 1
   
    # Multiply DIL SDM (distance map) with prostate mask, to determine minimum distance between DIL and prostate mask. 
    ccTestDILLabelSDMsitk = ccTestDILLabelADCMask.getSDM()
    testProsDILSDMsitk = ccTestDILLabelSDMsitk*sitk.Cast(testProssitk, sitk.sitkFloat32)
    testProsDILSDM = sitk.GetArrayFromImage(testProsDILSDMsitk)
    testProsDILSDMMinDist = np.min(testProsDILSDM[testPros == 1])
    print('    Distance to prostate: ', testProsDILSDMMinDist) 

    # Set pixels > 3 mm from prostate to zero. 
    if(testProsDILSDMMinDist>3):
      print('   +++Setting these pixels to zero since >3 mm from prostate')
      testDIL[jcoord[0], jcoord[1], jcoord[2]] = 0
      elimDILFarVal = elimDILFarVal + 1

  elimDILSmall.append(elimDILSmallVal)
  elimDILFar.append(elimDILFarVal)

  # Make testDIL into sitk image object. 
  newTestDILsitk = sitk.GetImageFromArray(testDIL)
  newTestDILsitk.SetSpacing(testDILsitk.GetSpacing())
  newTestDILsitk.SetOrigin(testDILsitk.GetOrigin())

  newDILname = os.path.join(DILOnlyDir, i)
  writerFilter.SetFileName(newDILname)
  writerFilter.Execute(newTestDILsitk)
  print('Wrote: ', newDILname)

  ########## Process reference DIL. 
  refFullName = os.path.join(imgDir, 'labelsTr', i)
  print('Loading: ', refFullName)

  readerFilter.SetFileName(refFullName)
  refMasksitk = readerFilter.Execute()

  refProssitk = refMasksitk == 1
  refPZsitk = refMasksitk == 2
  refDILsitk = refMasksitk == 3

  newRefDILname = os.path.join(RefDILOnlyDir, i)
  writerFilter.SetFileName(newRefDILname)
  writerFilter.Execute(refDILsitk)
  print('Wrote: ', newRefDILname)

print('elimDILSmall: ', elimDILSmall)
print('elimDILFar: ', elimDILFar)

df = pd.DataFrame({"Patients": patients, "elimDILSmall": elimDILSmall, "elimDILFar": elimDILFar})
df.to_csv('lesionElim.csv')
print('Wrote: lesionElim.csv')

