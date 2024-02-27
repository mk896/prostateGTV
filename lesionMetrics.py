#######################################################################################################
# Script: 2/26/24
#      Characterizes DIL lesions as TP, FP, or FN, based on PICAI-eval framework
#           Code adapted from https://github.com/DIAGNijmegen/picai_eval/blob/main/src/picai_eval/eval.py
#      Extracts DIL characteristics on a per-patient basis. 
#      Extracts DIL characteristics on a per-lesion basis. 
# Inputs:
#      imgDir: Base directory with ./imagesTr, ./labelsTr subdirectories
#      outSegDir: Directory of nnUNet predictions.
# Outputs:
#      Writes extracted per-patient DIL characteristics to metricsTest.csv
#      Writes extracted per-lesion DIL characteristics to metricsLesionTest.csv
#######################################################################################################


import matplotlib.pyplot as plt
import os
import numpy as np
from scipy import ndimage
import SimpleITK as sitk
import pandas as pd
from scipy.optimize import linear_sum_assignment
import surface_distance # Imported from https://github.com/google-deepmind/surface-distance

#Sitk filters
readerFilter = sitk.ImageFileReader()
bdFilter = sitk.BinaryDilateImageFilter()
ccFilter = sitk.ConnectedComponentImageFilter()
ccFilter.SetFullyConnected(True)
hdFilter = sitk.HausdorffDistanceImageFilter()

# Adapted from https://github.com/kkiser1/Autosegmentation-Spatial-Similarity-Metrics/blob/master/APIs/segmentationMetrics_APIs.py
def AddedPathLength(auto, gt):
  '''
  Returns the added path length, in pixels

  Steps:
  1. Find pixels at the edge of the mask for both auto and gt
  2. Count # pixels on the edge of gt that are not in the edge of auto
  '''

  # Check if auto and gt have same dimensions. If not, then raise a ValueError
  if auto.shape != gt.shape:
    raise ValueError('Shape of auto and gt must be identical!')

  # edge_auto has the pixels which are at the edge of the automated segmentation result
  edge_auto = getEdgeOfMask(auto)
  # edge_gt has the pixels which are at the edge of the ground truth segmentation
  edge_gt = getEdgeOfMask(gt)

  # Count # pixels on the edge of gt that are on not in the edge of auto
  apl = (edge_gt > edge_auto).astype(int).sum()

  return apl

# Adapted from https://github.com/kkiser1/Autosegmentation-Spatial-Similarity-Metrics/blob/master/APIs/segmentationMetrics_APIs.py
def getEdgeOfMask(mask):
  '''
  Computes and returns edge of a segmentation mask
  '''
  # edge has the pixels which are at the edge of the mask
  edge = np.zeros_like(mask)

  # mask_pixels has the pixels which are inside the mask of the automated segmentation result
  mask_pixels = np.where(mask > 0)

  for idx in range(0, mask_pixels[0].size):

    x = mask_pixels[0][idx]
    y = mask_pixels[1][idx]
    z = mask_pixels[2][idx]

    # Count # pixels in 3x3 neighborhood that are in the mask
    # If sum < 27, then (x, y, z) is on the edge of the mask
    if mask[x - 1:x + 2, y - 1:y + 2, z - 1:z + 2].sum() < 27:
      edge[x, y, z] = 1

  return edge

# Calculate properties based on lesion basis.
def calcDILPropArr(matchTestarr, ccTestDILsitk, imgsitk):
  tpvolarr = [] #Volume
  tpMeanContarr = [] #Total contrast-weighted by volume.
  tpTZarr = []

  lentpLabel = len(matchTestarr)
  numTP = lentpLabel
  lesionsextentarr = []
  for x in range(0, lentpLabel):
    print('  Label (Test): ', matchTestarr[x]) #Do not add 1, because matchTestarr first label at array 0.
    maskTestDILsitk = (ccTestDILsitk == (matchTestarr[x]+ 1)) #Add 1, because ccTestDILsitk first label starts at 1.
    tpdilprop = maskProperties(imgsitk, maskTestDILsitk)
    print('  Contrast: ', tpdilprop.calcCon())
    print('  Volume: ', tpdilprop.calcVol())
    maskTestDIL = sitk.GetArrayFromImage(maskTestDILsitk)

    tpvolarr.append(tpdilprop.calcVol())
    tpMeanContarr.append((tpdilprop.calcCon())) 

  return(numTP, tpvolarr, tpMeanContarr)

class maskProperties:
  def __init__(self, imgsitk, masksitk):
    self.vol = 0
    self.mean = -1
    self.conRatio = -1
    self._imgsitk = imgsitk
    self._masksitk = masksitk
    self.img = sitk.GetArrayFromImage(imgsitk)
    self.mask = sitk.GetArrayFromImage(masksitk)

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

  def calcVol(self):
    self.vol = self.mask.sum()*self._masksitk.GetSpacing()[0]*self._masksitk.GetSpacing()[1]*self._masksitk.GetSpacing()[2]
    return(self.vol)

  def calcCOMpix(self):
    COMpix = ndimage.measurements.center_of_mass(self.mask)
    return(COMpix)

  def calcCon(self):
    sdmFilter = sitk.SignedMaurerDistanceMapImageFilter()
    sdmFilter.SetUseImageSpacing(True)  # Utilize imaging spacing to calculate Euclidean distance
    sdmFilter.SquaredDistanceOff()  # No signed square distance needed.
    binaryThresholdImageFilter = sitk.BinaryThresholdImageFilter()

    masksdmsitk = sdmFilter.Execute(self._masksitk)
    masksdm = sitk.GetArrayFromImage(masksdmsitk)

    binaryThresholdImageFilter.SetLowerThreshold(0.1)
    binaryThresholdImageFilter.SetUpperThreshold(2)

    ringsitk = binaryThresholdImageFilter.Execute(masksdmsitk)
    ring = sitk.GetArrayFromImage(ringsitk)
    img = sitk.GetArrayFromImage(self._imgsitk)

    self.conRatio = np.mean(img[self.mask == 1])/np.mean(img[ring == 1])
    return (self.conRatio)

  def calcConInPros(self, prossitk):
    sdmFilter = sitk.SignedMaurerDistanceMapImageFilter()
    sdmFilter.SetUseImageSpacing(True)  # Utilize imaging spacing to calculate Euclidean distance
    sdmFilter.SquaredDistanceOff()  # No signed square distance needed.
    binaryThresholdImageFilter = sitk.BinaryThresholdImageFilter()

    masksdmsitk = sdmFilter.Execute(self._masksitk)
    masksdm = sitk.GetArrayFromImage(masksdmsitk)

    binaryThresholdImageFilter.SetLowerThreshold(0.1)
    binaryThresholdImageFilter.SetUpperThreshold(2)

    ringsitk = binaryThresholdImageFilter.Execute(masksdmsitk)
    ringprossitk = ringsitk*prossitk
    ringpros = sitk.GetArrayFromImage(ringprossitk)
    img = sitk.GetArrayFromImage(self._imgsitk)

    self.conRatio = np.mean(img[self.mask == 1])/np.mean(img[ringpros == 1])
    return (self.conRatio)

def calcDiceNP(y_det, y_true):
  #Calculate diceCoefficient
  epsilon = 1e-8
  dsc_num = np.sum(y_det[y_true == 1]) * 2.0
  dsc_denom = np.sum(y_det) + np.sum(y_true)
  return float((dsc_num + epsilon) / (dsc_denom + epsilon))

def readImageFcn(name):
  readerFilter = sitk.ImageFileReader()
  readerFilter.SetFileName(name)
  imgsitk = readerFilter.Execute()
  img = sitk.GetArrayFromImage(imgsitk)
  return(imgsitk, img)

imgDir = '..'
outSegDir = '..//predict'

# list to store files
patients = [f for f in os.listdir(outSegDir) if f.endswith('.nii.gz')]
patients.sort()

#Create arrays for arrays.
numTestDILarr = []; numRefDilarr = [];
numTParr = []; numFParr = []; numFNarr = []
testDILVolarr = []; refDILVolarr = []
tpvolarr = []; fpvolarr = []; fnvolarr = []
tpMeanContarr = []; fpMeanContarr = []; fnMeanContarr = []
ptTParr = []

lesionTP = []; lesionID = []; lesionRef = []; lesionOverlap = []; lesionVol = []
lesionContrast = []; lesionTZ = []; lesionHausdorff = []; lesionAPL = []; lesionSurfDice = []

for i in patients:
  print('============')
  # Open ADC img
  adcName = os.path.join(imgDir, 'imagesTr', i.split('.')[0] + '_0001.nii.gz')
  print('ADC name: ', adcName)
  adcsitk, img = readImageFcn(adcName)

########## Test DIL
  # Open test DIL img
  testDILname = os.path.join(outSegDir, 'DILOnly', i)
  print('Test DIL name: ', testDILname)
  testDILsitk, testDIL = readImageFcn(testDILname)
  pixelSpacing = testDILsitk.GetSpacing()

  # Calculate test DIL volume
  testDILmasksitk = maskProperties(adcsitk, testDILsitk)
  testDILVol = testDILmasksitk.calcVol() # DIL volume
  print('Total Test DIL volume: ', testDILVol)
  testDILVolarr.append(testDILVol)

  # Label connected components of test DIL.
  labeled_test, numTestDIL = ndimage.label(testDIL, structure = np.ones((3, 3, 3)))
  numTestDILarr.append(numTestDIL) # Array of test DIL components.
  test_lesion_ids = np.arange(numTestDIL)
  
  ccTestDILsitk = sitk.GetImageFromArray(labeled_test)
  ccTestDILsitk.SetSpacing(adcsitk.GetSpacing()); ccTestDILsitk.SetDirection(adcsitk.GetDirection()); ccTestDILsitk.SetOrigin(adcsitk.GetOrigin())

########## Reference DIL
  # Open reference DIL image.
  refDILName = os.path.join(outSegDir,  'RefDILOnly', i)
  print('Loading: ', refDILName)
  refDILsitk, refDIL = readImageFcn(refDILName)

  # Calculate reference DIL volume
  refDILMasksitk = maskProperties(adcsitk, refDILsitk)
  refDILVol = refDILMasksitk.calcVol()
  print('Total Ref DIL volume: ', refDILVol)
  refDILVolarr.append(refDILVol)

  # Label connected components of ref DIL.
  # Code adapted from PICAI_eval to mirror lesions selected for TP, FP, and FN. https://github.com/DIAGNijmegen/picai_eval
  labeled_ref, numRefDIL = ndimage.label(refDIL, structure=np.ones((3, 3, 3)))
  numRefDilarr.append(numRefDIL)
  ref_lesion_ids = np.arange(numRefDIL)

  ccRefDILsitk = sitk.GetImageFromArray(labeled_ref)
  ccRefDILsitk.SetSpacing(adcsitk.GetSpacing()); ccRefDILsitk.SetDirection(adcsitk.GetDirection()); ccRefDILsitk.SetOrigin(adcsitk.GetOrigin())

  overlap_matrix = np.zeros((numRefDIL, numTestDIL))
  hd_matrix = np.zeros((numRefDIL, numTestDIL))
  apl_matrix = np.zeros((numRefDIL, numTestDIL))
  surfDice_matrix = np.zeros((numRefDIL, numTestDIL))
  min_overlap = 0.1

  for ref_lesion_ct in ref_lesion_ids:
    # for each lesion in ground-truth (GT) label
    ref_cc_mask = (labeled_ref == (1 + ref_lesion_ct))

    # calculate overlap between each lesion candidate and the current GT lesion
    for test_lesion_ct in test_lesion_ids:
      # calculate overlap between lesion candidate and GT mask
      test_cc_mask = (labeled_test == (1 + test_lesion_ct))
      overlap_score = calcDiceNP(test_cc_mask, ref_cc_mask)

      # store overlap: Based on PICAI_eval
      overlap_matrix[ref_lesion_ct, test_lesion_ct] = overlap_score

      # APL: https://github.com/kkiser1/Autosegmentation-Spatial-Similarity-Metrics/blob/master/APIs/segmentationMetrics_APIs.py
      apl_matrix[ref_lesion_ct, test_lesion_ct] = AddedPathLength(test_cc_mask, ref_cc_mask)

      # Surface metrics: https://github.com/deepmind/surface-distance
      surface_distances = surface_distance.compute_surface_distances(test_cc_mask, ref_cc_mask, tuple(reversed(pixelSpacing)))
      hd_matrix[ref_lesion_ct, test_lesion_ct] = surface_distance.compute_robust_hausdorff(surface_distances, 100)
      surfDice_matrix[ref_lesion_ct, test_lesion_ct] = surface_distance.compute_surface_dice_at_tolerance(surface_distances, 1)
  print('overlap_matrix: ', overlap_matrix)

  # Adjust overlap matrix
  overlap_matrix[overlap_matrix < min_overlap] = 0  # remove indices where overlap is zero (don't match lesions with insufficient overlap)
  overlap_matrix[overlap_matrix > 0] += 1  # prioritize matching over the amount of overlap

  # match lesion candidates to ground truth lesion (for documentation on how this works, please see
  # https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.optimize.linear_sum_assignment.html)
  matched_ref_indices_assignment, matched_test_indices_assignment = linear_sum_assignment(overlap_matrix, maximize=True)
  match_matrix = (overlap_matrix[matched_ref_indices_assignment, matched_test_indices_assignment] > 0)
  
  matched_ref_lesions = matched_ref_indices_assignment[match_matrix]
  matched_test_lesions = matched_test_indices_assignment[match_matrix]

  # all lesion candidates that are matched are TPs
  print('TP')
  [numTP, tpvol, tpMeanCont] = calcDILPropArr(matched_test_lesions, ccTestDILsitk, adcsitk)
  print('NumTP: ', numTP)
  print('TPVol: ', tpvol)
  print('TPMeanCont: ', tpMeanCont)
  ct = 0
  for ref_lesion_ct, test_lesion_ct in zip(matched_ref_lesions, matched_test_lesions):
    overlap = overlap_matrix[ref_lesion_ct, test_lesion_ct]
    overlap -= 1  # return overlap to [0, 1] (1 added to overlap above)
    assert overlap > min_overlap, "Overlap must be greater than min_overlap!"

    hdval = hd_matrix[ref_lesion_ct, test_lesion_ct]
    aplval = apl_matrix[ref_lesion_ct, test_lesion_ct]
    surfDiceval = surfDice_matrix[ref_lesion_ct, test_lesion_ct]
    lesionID.append(i); lesionTP.append('TP'); lesionOverlap.append(overlap); lesionHausdorff.append(hdval); lesionAPL.append(aplval); lesionSurfDice.append(surfDiceval)
    lesionVol.append(tpvol[ct]); lesionContrast.append(tpMeanCont[ct]);
    ct = ct + 1

  # all ground truth lesions that are not matched are FNs
  print('FN')
  unmatched_ref_lesions = set(ref_lesion_ids) - set(matched_ref_lesions)
  [numFN, fnvol, fnMeanCont] = calcDILPropArr(np.array(tuple(unmatched_ref_lesions)), ccRefDILsitk, adcsitk)
  for ct in range(len(unmatched_ref_lesions)):
    lesionID.append(i); lesionTP.append('FN'); lesionOverlap.append(0); lesionHausdorff.append(0); lesionAPL.append(0); lesionSurfDice.append(0)
    lesionVol.append(fnvol[ct]); lesionContrast.append(fnMeanCont[ct]); 

  # all lesion candidates with insufficient overlap/not matched to a reference lesion are FPs
  allow_unmatched_candidates_with_minimal_overlap = True
  if allow_unmatched_candidates_with_minimal_overlap:
    candidates_sufficient_overlap = test_lesion_ids[(overlap_matrix > 0).any(axis=0)]
    unmatched_candidates = set(test_lesion_ids) - set(candidates_sufficient_overlap)
  else:
    unmatched_candidates = set(test_lesion_ids) - set(matched_test_lesions)

  print('FP')
  [numFP, fpvol, fpMeanCont] = calcDILPropArr(np.array(tuple(unmatched_candidates)), ccTestDILsitk, adcsitk)
  ct = 0
  for lesion_candidate_id in unmatched_candidates:
    lesionID.append(i); lesionTP.append('FP'); lesionOverlap.append(0); lesionHausdorff.append(0); lesionAPL.append(0); lesionSurfDice.append(0)
    lesionVol.append(fpvol[ct]); lesionContrast.append(fpMeanCont[ct]); 
    ct = ct + 1

  numTParr.append(numTP)
  numFParr.append(numFP)
  numFNarr.append(numFN)

  tpvolarr.append(np.sum(tpvol))
  fpvolarr.append(np.sum(fpvol))
  fnvolarr.append(np.sum(fnvol))

  # Case-level.
  if ((numRefDIL > 0) & (numTP > 0)): ptTP = 'TP'
  if ((numRefDIL > 0) & (numTP == 0)): ptTP = 'FN'
  if ((numRefDIL == 0) & (numTestDIL > 0)): ptTP = 'FP'
  if ((numRefDIL == 0) & (numTestDIL == 0)): ptTP = 'TN'

  ptTParr.append(ptTP)

# CSV: Metric patient ID
df = pd.DataFrame({"Patients": patients,
                   "numTestDIL": numTestDILarr, "numRefDIL": numRefDilarr,
                   "numTP": numTParr, "numFP" : numFParr, "numFN": numFNarr,
                   "testvol": testDILVolarr, "refvol": refDILVolarr,
                   "tpvol": tpvolarr, "fpvol": fpvolarr, "fnvol": fnvolarr,
                   'ptTParr': ptTParr})
metricsTestName = os.path.join('metricsTest.csv')
df.to_csv(metricsTestName, index=False)
print('Wrote: ', metricsTestName)

# CSV: Metric lesion ID
df2 = pd.DataFrame({"ID": lesionID, "lesionTP": lesionTP, "Overlap": lesionOverlap, "Hausdorff": lesionHausdorff, "APL": lesionAPL, "surfDice": lesionSurfDice,
                    "Vol": lesionVol, "Contrast": lesionContrast})
metricsLesionTestName = os.path.join('metricsLesionTest.csv')
df2.to_csv(metricsLesionTestName, index=False)
print('Wrote: ', metricsLesionTestName)

