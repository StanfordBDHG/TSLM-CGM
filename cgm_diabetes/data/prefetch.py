from cgm_diabetes.data.CGMDiabetesDataset import CGMDiabetesDataset
import sys

split = sys.argv[1] if len(sys.argv) > 1 else None
CGMDiabetesDataset.prefetch(split)