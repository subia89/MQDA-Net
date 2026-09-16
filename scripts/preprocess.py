#!/usr/bin/env python
"""Pre-processing for raw multi-parametric MRI (paper Fig. 1, stage 1).

N4 bias-field correction, resampling to 1 mm isotropic and optional affine
registration of every sequence to an MNI template (needed for NiftiAtlas
localisation). BraTS data are already skull-stripped and co-registered; for
other data run a skull stripper such as HD-BET first.

python scripts/preprocess.py --in raw/case01 --out data/case01 \
    [--template MNI152_T1_1mm_brain.nii.gz] [--reference-suffix t1c]
Requires SimpleITK (pip install SimpleITK).
"""
import argparse
import glob
import os

import SimpleITK as sitk


def n4(img):
    img = sitk.Cast(img, sitk.sitkFloat32)
    mask = sitk.OtsuThreshold(img, 0, 1, 200)
    shrink = sitk.Shrink(img, [2] * img.GetDimension())
    mshrink = sitk.Shrink(mask, [2] * img.GetDimension())
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([50, 50, 30])
    corrector.Execute(shrink, mshrink)
    log_bias = corrector.GetLogBiasFieldAsImage(img)
    return img / sitk.Exp(log_bias)


def resample(img, spacing=(1.0, 1.0, 1.0), interp=sitk.sitkLinear):
    size = [int(round(sz * sp / ns)) for sz, sp, ns in zip(img.GetSize(), img.GetSpacing(), spacing)]
    return sitk.Resample(img, size, sitk.Transform(), interp, img.GetOrigin(), spacing,
                         img.GetDirection(), 0, img.GetPixelID())


def register(moving, fixed):
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(50)
    reg.SetOptimizerAsRegularStepGradientDescent(2.0, 1e-4, 200)
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([2, 1, 0])
    reg.SetInterpolator(sitk.sitkLinear)
    init = sitk.CenteredTransformInitializer(fixed, moving, sitk.AffineTransform(3),
                                             sitk.CenteredTransformInitializerFilter.GEOMETRY)
    reg.SetInitialTransform(init, inPlace=False)
    return reg.Execute(sitk.Cast(fixed, sitk.sitkFloat32), sitk.Cast(moving, sitk.sitkFloat32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--template", default=None)
    ap.add_argument("--reference-suffix", default="t1c")
    ap.add_argument("--no-n4", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.inp, "*.nii*")))
    ref = next((f for f in files if args.reference_suffix in os.path.basename(f)), files[0])
    tx = None
    if args.template:
        tx = register(resample(sitk.ReadImage(ref)), sitk.ReadImage(args.template))
    for f in files:
        is_seg = "seg" in os.path.basename(f)
        img = sitk.ReadImage(f)
        if not is_seg and not args.no_n4:
            img = n4(img)
        interp = sitk.sitkNearestNeighbor if is_seg else sitk.sitkLinear
        img = resample(img, interp=interp)
        if tx is not None:
            fixed = sitk.ReadImage(args.template)
            img = sitk.Resample(img, fixed, tx, interp, 0)
        sitk.WriteImage(img, os.path.join(args.out, os.path.basename(f)))
        print("wrote", os.path.basename(f))


if __name__ == "__main__":
    main()
