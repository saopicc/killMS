# killMS

### A direction-dependent radio interferometric calibration package

Copyright (C) 2013-2021  Cyril Tasse, l'Observatoire de Paris,
SKA South Africa, Rhodes University

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.



# What's this package about?

The Direction Dependent calibration problem (also known as third generation calibration) naturally arises in the Radio Interferometry Measurement Equation (RIME, Hamaker et al. 1994), but only became overwhelmingly problematic with the construction of the SKA precursors and pathfinders. Solving for the DDE calibration problems basically consists in inverting a number of non-linear equation, while the system is (i) very large and (ii) often subject to ill conditioning. killMS's name originates from the early LOFAR commissioning phases, when understanding the interferometric data in a Measurement Set was a real challenge.

# Wirtinger DDE calibration

killMS implements two very efficient algorithms for solving the Direction-Dependent calibration problem. The current status of the software and use cases (as well as connection with other softwares such as DDFacet) are summarised in

http://www.astron.nl/lowfrequencyobserving2017/Documents/Wednesday/LFO2017_Tasse.pdf

The two algorithms (CohJones and Kafka) are based on complex optimisation techniques. They use the properties of the complex ("Wirtinger") Jacobian to exploit algorithmic shortcuts. The basics of the application of the Wirtinger Jacobian and Hessians to the RIME, as well as the related implemented algorithms are described in

Tasse 2014: https://arxiv.org/abs/1410.8706

Smirnov & Tasse 2015: https://arxiv.org/abs/1502.06974

killMS also runs an extended Kalman filter that uses the Wirtinger (half) Jacobian (to be published, a similar approach is described in https://arxiv.org/abs/1403.6308)

Preliminary documentation is given below.


# Installation KillMS
To build from source:
```
virtualenv myvenv
source myvenv/bin/activate
(myvenv)$ pip install DDFacet # installs the latest DDF release from PyPI
(myvenv)$ pip install <path to checked out killMS>
```
If you want to run in development mode:
```
(myvenv)$ pip install -e <path to checked out killMS>
(myvenv)$ cd <path to checked out killMS>
(myvenv)$ python setup.py build  #this rebuilds the backend
```

The old build system is still in place - you can still do:
```
cd Predict
make
cd ../Array/Dot
make
cd ../../Gridder
make
```

# Main programs you'll need for DDE calibration and imaging

* kMS.py -> Does DDE calibration using the LM (CohJones) or the Kalman filter (KAFCA)
* DDF.py -> Applies DDE calibration in deconvolution
* MakeModel.py -> Clusters the sky, etc
* MakeMask.py -> To construct masks

# To get Documentation

Type

```
kMS.py -h
MakeModel.py -h
DDF.py -h
MakeMask.py -h
```

# Example of data reduction with killMS/DDFacet

in a file .txt (here mslist.txt), put the path to your MSs, for example

```
/data/tasse/BootesObs/L374583/L374583_SB244_uv.dppp.pre-cal_127080C79t_121MHz.pre-cal.ms
/data/tasse/BootesObs/L374583/L374583_SB254_uv.dppp.pre-cal_127080C79t_123MHz.pre-cal.ms
/data/tasse/BootesObs/L374583/L374583_SB264_uv.dppp.pre-cal_127080C79t_125MHz.pre-cal.ms
/data/tasse/BootesObs/L374583/L374583_SB274_uv.dppp.pre-cal_127080C79t_127MHz.pre-cal.ms
```

## Strategy

In the following, we do
* A direction independent image called "image_DI"
* We cluster the sky in 10 directions
* We solve for scalar Jones matrices in 10 directions, using the KAFCA solver, solution named testKAFCA
* We deconvolve using the direction dependent solutions, and create the "image_DD" corrected image

## Do DI image:

```
DDF.py --Output-Name=image_DI --Data-MS mslist.txt --Deconv-PeakFactor 0.001000 --Data-ColName DATA --Parallel-NCPU=40 --Image-Mode=Clean --Deconv-CycleFactor=0 --Deconv-MaxMajorIter=3 --Deconv-Mode SSD --Weight-Robust -0.15 --Image-NPix=10000 --CF-wmax 100000 --CF-Nw 100 --Output-Also onNeds --Image-Cell 3 --Facets-NFacets=11 --SSDClean-NEnlargeData 0 --Freq-NDegridBand 1 --Beam-NBand 1 --Deconv-RMSFactor=3.000000 --Data-Sort 1 --Cache-Dir=. --Freq-NBand=2 --Mask-Auto=1 --Mask-SigTh=5.00 --Cache-Reset 0 --SSDClean-MinSizeInitHMP=10
```

## Cluster the sky in 10 directions

```
MakeModel.py --BaseImageName image_DI --NCluster 10
```
-> creates a cluster nodes catalog: image_DI.npy.ClusterCat.npy 

## From the model, calibrate all ms:
```
kMS.py --MSName mslist.txt --SolverType KAFCA --PolMode Scalar --BaseImageName image_DI --dt 1 --NCPU 40 --OutSolsName testKAFCA --NChanSols 1 --InCol DATA --OutCol DATA --Weighting Natural --NodesFile image_DI.npy.ClusterCat.npy --MaxFacetSize 1.5
```
--> creates solution files inside each <MS>/killMS.testKAFCA.sols.npz

## The image taking the DDE into account:
```
DDF.py --Output-Name=image_DD --Data-MS mslist.txt --Deconv-PeakFactor 0.001000 --Data-ColName DATA --Parallel-NCPU=40 --Image-Mode=Clean --Deconv-CycleFactor=0 --Deconv-MaxMajorIter=3 --Deconv-Mode SSD --Weight-Robust -0.15 --Image-NPix=10000 --CF-wmax 100000 --CF-Nw 100 --Output-Also onNeds --Image-Cell 3 --Facets-NFacets=11 --SSDClean-NEnlargeData 0 --Freq-NDegridBand 1 --Beam-NBand 1 --Deconv-RMSFactor=3.000000 --Data-Sort 1 --Cache-Dir=. --Freq-NBand=2 --Mask-Auto=1 --Mask-SigTh=5.00 --Cache-Reset 0 --SSDClean-MinSizeInitHMP=10 --DDESolutions-DDSols testKAFCA --Predict-InitDicoModel image_DI.DicoModel --Facets-DiamMax 1.5 --Facets-DiamMin 0.1
```

