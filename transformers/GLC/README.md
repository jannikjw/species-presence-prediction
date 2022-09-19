# GeoLifeCLEF 2022

![Example of patch data](https://raw.githubusercontent.com/maximiliense/GLC/master/images/patches_sample_FR.jpg "Example of patch data")

Automatic prediction of the list of species most likely to be observed at a given location is useful for many scenarios related to biodiversity management and conservation.
First, this would allow to improve species identification tools - automatic, semi-automatic or based on traditional field guides - by reducing the list of candidate species observable at a given site.
More generally, it could facilitate biodiversity inventories through the development of location-based recommendation services (e.g. on mobile phones), encourage the involvement of citizen scientist observers, and accelerate the annotation and validation of species observations to produce large, high-quality data sets.
Finally, this could be used for educational purposes through biodiversity discovery applications with features such as contextualized educational pathways.

This repository contains some Python code useful for the [GeoLifeCLEF 2022 challenge](https://www.kaggle.com/c/geolifeclef-2022-lifeclef-2022-fgvc9/) hosted on Kaggle.

In particular, make sure to check the `Getting started` notebooks in directory of the same name for some starter code for data loading and visualization as well as some baselines computation.

We also provide a Pytorch dataset loader which can be used out of hands or adapt to one's needs, it can be found in `data_loading/pytorch_dataset.py`.
