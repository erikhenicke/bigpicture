# Seeing the Big Picture

This thesis is set out to explore how spatial context of different resolutions influeces the performance on downstream classification tasks in earth observations.

## Imagery data

First, high resolutions images from the Functional Map of the World (fMoW) dataset will be augmented with additional low-resolution satellite data from Landsat (Google Earth Engine).

fMoW-images can be obtained via `AWS-CLI`:

```shell
aws s3 ls --no-sign-request s3://spacenet-dataset/Hosted-Datasets/fmow/fmow-rgb/
aws s3 cp --no-sign-request s3://spacenet-dataset/Hosted-Datasets/fmow/fmow-rgb/<split>/<category>/ .    
```
