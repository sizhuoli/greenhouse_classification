
# Greenhouse image classification

This repo contains code on image classification for the paper [Global area boom for greenhouse cultivation revealed by satellite mapping](https://www.researchsquare.com/article/rs-3231996/v1 'link to paper')]

The purpose is to find the presence of greenhouses globally tile by tile (a region of approximately 1 degree cell). The tiles with a positive prediction of greenhouses will be served to the [image segmentation model (https://doi.org/10.5281/zenodo.3978185 'link to paper')]
 
### Key features

- This code splits the image chip labels with the target 'greenhouse' and background 'non-greenhouse', which were saved as csv, into training, validation and testing data.

- Train model using EfficientNet backbones

- Predict at 1km grid for a large area



## Code structure:


### Prepare labels

```
python data_prepare_classification.py
```

--- :bookmark: set configs ---

config/config_classification.yaml

-------------------------------------------------------------------------------------------------------

### Train 1st model: Greenhouse image classification:

```
python main_classification.py
```


-------------------------------------------------------------------------------------------

### Test 1st model: Predict at 1km grid for large area using satellite images (e.g. PlanetScope):

```
python inference_run_classification.py
```

--- :bookmark: set configs ---

config/config_inference_planet.py





