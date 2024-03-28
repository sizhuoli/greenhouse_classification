
# Greenhouse image classification

This repo contains code for the paper [Global area boom for greenhouse cultivation revealed by satellite mapping](https://www.researchsquare.com/article/rs-3231996/v1 'link to paper')
 
### Key features

For the Danish dataset:

- We offer image patches preprocessed in two ways: a. patch-normalization to 0 mean and unit std (used in paper); b. raw patches with orginial pixel intensities.

- There are several empty patches with no crown delineations (used as negative sample for training), which can be removed 

- Coordinates have been removed



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

### Test 1st model: Predict at 1km grid for large image (e.g. PlanetScope):

```
python inference_run_classification.py
```

--- :bookmark: set configs ---

config/config_inference_planet.py





