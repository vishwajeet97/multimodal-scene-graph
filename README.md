# Image Retrieval using Scene-Graphs

Image retrieval currently depends on the meta-tags associated with the image.
This leads to fairly good searches as long as the query doesn't contain
associations between the objects or their attributes. If a natural langauge sentence
is given as query which demands certain associations in the image then more nuanced 
and robust retrieval needs to be designed which takes into account the noisy and free-form
natural language description alongwith fast searches over the catalouge of images.

We propose a scene-graph assisted architecture which converts the query into a scene-graph 
using SPICE(put link here) and generating scene-graphs of images in catalouge by training
on a cleaned-up version of Visual Genome(put link here) released with GQA challenge(put link here).

## Annotations

Create a data folder at the top-level directory of the repository:
```
# ROOT=path/to/cloned/repository
cd $ROOT
mkdir data
cd data/
mkdir gqa coco
```

## Visual Genome
Download it [here](https://nlp.stanford.edu/data/gqa/sceneGraphs.zip). Unzip it under the data folder. You should see a `gqa` folder unzipped there. 

## Glove Embeddings
Download it [here](http://nlp.stanford.edu/data/glove.42B.300d.zip). Unzip it under the `$ROOT/resources/glove/`. 

## COCO Captions
Download it [here](https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip), this is the Karpathy's cleaned verison of COCO captions. Unzip it and place the `dataset_coco.json` under `$ROOT/data/coco/`. 

## SPICE
This requires StanfordCoreNLP which can be downloaded [here](http://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip). 
Download the pre-compiled jar file for SPICE from [here](https://panderson.me/images/SPICE-1.0.zip). 
Unzip SPICE under `$ROOT/resources/` as `spice` to get `$ROOT/resources/spice/`. Copy the standford jar files inside spice as specified in the readme.

 