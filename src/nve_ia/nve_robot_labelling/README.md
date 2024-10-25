# Robot Labelling 
This package contains tools and nodes for robot self-labelling.  

## LFE Map only
offline_lfe_cloud generates a 

## LFE Feature Extraction
Generates a (no ohm) csv file

## Note: 

All labler in this repository use the @c ohm::SemantiVoxel to store information about the label of a voxel. The `label` contains the current label identifier, the `label_prob` the label_probability in [0,1] form and `state_label` is a field that contains information based on the labler used. 

## Labelling from Experience

There are a n different labelling strategies.
