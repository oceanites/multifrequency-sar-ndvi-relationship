# Material for the relationship analysis between SAR backscatter and NDVI values
This material belongs to the paper *"Comparing the relationship between NDVI and SAR backscatter across different frequency bands in agricultural areas"*, in which the relationship between SAR backscatter and NDVI values is analysed.
The paper is published in [Remote Sensing of Environment](https://www.sciencedirect.com/science/article/pii/S0034425725000161) (DOI: [10.1016/j.rse.2025.114612](https://doi.org/10.1016/j.rse.2025.114612)).

The material is additionally archived at the [Open Data Hub of the University of the Bundeswehr Munich](https://open-data.unibw.de/dataset.xhtml?persistentId=doi:10.60776/4WIAJ0) (DOI: [10.60776/4WIAJ0](https://doi.org/10.60776/4WIAJ0)).

## Field outlines

These field outlines were manually labeled using Sentinel-1 and -2 time series, augmented by Google Satellite imagery.

| File                                                 | Field outlines of study area   |
| ---------------------------------------------------- | -------------------------------|
| `fields-outlines-bell-ville-argentina.geojson`       | around Bell Ville, Argentina   |
| `fields-outlines-boort-australia.geojson`            | area around Boort, Australia   |
| `fields-outlines-mekong-river-delta-vietnam.geojson` | in Mekong River delta, Vietnam |

## Extracted data/values
For each study area a csv file with the extracted data is uploaded in the folder `extracted-values`.

## Notebooks and processing scripts
Most of the data extraction and analysis workflow was done using Jupyter Notebooks, while some other steps were done using separate python scripts.

| File                                            |                                                                                             |
| ----------------------------------------------- | ------------------------------------------------------------------------------------------- |
| `multifrequtils.py`                             | Methods used by multiple processing scripts and notebooks.                                  |
| `statistics-mekong-river-delta.ipynb`           | Code for data extraction and analysis for the Mekong River Delta (Vietnam) study area.      |
| `statistics-bell-ville.ipynb`                   | Code for data extraction and analysis for the Bell Ville (Argentina) study area.            |
| `statistics-boort.ipynb`                        | Code for data extraction and analysis for the Boort (Australia) study area.                 |
| `2024-10-14-create-rvi-tif.py`                  | Script to calculate the RVI using two polarizations given some backscatter data.            |
| `2024-10-14-create-rvi-novasar-threepol-tif.py` | Script to calculate the RVI using three polarizations given some backscatter data.          |
| `2024-10-15-calibrate-novasar.py`               | Script to calibrate the NovaSAR data, converting it from DN to backscatter in linear scale. |


## SAR and optical data processing
Processing of the SAR data is done mostly using SNAP, for which the processing graphs are included in the folder `snap-graphs/`. Additionally, for some steps (like RVI calculation or some NovaSAR processing steps), Python scripts were created, which are contained in `code/`.
 SNAP processing graphs

| Snap Graph                               | Description                                                                                                                                                                  |
| ---------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `2024-07-25-s1-grd-boort.xml`            | S1 processing with Subset for Boort study area                                                                                                                               |
| `2024-07-25-s1-grd-bell-ville.xml`       | S1 processing with Subset for Bell Ville study area                                                                                                                          |
| `2024-07-25-s1-grd-vietnam.xml`          | S1 processing with Subset for Mekong River Delta study area                                                                                                                  |
| `2024-06-25-novasar-tc.xml`              | Terrain correction for NovaSAR SCD files, calibration is done using a python script.                                                                                         |
| `2024-07-10-novasar-incidence-angle.xml` | Create a tif image containing the incidence angle of NovaSAR images.                                                                                                         |
| `2024-07-25-saocom-ml-tc.xml`            | SAOCOM processing                                                                                                                                                            |
| `multifreq-2024-06-24-csg-ml3-tc.xml`    | CSK processing                                                                                                                                                               |
| `beam-dimap-to-tif.xml`                  | Convert BEAM DIMAP files to tifs. Processing in SNAP is way faster, when the output is written as a beam dimap file. Converting this to a tif image is then still very fast. |

# Citation

To cite the paper this text can be used:
> Roßberg T, Schmitt M (2025). “Comparing the relationship between NDVI and SAR backscatter across different frequency bands in agricultural areas”. Remote Sensing of Environment 319, Art. No. 114612

or the following bibtex/biblatex snippet:

```bibtex
@article{Rossberg2025,
 author = {Thomas Roßberg and Michael Schmitt},
 title = {Comparing the relationship between {NDVI} and {SAR} backscatter across different frequency bands in agricultural areas},
 journal = {Remote Sensing of Environment},
 volume = {319},
 eid = {114612},
 year = {2025},
 issn = {0034-4257},
 doi = {https://doi.org/10.1016/j.rse.2025.114612},
 url = {https://www.sciencedirect.com/science/article/pii/S0034425725000161},
}
```
