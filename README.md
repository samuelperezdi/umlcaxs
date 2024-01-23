#  UNSUPERVISED MACHINE LEARNING FOR THE CLASSIFICATION OF ASTROPHYSICAL X-RAY SOURCES
#### Final repository

[Read the paper here. (now online!)](https://arxiv.org/abs/2401.12203)

[Explore our results interactively here.](https://umlcaxs-playground.streamlit.app/)

---
## Abstract
The automatic classification of X-ray detections is a necessary step in extracting astrophysical information from compiled catalogs of astrophysical sources. Classification is useful for the study of individual objects, statistics for population studies, as well as for anomaly detection, i.e., the identification of new unexplored phenomena, including transients and spectrally extreme sources. Despite the importance of this task, classification remains challenging in X-ray astronomy due to the lack of optical counterparts and representative training sets. We develop an alternative methodology that employs an unsupervised machine learning approach to provide probabilistic classes to Chandra Source Catalog sources with a limited number of labeled sources, and without ancillary information from optical and infrared catalogs. We provide a catalog of probabilistic classes for 8,756 sources, comprising a total of 14,507 detections, and demonstrate the success of the method at identifying emission from young stellar objects, as well as distinguishing between small-scale and large-scale compact accretors with a significant level of confidence. We investigate the consistency between the distribution of features among classified objects and well-established astrophysical hypotheses such as the unified AGN model. This provides interpretability to the probabilistic classifier. Code and tables are available publicly through GitHub. We provide a web playground for readers to explore our final classification at [playground](https://umlcaxs-playground.streamlit.app).

---
**Authors:**

- Víctor Samuel Pérez-Díaz<sup>1,2</sup><sup>*</sup>  
  E-mail: samuelperez.di@gmail.com
- Juan Rafael Martínez-Galarza<sup>1</sup>
- Alexander Caicedo<sup>3, 4</sup>
- Raffaele D'Abrusco<sup>1</sup>

**Affiliations:**

1. Center for Astrophysics | Harvard & Smithsonian, 60 Garden Street, Cambridge, MA 02138, USA
2. School of Engineering, Science and Technology, Universidad del Rosario, Cll. 12C No. 6-25, Bogotá, Colombia
3. Department of Electronics Engineering, Pontificia Universidad Javeriana, Cra. 7 No. 40-62, Bogotá, Colombia
4. Ressolve, Cra. 42 # 5 Sur - 145, Medellín, Colombia

---
## Reproducing the results

**Note**: Due to the ever-changing nature of the SIMBAD database, the exact results presented in the paper may not be reproducible. Our crossmatch was performed in August 2022. We provide the original `cluster_csc_simbad.csv`. With this, you can reproduce the same result starting from step 3.

For exact reproducibility, we suggest to install package versions provided in the `environment.txt` file.

#### Pipeline execution steps

Follow the steps below to execute the research pipeline:

##### 1. Clustering
Execute the clustering script to generate clusters.
```bash
python3 clustering.py
```

##### 2. SIMBAD Crossmatch
Crossmatch the generated `cluster_csc.csv` file with the SIMBAD database within a 1" radius, choosing the Best match. Retain all rows in the `cluster_csc.csv` set. Use [TOPCAT](https://www.star.bris.ac.uk/~mbt/topcat/sun253/sun253.html) for a convenient crossmatch process.

##### 3. Classification
Execute the classification script.
```bash
python3 classification.py
```

##### 4. Master Classification
Execute the master classification script.
```bash
python3 master_classification.py
```

##### 5. Result Inspection
Inspect the resulting tables in the `/out_data` directory:
- `detection_level_classification.csv`
- `uniquely_classified.csv`
- `ambiguous_classification.csv`

By following these steps, you should be able to closely replicate the pipeline presented in the paper.

---
## Classify your own X-ray source

Check the notebook `classify_your_source.ipynb` for instructions on how to classify your own X-ray source(s) with our pipeline.
