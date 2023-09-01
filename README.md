#  UNSUPERVISED MACHINE LEARNING FOR THE CLASSIFICATION OF ASTROPHYSICAL X-RAY SOURCES
#### Final repository

[Read the paper here. (soon)]()

[Explore our results interactively here.](https://umlcaxs-playground.streamlit.app/)

---
## Abstract
The Chandra Source Catalog (CSC) is the ultimate repository of sources detected by Chandra and represents a fertile ground for discovery in high energy astrophysics. A significant fraction of the sources in CSC have not been studied in detail and only a small fraction of them have been classified. Among the potentially interesting sources that we can look for in Chandra data are compact object mergers, extrasolar planet transits, tidal disruption events, etc. In order to conduct a thorough investigation of the CSC sources, we need to classify as many as we can. This work proposes an unsupervised machine learning approach to classify previously uncategorized Chandra Source Catalog sources by using only the X-ray data available. We propose a new methodology that performs Gaussian Mixture clustering to a set of CSC source properties, and then associates cluster members with objects previously classified in order to provide a probabilistic classification for the sources. Together with classification catalogs for 8756 sources, we present a discussion on the astrophysical implications, specific classification cases, comparison of the results with other catalogs, and limitations of the work. We demonstrate that under certain circumstances, we can distinguish between X-ray binaries from AGNs candidates. We classify YSO and AGN+Seyfert sources with certain confidence (avg. recalls >0.77 and >0.74). In particular cases, AGNs are confused depending on the level of obscuration. Our study offers novel insights into the diverse X-ray properties of astronomical sources, highlighting distinctive variability patterns across different classes and suggesting potential parallels between the AGN unified model and X-ray binaries. Overall, our results demonstrate that it is possible to assign probabilistic classes to a significant fraction of new X-ray sources based on the X-ray information only.

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
