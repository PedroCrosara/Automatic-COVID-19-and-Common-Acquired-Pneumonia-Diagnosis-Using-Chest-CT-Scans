# Automatic COVID-19 and Common-Acquired Pneumonia Diagnosis Using Chest CT Scans
Implementation of our paper about COVID-19 lesion segmentation and COVID-19 or CAP classification. 

Abstract: Even with over 80% of the population being vaccinated against COVID-19, the disease continues to claim victims. Therefore, it is crucial to have a secure Computer-Aided Diagnostic system that can assist in identifying COVID-19 and determining the necessary level of care. This is especially important in the Intensive Care Unit to monitor disease progression or regression in the fight against this epidemic. To accomplish this, we merged public datasets from the literature to train lung and lesion segmentation models with five different distributions. We then trained eight CNN models for COVID-19 and Common-Acquired Pneumonia classification. If the examination was classified as COVID-19, we quantified the lesions and assessed the severity of the full CT scan. To validate the system, we used Resnetxt101 Unet++ and Mobilenet Unet for lung and lesion segmentation, respectively, achieving accuracy of 98.05%, F1-score of 98.70%, precision of 98.7%, recall of 98.7%, and specificity of 96.05%. This was accomplished in just 19.70 s per full CT scan, with external validation on the SPGC dataset. Finally, when classifying these detected lesions, we used Densenet201 and achieved accuracy of 90.47%, F1-score of 93.85%, precision of 88.42%, recall of 100.0%, and specificity of 65.07%. The results demonstrate that our pipeline can correctly detect and segment lesions due to COVID-19 and Common-Acquired Pneumonia in CT scans. It can differentiate these two classes from normal exams, indicating that our system is efficient and effective in identifying the disease and assessing the severity of the condition.

Paper DOI: [10.3390/bioengineering10050529](10.3390/bioengineering10050529)

## Running
In order to run the web-based application you need to first train the models or download the weights from: [Drive](https://drive.google.com/drive/folders/17cy5r-ueuTg180Fnxj698q8TNMBjmtFP?usp=drive_link).

An Lung CT is P002.zip is provided for testing.


## Citation
```
@Article{bioengineering10050529,
AUTHOR = {Motta, Pedro Crosara and Cortez, Paulo César and Silva, Bruno R. S. and Yang, Guang and Albuquerque, Victor Hugo C. de},
TITLE = {Automatic COVID-19 and Common-Acquired Pneumonia Diagnosis Using Chest CT Scans},
JOURNAL = {Bioengineering},
VOLUME = {10},
YEAR = {2023},
NUMBER = {5},
ARTICLE-NUMBER = {529},
URL = {https://www.mdpi.com/2306-5354/10/5/529},
ISSN = {2306-5354},
DOI = {10.3390/bioengineering10050529}
}
```
