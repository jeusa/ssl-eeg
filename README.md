# Bachelor thesis - Supervised and self-supervised contrastive learning on EEG data for classification of mental workload

© 2024 [Jeúsa Hamer](https://orcid.org/0000-0001-8562-8806)

Supervisor: [Felix Putze](https://orcid.org/0000-0001-5203-8797)

University of Bremen

## Repository structure
- **code**: contains the code for preprocessing the EEG data as well as training and evaluating the encoders and the classifiers
    - **ssl_eeg**
        - preprocessing, training, EEGNet, augmentations, plotting, evaluation, classification
    - `main_sl.py`, `main_ssl.py`, `main_head.py`
        - start training for supervised encoders, self-supervised encoders and classifiers
    - jupyter notebooks
        - generate plots, evaluate models, EEG data analysis
    - `env.yml`
        - anaconda dependency file
- **data**: contains labeled n-back EEG data as XDF files and link to unlabeled EEG data
- **models**: contains models that were evaluated in the thesis and some more
    - CSV files
        - configuration and documentation of the trained models
        - evaluated models marked in conf-files as relevant=True
        - `emissions_full.csv`: recordings of power consumption
- **n-back**: contains code to start the n-back application
    - **code**/`main.py`
        - start application in browser, application sends LSL markers
    - `env.yml`
        - anaconda dependency file
    - `n-back_Design.odt`
        - description of the n-back experiment design
- **plots**: confusion matrix, UMAP etc.
- `Bachelorarbeit_de.pdf`: bachelor thesis in german
- `Kolloquium_en.pdf`: presentation slides in english