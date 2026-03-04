# Variational Autoencoder for Shoes
Dataset: https://drive.google.com/drive/folders/1XJCrXah3QdzCXaflQRaDx93pS9XhLr3x?usp=sharing
```
Image Reconstruction
├── README.md                   <- This file.
│
├── Code
│   ├── design
│   │   ├── cvae.ipynb          <- Main notebook: architecture, training, evaluation, and generation.
│   │   └── cvae_model          <- Saved trained model (TensorFlow SavedModel format).
│   │       ├── saved_model.pb
│   │       └── variables/
│   │
│   ├── drafts
│   │   └── experimental_classes.py  <- Experimental code and class prototypes (not production).
│   │
│   ├── preprocessing
│   │   ├── exploration.ipynb        <- Dataset exploration: color profiles, size distribution.
│   │   ├── preprocessing.ipynb     <- Image resizing, normalization, train/test split.
│   │   └── ternary_density.html    <- Interactive RGB density visualization.
│   │
│   └── profiling
│       ├── cvae.py                  <- Script version of the model for profiling runs.
│       ├── ncu-report.txt          <- NVIDIA Nsight Compute profiling report (text summary).
│       └── nsight-report.json      <- Full Nsight report (large file, not tracked by git).
│
├── Dataset                     <- Shoe images scraped from StockX.
│   ├── ZAPATOS                     <- Original downloaded images (800x571 px).
│   └── preprocessed            <- Resized (64x64), normalized images split into train/test.
│
└── Report
    ├── Log File 1.tex          <- LaTeX source of the academic report.
    ├── Log File 1.pdf          <- Compiled PDF report.
    └── Images                  <- Figures used in the report.
```
