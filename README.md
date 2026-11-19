# Machine Learning Risk Reduction Demo

This repository is to demonstrate the training of a model capable of identifying solder defects on a sample data set. This demo uses a R-CNN to locate and classify defect/no defect for individual solder pads on a cropped IC image.

# Setup

To train and evaluate the ML model, run the following (you will need uv and the Kaggle CLI installed and authenticated)
```
git clone https://github.com/OpenInspect/ml-rrdemo
kaggle datasets download kubeedgeianvs/pcb-aoi
unzip pcb-aoi.zip
uv sync
uv run python train_detector.py --train-dir train_data --test-dir test_data --epochs 40
```