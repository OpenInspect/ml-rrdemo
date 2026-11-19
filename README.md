# Machine Learning Risk Reduction Demo

This repository is to demonstrate the training of a model capable of identifying solder defects on a sample data set. This demo uses a R-CNN (ResNet50-FPN v2 w/COCO-pretrained weights) to locate and classify defect/no defect for individual solder pads on a cropped IC image.

# Setup

To train and evaluate the ML model, run the following (you will need uv and the Kaggle CLI installed and authenticated)
```
git clone https://github.com/OpenInspect/ml-rrdemo
kaggle datasets download kubeedgeianvs/pcb-aoi
unzip pcb-aoi.zip
uv sync
uv run python train_detector.py --train-dir train_data --test-dir test_data --epochs 40
```

# Initial Results

Final training loss was 0.026 and peak VRAM was 4.0 GiB. The validation column is the last epoch (40/40); the test column uses the checkpoint with the best validation mAP@.5 (0.884). mAP was measured using IoU.

### Detection accuracy

| Metric | Validation (epoch 40) | Held-out test |
|---|---:|---:|
| mAP@0.5 | 0.865 | 0.635 |
| mAP@0.5:0.95 | 0.416 | 0.236 |
| AR@100 | 0.509 | 0.352 |
| mAP@0.5:0.95, `Bad_podu` | 0.459 | 0.252 |
| mAP@0.5:0.95, `Bad_qiaojiao` | 0.374 | 0.220 |

### Defect localization (overlap metrics)

These preliminary results show modest defect detection performance. These results were generated with the model with the best mAP after 40 training epochs on the augmented training set. About 80% of annotated solder defects in the test set have a predicted bounding box with at least 0.5 intersection. Currently, these results do not suggest this model could be used for production use. Given the small size of the unaugmented data set and the possibility for lighting configuration differences, we will likely need to collect our own data set.

These metrics are more forgiving than IoU: a prediction counts as *inside* an annotated defect if at least 50% of the predicted box lies within it, so a smaller box that still highlights the defect is a hit. Predictions are filtered at a confidence of 0.5.

| Metric | Validation (990 preds / 898 GT) | Test (338 preds / 332 GT) |
|---|---:|---:|
| Predictions inside a GT box | 0.856 | 0.781 |
| Predictions inside a GT box, correct class | 0.848 | 0.781 |
| GT defects covered by a prediction | 0.943 | 0.795 |
| GT defects covered, correct class | 0.935 | 0.795 |