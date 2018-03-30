# Datasets

Put your dataset in a new directory, including RGB-images/labelid-images/... and all A-to-B pairs must be writen into `list.txt`.

## Examples

```sh
# tree datasets
datasets/
├── cityscape/
│   ├── gtFine/
│   ├── leftImg8bit/
│   ├── list_color.txt
│   ├── list_shortcut.txt
│   └── list.txt
│ 
└── README.md


# tail datasets/cityscape/list.txt
leftImg8bit/train/bremen/bremen_000306_000019_leftImg8bit.png gtFine/train/bremen/bremen_000306_000019_gtFine_labelTrainIds.png
leftImg8bit/train/bremen/bremen_000307_000019_leftImg8bit.png gtFine/train/bremen/bremen_000307_000019_gtFine_labelTrainIds.png
leftImg8bit/train/bremen/bremen_000308_000019_leftImg8bit.png gtFine/train/bremen/bremen_000308_000019_gtFine_labelTrainIds.png
leftImg8bit/train/bremen/bremen_000309_000019_leftImg8bit.png gtFine/train/bremen/bremen_000309_000019_gtFine_labelTrainIds.png
leftImg8bit/train/bremen/bremen_000310_000019_leftImg8bit.png gtFine/train/bremen/bremen_000310_000019_gtFine_labelTrainIds.png
leftImg8bit/train/bremen/bremen_000311_000019_leftImg8bit.png gtFine/train/bremen/bremen_000311_000019_gtFine_labelTrainIds.png
leftImg8bit/train/bremen/bremen_000312_000019_leftImg8bit.png gtFine/train/bremen/bremen_000312_000019_gtFine_labelTrainIds.png
leftImg8bit/train/bremen/bremen_000313_000019_leftImg8bit.png gtFine/train/bremen/bremen_000313_000019_gtFine_labelTrainIds.png
leftImg8bit/train/bremen/bremen_000314_000019_leftImg8bit.png gtFine/train/bremen/bremen_000314_000019_gtFine_labelTrainIds.png
leftImg8bit/train/bremen/bremen_000315_000019_leftImg8bit.png gtFine/train/bremen/bremen_000315_000019_gtFine_labelTrainIds.png
```
