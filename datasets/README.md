# Datasets

Put your dataset in a new directory, including RGB-images/labelid-images/... and all A-to-B pairs must be writen into `list.txt`.

## Examples

```sh
$ ls -al datasets/
total 28
drwxrwxr-x 1 1004 1004    78 Mar 30 07:46 .
drwxrwxr-x 1 1004 1004   178 Mar 30 07:41 ..
lrwxrwxrwx 1 1004 1004    21 Mar 30 06:06 cityscape -> ../../DATA/cityscape/
-rw-rw-r-- 1 1004 1004   185 Mar 30 07:46 README.md


$ ls -al datasets/cityscape/
total 780
drwxrwxr-x 1 1004 1004    126 Mar 28 17:43 .
drwxrwxr-x 1 1004 1004    118 Mar 29 02:40 ..
drwxrwxr-x 1 1004 1004     58 Mar 21 22:27 gtFine
drwxrwxr-x 1 1004 1004     58 Mar 21 21:45 leftImg8bit
-rwxrwxr-x 1 1004 1004 374980 Mar 28 17:43 list_color.txt
-rwxrwxr-x 1 1004 1004   1320 Mar 22 17:25 list_shortcut.txt
-rwxrwxr-x 1 1004 1004 398780 Mar 22 00:10 list.txt

$ tail datasets/cityscape/list.txt
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
