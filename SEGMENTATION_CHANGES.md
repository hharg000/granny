## Segmentation: fruit numbering, false-detection filtering and research mode

This fork changes the `segmentation` analysis so that the numbers on the images match the file names, obvious false detections are filtered out (and kept for review), and fruit can optionally be labeled by their position on a tray. All changes are in `Granny/Analyses/Segmentation.py`, with two small supporting edits (`Analysis.py` help text and `StringValue.py`).

### What was fixed

**Numbers now match file names.** Previously the number drawn on the masked image did not always agree with the `fruit_NN` number in the cropped image's file name. The same ordering is now used for both, so `fruit_07` is the fruit labeled 7 on the masked image.

**False detections are filtered out.** The model sometimes detects things that are not fruit, such as a sliver of tray rim at the edge of the photo or a long strip of background. A fruit detection is discarded when its bounding box is much smaller than the typical fruit on that tray image (`--min_area_ratio`, default 0.3 of the median area) or much longer than it is wide (`--max_aspect_ratio`, default 3.0). Setting either option to 0 turns that check off.

**Default numbering is system-agnostic.** In the default (`standard`) mode, fruit are numbered 1, 2, 3, ... in the order they are found (top row first, right to left), for any number of fruit and any tray. Nothing about a particular tray is assumed.

### Research mode

Research mode labels each fruit by its position on a tray you describe, so a given number always refers to the same position from photo to photo.

```
granny -i cli --analysis segmentation --mode research --nexpect 18 --tray_layout my_tray_layout.csv
```

- `--nexpect` is the number of fruit a full tray holds.
- `--tray_layout` is a CSV file describing the tray. Each line of the file is one row of the tray, read top to bottom, and the values are the labels of the positions in that row, left to right, as the oriented photo is viewed. For example, a 4/5/4/5 tray:

  ```
  1,2,3,4
  5,6,7,8,9
  10,11,12,13
  14,15,16,17,18
  ```

  Blank cells, comments, byte-order marks and semicolon or tab separators are tolerated, and duplicate labels are reported as errors.
- Fruit are assigned to positions row by row. If a fruit is missing, its position is left empty and the remaining fruit keep the labels of the positions they actually sit on, rather than shifting. Anything that cannot be placed with confidence is labeled `U01`, `U02`, ... and is never guessed.
- Each tray image gets a status in `numbering_report.csv`: `ok` (all positions filled), `fewer` or `more` (compared with `--nexpect`), or `check` (something needs a human look). A note on the masked image and in the report explains why.

**Automatic orientation.** Photos in one folder are not always taken the same way up. In research mode (with the default `--orientation auto`) Granny tries the image turned 0, 90, 180 and 270 degrees and keeps the turn that best fits the tray layout, so every photo ends up with the same view, for example 4 fruit in the top row and 5 in the bottom row. Portrait images are first turned to landscape. In the default mode, `--orientation` can also be set to 0, 90, 180 or 270 to turn every image by a fixed amount.

**Symmetric trays and the color card.** Some trays look identical upside down (for example a 25-position 5x5 tray), so the layout alone cannot say which way up the photo should be. For these, Granny looks for a color card (ColorChecker) and turns the image so the card is on the side given by `--card_position` (`top-right` by default; also `top-left`, `bottom-right`, `bottom-left`). If no card is found, the image is **not** turned, no assumption is made, and a note is added to the report.

**Optional tray reference.** If the camera and tray are in about the same place in every photo, `--tray_reference` can be given a CSV of recorded position coordinates. It is used only to help place fruit when the layout rows cannot (for example when a whole row is missing), and only when the tray has moved by less than about a third of the spacing between neighboring positions. A photo whose tray sits noticeably off the reference is flagged for checking, and a fruit that does not clearly sit on a recorded position is never labeled from it. Granny writes a `tray_reference.csv` automatically at the end of a research run (best made from photos of full trays), which can be reused on later runs.

### Output folders

Each run writes to `results/segmentation/<date-time>/`. The outputs are arranged so a person can always audit what the filter did:

| Folder or file | Contents |
|---|---|
| `full_masked_images/` | Everything the model detected on the tray (fruit, tray information, true and false detections alike), with colored masks, boxes and confidence scores. No fruit numbers. This is the original Granny style of output. |
| `full_masked_images_clean/` | The same image turned as specified, with false detections removed and every fruit numbered to match its file name. |
| `segmented_images/` | The numbered crops of the true fruit (`<image>_fruit_NN.png`). |
| `segmented_images_false_positive/` | A crop of each detection that was discarded as a false detection (`<image>_false_positive_NN.png`). Only created when something was discarded. |
| `tray_infos/` | Tray information crops, unchanged. |
| `fruit_manifest.csv` | Each fruit file with its label, position on the image and how it was placed. |
| `false_positives.csv` | Each discarded detection with its file name, the reason, its box coordinates and its confidence. |
| `numbering_report.csv` | Research mode only: one line per tray image with the count check, status, rotation applied and notes. |
| `tray_reference.csv` | Research mode only: positions measured from this run, for reuse with `--tray_reference`. |

Comparing `full_masked_images/` with `full_masked_images_clean/`, and looking through `segmented_images_false_positive/` and `false_positives.csv`, shows whether a discarded detection was a true false positive (the model may need retraining) or something the filter removed wrongly (the filter settings may need adjusting).

### Options added

| Option | Purpose |
|---|---|
| `--mode` | `standard` (default) or `research`. |
| `--nexpect` | Number of fruit on a full tray (research mode). |
| `--tray_layout` | CSV describing the tray rows and position labels (research mode). |
| `--tray_reference` | Optional CSV of recorded position coordinates. |
| `--orientation` | `auto` (default), or a fixed turn of 0, 90, 180 or 270 degrees clockwise. |
| `--card_position` | Where the color card should end up for symmetric trays (default `top-right`). |
| `--min_area_ratio` | Smallest allowed fruit size relative to the tray's median fruit (default 0.3; 0 disables). |
| `--max_aspect_ratio` | Longest allowed bounding-box shape (default 3.0; 0 disables). |
| `--row_tolerance` | Existing option; controls how fruit are grouped into rows when numbering (default 20). |

The new output folders are written automatically and have no command-line options. Run `granny -i cli --analysis segmentation --help` (together with `--input`) for the full descriptions.
