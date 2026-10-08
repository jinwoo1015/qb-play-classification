# Hips Don't Lie: Quarterback Play Classification from Broadcast Video

Classifying football plays (pass, run, sack, handoff) from standard TV broadcast video using YOLOv8 pose estimation and a Random Forest. Coaches at the collegiate and high school level rarely have sensor tracking data, so this project asks how far plain 2D video can go.

Jinwoo Choi, University of Arizona, Statistics and Data Science (DATA 474 final project).

## Result

| Metric | Value |
| --- | --- |
| Plays | 13 University of Arizona broadcast clips (about 1,970 tracked frames) |
| Validation | Leave one video out cross validation, so no frames from the test play are seen in training |
| All 6 engineered features | about 46% accuracy |
| Best subset (hip horizontal velocity and hip vertical velocity) | **69.2% accuracy** |
| Passes identified | 5 of 5 |
| Sacks identified | 3 of 4 |
| Weak spot | Handoffs versus runs, which look alike when the quarterback sprints forward |

The main finding is that the simplest model won. The hips are a large, central body part that YOLO rarely loses track of, and they act as a proxy for the center of mass. Wrist features were noisier, and the exhaustive search over every feature combination dropped them.

## Method

1. **Crop and track.** Each clip is cropped tightly around the quarterback, then YOLOv8 pose estimation extracts 17 joint coordinates per frame.
2. **Clean the tracks.** The quarterback's track ID fragments are stitched into one continuous track per play.
3. **Engineer six features** from the raw coordinates, using ideas from throwing biomechanics and fall detection research:
   - smoothed maximum wrist velocity and wrist acceleration
   - forward (horizontal) hip velocity
   - vertical hip drop velocity and hip drop acceleration
   - play progress (position in time within the play)
4. **Classify.** A Random Forest (100 trees, max depth 5, balanced class weights) predicts a label for every frame, and the play gets the majority label.
5. **Select features.** All 63 feature combinations are scored with leave one video out validation.

## Repository

| Path | What it is |
| --- | --- |
| `play_classification_paper.qmd` | Full paper and analysis code (Quarto) |
| `play_classification_paper.pdf` | Rendered paper |
| `Clean_QBs/` | Per frame quarterback pose coordinates, one CSV per play |
| `play_features.csv` | Per play summary features and hand labeled play type |
| `final_ai_predictions.csv` | Per play summary features with a cluster ID and an early prediction, from the exploration stage |
| `requirements.txt` | Python dependencies |

## Reproduce the analysis

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
quarto render play_classification_paper.qmd --to pdf
```

The paper imports `cv2` and `ultralytics` at the top, so install the requirements even though the modeling code only needs the CSVs in this repo.

## What is not included

- **Broadcast video and tracked clips.** The source footage is third party TV broadcast content, so it is not published here. The pose extraction code is in the paper (those chunks are set not to run), and it expects video files named `playNN_label.mp4`.
- **Raw per frame detections for every player on the field.** Only the stitched quarterback tracks in `Clean_QBs/` are included.
- **YOLO weights.** The paper uses `yolov8x-pose.pt`, which Ultralytics downloads automatically on first use.
- **Frame figures.** Two figures in the paper are stills from the broadcast and are replaced with a note.
- **Player names.** Plays are numbered and labeled by outcome only.

## Limitations

Thirteen plays is a very small dataset, so the 69.2% figure carries wide uncertainty and should be read as a proof of concept. Pose alone cannot see the ball or the defense. Future work would add defender distance, ball location and a larger, automatically processed dataset.
