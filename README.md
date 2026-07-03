# The Visual Storyteller

## What this project does

This project builds a model that looks at an image and writes a caption for it in plain English. It's an image captioning system: you give it a picture, it gives you back a sentence describing what's in the picture (objects, actions, setting).


## How the model works

- **Encoder (CNN):** A pretrained ResNet50 extracts a grid of visual features from the image, instead of just a single flattened vector. This keeps spatial information (which region of the image contains what) so the model can "look" at different parts of the image at different times.
- **Attention:** At every step of generating a word, the model computes an attention map over the image grid, deciding which regions are relevant to the word it's about to produce.
- **Decoder (LSTM):** An LSTM generates the caption one word at a time, using the attended image features plus the words generated so far.
- **Training trick — scheduled sampling:** Early in training the model is shown the correct previous word (teacher forcing). As training progresses, it's gradually forced to rely more on its own previous predictions, which makes it more robust at inference time.
- **Two-phase training:** The ResNet is frozen for most of training (only the LSTM/attention learn), then in later epochs the last conv block of the ResNet is unfrozen and fine-tuned.

## Project structure

```
├── data_and_training.ipynb    # Data loading, model definition, training, saving checkpoints
├── inference.ipynb            # Loads trained model, generates captions on unseen images
├── content/
│   ├── Images/                # Raw dataset images
│   └── captions.txt           # image_filename,caption 
├── experiments/               # Auto-created. One folder per training run:
│   └── <timestamp>_<experiment_name>/
│       ├── config.json
│       ├── vocab.pth
│       ├── logs/              # TensorBoard logs
│       └── weights/
│           ├── checkpoint_epoch_N.pth.tar
│           └── best_model.pth.tar # best checkpoint (lowest val loss)
└── inference_results/ (or content/caption_results/)
    ├── individual/            # one captioned image per test image
    ├── contact_sheets/        # grids of many captioned images for fast visual comparison
    └── predictions.csv        # filename -> generated caption
```

## Dataset

The provided dataset of 8,000 images is used, each paired with 5 human-written captions.

1. Download the dataset zip from the link provided in the assignment.
2. Unzip it so you end up with:
   - `content/Images/` — all the image files
   - `content/captions.txt` — a CSV-style file with lines like `image_filename,caption`
3. Place the `content/` folder in the project root (same folder as the notebooks).

The training code automatically:
- Drops empty or unusually long captions (outliers that would distort batch padding).
- Splits the data by **image**, not by caption, so all 5 captions of one image stay in the same split (train/val/test). This prevents the model from "seeing" an image during training and then being tested on a slightly different caption for that same image.
- Builds the vocabulary using only the training captions, to avoid vocabulary leakage from val/test data.

## Setup

### Requirements
- Python 3.9+
- A GPU is recommended (training is much faster with CUDA), but the code also runs on CPU.

### Install dependencies

```bash
pip install torch torchvision torchmetrics tensorboard pillow numpy pandas matplotlib tqdm
```

If you're using a GPU, install the CUDA-enabled build of PyTorch instead — check [pytorch.org](https://pytorch.org/get-started/locally/) for the correct command for your system.

## How to run

### 1. Train the model — `data_and_training.ipynb`

1. Make sure `content/Images/` and `content/captions.txt` exist (see Dataset section above).
2. Open `data_and_training.ipynb` and run all cells, or run the script version:
   ```bash
   python data_and_training.py
   ```
3. This will:
   - Load and split the dataset (train/val/test).
   - Build the vocabulary.
   - Train the CNN+Attention+LSTM model for up to 30 epochs (with early stopping if validation loss stops improving).
   - Periodically evaluate BLEU-4 score on validation data.
   - Save a checkpoint every epoch and the best model separately to `experiments/<run_name>/weights/best_model.pth.tar`.

Training progress (loss per epoch, BLEU scores) is printed to the console and also logged to TensorBoard. To view TensorBoard logs:

```bash
tensorboard --logdir experiments
```

### 2. Generate captions — `inference.ipynb`

1. Open `inference.ipynb`.
2. It automatically finds the most recently trained checkpoint under `experiments/**/best_model.pth.tar` (or you can set `CHECKPOINT_FILE` manually to a specific path).
3. Run all cells. This will:
   - Load the trained model, vocabulary, and image preprocessing pipeline from the checkpoint.
   - Run `generate_caption(image_path, model)` on a set of unseen test images.
   - Save a labeled image + caption for every test image, plus grid "contact sheets" that combine many results into one image for quick visual comparison.
   - Save all predictions to a `predictions.csv` file.

The core function used everywhere is:

```python
def generate_caption(image_path: str, model: CaptionModel) -> str:
    """Takes a path to an image and returns a generated caption string."""
```

`model` here is a `CaptionModel`, built with `CaptionModel.from_checkpoint("path/to/best_model.pth.tar")`, which bundles the trained network, vocabulary, transform, and device together.

Captioning can run in two modes:
- **Beam search** (default, `use_beam_search=True`) — explores multiple candidate sentences and picks the best-scoring one. Slower but usually produces better captions.
- **Greedy decoding** (`use_beam_search=False`) — picks the single most likely next word at each step. Faster, lower quality.

## Notes on evaluation

- BLEU-4 scores reported during training are computed against a **single** reference caption per image (not all 5), so treat them as a relative, epoch-over-epoch progress signal rather than a paper-grade benchmark number.
- The `inference.ipynb` notebook contains examples of both successful captions and failure cases for qualitative analysis, alongside the quantitative BLEU score from training.
```
