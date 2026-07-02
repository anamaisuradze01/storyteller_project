import nbformat as nbf

def md(src):
    return nbf.v4.new_markdown_cell(src)

def code(src):
    return nbf.v4.new_code_cell(src)

# =========================================================================
# NOTEBOOK 1: data_and_training.ipynb
# =========================================================================
nb1 = nbf.v4.new_notebook()
nb1.cells = []

nb1.cells.append(md(r"""# Visual Storyteller — Data & Training
### ResNet101 + Attention LSTM Image Captioning (Flickr8k)

**Run this on Google Colab with a GPU runtime** (`Runtime -> Change runtime type -> T4 GPU`).

This notebook combines `data_loader.py`, `model.py`, `utils.py`, and `train.py` into one place so it can train inside a single Colab session.

**Compute budget note (free Colab, T4 GPU):**
With the encoder frozen and batch size 32, one epoch over the ~6,400-image train split is roughly **3–6 minutes** once the dataset lives on the local Colab disk (not read live from Drive). That puts:
- 20 epochs ≈ **1–2 hours**
- 30 epochs ≈ **1.5–3 hours**

This is why the config below defaults to `num_epochs=20` (down from the assignment's 100) with early stopping (`patience=5`) — it will very likely stop even earlier once validation loss plateaus. This is far more than enough for a Flickr8k captioning model to produce coherent captions; diminishing returns set in well before epoch 100 for this dataset size.

Free Colab GPUs disconnect after ~90 minutes of *browser inactivity* and sessions are capped at ~12 hours total, so keep the tab open/interact occasionally, or wrap training in the resumable checkpointing already built into `train()` (it can resume from `checkpoint_path`)."""))

nb1.cells.append(md(r"""## 0. Setup: GPU check + dependencies"""))

nb1.cells.append(code(r"""!nvidia-smi
!pip install -q torchmetrics tensorboard
"""))

nb1.cells.append(md(r"""## 1. Get the dataset onto local Colab disk

The dataset link in the assignment is a SharePoint share link, which Colab can't fetch anonymously. Do **one** of the following, then run the cell below:

**Option A (recommended, fastest):**
1. Download `caption_data.zip` from the SharePoint link on your own machine.
2. Upload it to your Google Drive (e.g. `MyDrive/caption_data.zip`).
3. Run the cell below — it mounts Drive and copies+unzips to local disk `/content/caption_data` (local disk is much faster for training than reading directly off a mounted Drive).

**Option B (small/no Drive space):**
Use the Colab file browser's upload button (or `files.upload()`) to upload `caption_data.zip` directly into `/content/`, then just run the unzip part of the cell."""))

nb1.cells.append(code(r"""import os, shutil, zipfile

DRIVE_ZIP_PATH = "/content/drive/MyDrive/caption_data.zip"  # <-- update if your path differs
LOCAL_ZIP_PATH = "/content/caption_data.zip"
DATA_ROOT = "/content/caption_data"

USE_DRIVE = True  # set False if you uploaded the zip directly to /content instead

if USE_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
    if not os.path.exists(LOCAL_ZIP_PATH):
        print("Copying zip from Drive to local disk (this speeds up training a lot)...")
        shutil.copy(DRIVE_ZIP_PATH, LOCAL_ZIP_PATH)

if not os.path.exists(DATA_ROOT):
    print("Unzipping...")
    with zipfile.ZipFile(LOCAL_ZIP_PATH, "r") as zf:
        zf.extractall("/content/")
    print("Done.")
else:
    print("Dataset already extracted at", DATA_ROOT)

# Sanity check: adjust these two paths if your zip's internal folder names differ
IMAGES_DIR = os.path.join(DATA_ROOT, "Images")
ANNOTATIONS_FILE = os.path.join(DATA_ROOT, "captions.txt")
print("Images dir exists:", os.path.exists(IMAGES_DIR))
print("Annotations file exists:", os.path.exists(ANNOTATIONS_FILE))
if os.path.exists(IMAGES_DIR):
    print("Num images:", len(os.listdir(IMAGES_DIR)))
"""))

nb1.cells.append(md(r"""## 2. Imports"""))

nb1.cells.append(code(r"""import os
import re
import random
import json
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models import ResNet101_Weights
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

print("Torch:", torch.__version__, "| CUDA available:", torch.cuda.is_available())
"""))

nb1.cells.append(md(r"""## 3. Data pipeline (from `data_loader.py`)"""))

nb1.cells.append(code(r"""class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text):
        text = text.lower().strip()
        text = re.sub(r"([.,!?;:'\"])", r" \1 ", text)
        return text.split()

    def build_vocabulary(self, sentence_list):
        frequencies = {}
        idx = 4
        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                frequencies[word] = frequencies.get(word, 0) + 1
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)
        return [self.stoi.get(token, self.stoi["<UNK>"]) for token in tokenized_text]


class FlickrDataset(Dataset):
    def __init__(self, root_dir, imgs, captions, vocab, transform=None):
        self.root_dir = root_dir
        self.imgs = imgs
        self.captions = captions
        self.vocab = vocab
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def _load_image(self, index):
        img_path = os.path.join(self.root_dir, self.imgs[index])
        return Image.open(img_path).convert("RGB")

    def __getitem__(self, index):
        caption = self.captions[index]
        try:
            image = self._load_image(index)
        except (FileNotFoundError, UnidentifiedImageError, OSError) as e:
            print(f"[FlickrDataset] Skipping unreadable image '{self.imgs[index]}' ({e}); substituting a different sample.")
            fallback_index = random.randrange(len(self.imgs))
            return self.__getitem__(fallback_index)

        if self.transform is not None:
            image = self.transform(image)

        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<EOS>"])
        return image, torch.tensor(numericalized_caption)


def collate_batch_first(batch, pad_idx):
    imgs = torch.stack([item[0] for item in batch], dim=0)
    captions = [item[1] for item in batch]
    padded = pad_sequence(captions, batch_first=True, padding_value=pad_idx)
    return imgs, padded


class CaptionCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        return collate_batch_first(batch, self.pad_idx)


def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])


def _read_annotations(annotation_file, max_caption_length=40):
    all_imgs, all_captions = [], []
    dropped = 0
    with open(annotation_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        if lines and "image,caption" in lines[0]:
            lines = lines[1:]
        for line in lines:
            parts = line.strip().split(",", 1)
            if len(parts) != 2:
                continue
            img_id, caption = parts
            n_tokens = len(Vocabulary.tokenizer_eng(caption))
            if n_tokens == 0 or n_tokens > max_caption_length:
                dropped += 1
                continue
            all_imgs.append(img_id)
            all_captions.append(caption)
    if dropped:
        print(f"[data_loader] Dropped {dropped} caption(s) that were empty or longer than {max_caption_length} tokens.")
    return all_imgs, all_captions


def _split_by_image(all_imgs, val_size, test_size, split_seed=42):
    unique_imgs = list(set(all_imgs))
    rng = random.Random(split_seed)
    rng.shuffle(unique_imgs)
    total = len(unique_imgs)
    v_count = int(total * val_size)
    t_count = int(total * test_size)
    train_count = total - v_count - t_count
    train_ids = set(unique_imgs[:train_count])
    val_ids = set(unique_imgs[train_count:train_count + v_count])
    test_ids = set(unique_imgs[train_count + v_count:])
    return train_ids, val_ids, test_ids


def get_loaders(root_folder, annotation_file, transform=None, train_transform=None,
                 val_transform=None, batch_size=32, num_workers=2, shuffle=True,
                 pin_memory=True, test_size=0.1, val_size=0.1, freq_threshold=5,
                 max_caption_length=40, split_seed=42):
    train_transform = train_transform or transform
    val_transform = val_transform or transform

    all_imgs, all_captions = _read_annotations(annotation_file, max_caption_length)
    train_ids, val_ids, test_ids = _split_by_image(all_imgs, val_size, test_size, split_seed)

    train_imgs, train_caps = [], []
    val_imgs, val_caps = [], []
    test_imgs, test_caps = [], []

    for img, cap in zip(all_imgs, all_captions):
        if img in train_ids:
            train_imgs.append(img); train_caps.append(cap)
        elif img in val_ids:
            val_imgs.append(img); val_caps.append(cap)
        elif img in test_ids:
            test_imgs.append(img); test_caps.append(cap)

    print(f"[data_loader] Split: {len(train_ids)} train / {len(val_ids)} val / {len(test_ids)} test images "
          f"({len(train_caps)} / {len(val_caps)} / {len(test_caps)} captions)")

    vocab = Vocabulary(freq_threshold)
    vocab.build_vocabulary(train_caps)
    print(f"[data_loader] Vocabulary size: {len(vocab)}")

    train_dataset = FlickrDataset(root_folder, train_imgs, train_caps, vocab, transform=train_transform)
    val_dataset = FlickrDataset(root_folder, val_imgs, val_caps, vocab, transform=val_transform)
    test_dataset = FlickrDataset(root_folder, test_imgs, test_caps, vocab, transform=val_transform)

    pad_idx = vocab.stoi["<PAD>"]
    collate_fn = CaptionCollate(pad_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers,
                               shuffle=shuffle, pin_memory=pin_memory, collate_fn=collate_fn,
                               drop_last=True, persistent_workers=num_workers > 0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers,
                             shuffle=False, pin_memory=pin_memory, collate_fn=collate_fn,
                             persistent_workers=num_workers > 0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers,
                              shuffle=False, pin_memory=pin_memory, collate_fn=collate_fn,
                              persistent_workers=num_workers > 0)

    return train_loader, val_loader, test_loader, vocab
"""))

nb1.cells.append(md(r"""## 4. Model definition (from `model.py`)

ResNet101 CNN encoder (frozen by default, optionally fine-tuned later) + Bahdanau-attention LSTM decoder, with greedy and beam-search caption generation."""))

nb1.cells.append(code(r"""class EncoderCNN(nn.Module):
    def __init__(self, encoded_image_size=8, fine_tune=False):
        super(EncoderCNN, self).__init__()
        self.encoded_image_size = encoded_image_size
        resnet = models.resnet101(weights=ResNet101_Weights.DEFAULT)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        self.set_fine_tune(fine_tune)

    def forward(self, images):
        features = self.resnet(images)
        features = self.adaptive_pool(features)
        features = features.permute(0, 2, 3, 1)
        features = features.view(features.size(0), -1, features.size(-1))
        return features

    def set_fine_tune(self, fine_tune=False, unfreeze_from_block=7):
        for p in self.resnet.parameters():
            p.requires_grad = False
        if fine_tune:
            children = list(self.resnet.children())
            for block in children[unfreeze_from_block:]:
                for p in block.parameters():
                    p.requires_grad = True


class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.attn_proj = nn.Linear(encoder_dim + decoder_dim, attention_dim)
        self.score_proj = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, encoder_out, decoder_hidden):
        num_pixels = encoder_out.size(1)
        hidden_expanded = decoder_hidden.unsqueeze(1).expand(-1, num_pixels, -1)
        energy = torch.tanh(self.attn_proj(torch.cat((encoder_out, hidden_expanded), dim=2)))
        scores = self.score_proj(energy).squeeze(2)
        alpha = F.softmax(scores, dim=1)
        context = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)
        return context, alpha


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, encoder_dim=2048,
                 attention_dim=256, dropout=0.5, embed_dropout=0.2):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.encoder_dim = encoder_dim
        self.attention = Attention(encoder_dim, hidden_size, attention_dim)
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.embed_dropout = nn.Dropout(embed_dropout)
        self.lstm_cell = nn.LSTMCell(embed_size + encoder_dim, hidden_size, bias=True)
        self.init_h = nn.Linear(encoder_dim, hidden_size)
        self.init_c = nn.Linear(encoder_dim, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self._init_weights()

    def _init_weights(self):
        nn.init.uniform_(self.embed.weight, -0.1, 0.1)
        nn.init.uniform_(self.fc.weight, -0.1, 0.1)
        nn.init.zeros_(self.fc.bias)

    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, captions, teacher_forcing_ratio=1.0):
        batch_size = encoder_out.size(0)
        device = encoder_out.device
        num_steps = captions.size(1) - 1
        num_pixels = encoder_out.size(1)

        h, c = self.init_hidden_state(encoder_out)
        predictions = torch.zeros(batch_size, num_steps, self.vocab_size, device=device)
        alphas = torch.zeros(batch_size, num_steps, num_pixels, device=device)
        input_word = captions[:, 0]

        for t in range(num_steps):
            embeddings = self.embed_dropout(self.embed(input_word))
            context, alpha = self.attention(encoder_out, h)
            h, c = self.lstm_cell(torch.cat((embeddings, context), dim=1), (h, c))
            h_norm = self.layer_norm(h)
            preds = self.fc(self.dropout(h_norm))
            predictions[:, t, :] = preds
            alphas[:, t, :] = alpha

            use_teacher_forcing = self.training and (random.random() < teacher_forcing_ratio)
            if use_teacher_forcing:
                input_word = captions[:, t + 1]
            else:
                input_word = preds.argmax(dim=1).detach()

        return predictions, alphas


class CNNtoRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, attention_dim=256,
                 encoder_dim=2048, dropout=0.5, fine_tune_encoder=False):
        super(CNNtoRNN, self).__init__()
        self.encoderCNN = EncoderCNN(fine_tune=fine_tune_encoder)
        self.decoderRNN = DecoderRNN(embed_size=embed_size, hidden_size=hidden_size,
                                      vocab_size=vocab_size, encoder_dim=encoder_dim,
                                      attention_dim=attention_dim, dropout=dropout)

    def forward(self, images, captions, teacher_forcing_ratio=1.0):
        encoder_out = self.encoderCNN(images)
        predictions, alphas = self.decoderRNN(encoder_out, captions, teacher_forcing_ratio)
        return predictions, alphas

    def unfreeze_encoder(self):
        self.encoderCNN.set_fine_tune(fine_tune=True)

    @torch.no_grad()
    def caption_image(self, image, vocabulary, max_length=50, device="cuda"):
        self.eval()
        encoder_out = self.encoderCNN(image)
        h, c = self.decoderRNN.init_hidden_state(encoder_out)
        word = torch.tensor([vocabulary.stoi["<SOS>"]], device=device)
        result_caption = []
        for _ in range(max_length):
            embeddings = self.decoderRNN.embed(word)
            context, _ = self.decoderRNN.attention(encoder_out, h)
            h, c = self.decoderRNN.lstm_cell(torch.cat((embeddings, context), dim=1), (h, c))
            h_norm = self.decoderRNN.layer_norm(h)
            preds = self.decoderRNN.fc(h_norm)
            predicted = preds.argmax(dim=1)
            token = vocabulary.itos[predicted.item()]
            if token == "<EOS>":
                break
            if token != "<SOS>":
                result_caption.append(token)
            word = predicted
        self.train()
        return result_caption

    @torch.no_grad()
    def beam_search_caption(self, image, vocabulary, max_length=50, beam_width=3, device="cuda"):
        self.eval()
        encoder_out = self.encoderCNN(image)
        h0, c0 = self.decoderRNN.init_hidden_state(encoder_out)

        start_word = torch.tensor([vocabulary.stoi["<SOS>"]], device=device)
        embeddings = self.decoderRNN.embed(start_word)
        context, _ = self.decoderRNN.attention(encoder_out, h0)
        h, c = self.decoderRNN.lstm_cell(torch.cat((embeddings, context), dim=1), (h0, c0))
        h_norm = self.decoderRNN.layer_norm(h)
        log_probs = F.log_softmax(self.decoderRNN.fc(h_norm), dim=1)
        top_probs, top_idx = log_probs.topk(beam_width, dim=1)

        beams = []
        for i in range(beam_width):
            word_idx = top_idx[0][i]
            score = top_probs[0][i].item()
            beams.append((score, word_idx.unsqueeze(0), h, c, [word_idx.item()]))

        for _ in range(max_length - 1):
            candidates = []
            for score, last_word, h_i, c_i, seq in beams:
                if vocabulary.itos[seq[-1]] == "<EOS>":
                    candidates.append((score, last_word, h_i, c_i, seq))
                    continue
                embeddings = self.decoderRNN.embed(last_word)
                context, _ = self.decoderRNN.attention(encoder_out, h_i)
                h_new, c_new = self.decoderRNN.lstm_cell(torch.cat((embeddings, context), dim=1), (h_i, c_i))
                h_norm = self.decoderRNN.layer_norm(h_new)
                log_probs = F.log_softmax(self.decoderRNN.fc(h_norm), dim=1)
                top_probs, top_idx = log_probs.topk(beam_width, dim=1)
                for i in range(beam_width):
                    new_word_idx = top_idx[0][i]
                    new_score = score + top_probs[0][i].item()
                    new_seq = seq + [new_word_idx.item()]
                    candidates.append((new_score, new_word_idx.unsqueeze(0), h_new, c_new, new_seq))
            beams = sorted(candidates, key=lambda x: x[0], reverse=True)[:beam_width]
            if all(vocabulary.itos[b[4][-1]] == "<EOS>" for b in beams):
                break

        best_seq = beams[0][4]
        result_caption = [vocabulary.itos[idx] for idx in best_seq if vocabulary.itos[idx] not in ("<EOS>", "<SOS>")]
        self.train()
        return result_caption
"""))

nb1.cells.append(md(r"""## 5. Training utilities (from `utils.py`)"""))

nb1.cells.append(code(r"""def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Experiment:
    def __init__(self, name, config, root="experiments"):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.name = f"{timestamp}_{name}"
        self.dir = os.path.join(root, self.name)
        self.weights_dir = os.path.join(self.dir, "weights")
        self.logs_dir = os.path.join(self.dir, "logs")
        os.makedirs(self.weights_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        self.save_config(config)
        print(f"[Experiment] Initialized: {self.dir}")

    def save_config(self, config):
        with open(os.path.join(self.dir, "config.json"), "w") as f:
            json.dump(config, f, indent=4)

    def get_checkpoint_path(self, epoch):
        return os.path.join(self.weights_dir, f"checkpoint_epoch_{epoch}.pth.tar")


def _generate(model, img, vocab, device, use_beam_search=False, beam_width=3):
    if use_beam_search:
        return model.beam_search_caption(img, vocab, beam_width=beam_width, device=device)
    return model.caption_image(img, vocab, device=device)


def print_examples(model, device, dataset, vocab, n=2, use_beam_search=True, beam_width=3):
    model.eval()
    print("\n--- Example Predictions ---")
    indices = np.random.choice(len(dataset), size=min(n, len(dataset)), replace=False)
    for idx in indices:
        img, caption_tensor = dataset[idx]
        img = img.unsqueeze(0).to(device)
        truth = []
        for token_idx in caption_tensor:
            word = vocab.itos[token_idx.item()]
            if word == "<EOS>":
                break
            if word not in ("<SOS>", "<PAD>"):
                truth.append(word)
        with torch.no_grad():
            output = _generate(model, img, vocab, device, use_beam_search, beam_width)
        print(f"Truth: {' '.join(truth)}")
        print(f"Pred : {' '.join(output)}")
        print("---------------------------")
    model.train()


def evaluate_bleu(loader, model, device, vocab, limit_batches=50, use_beam_search=False, beam_width=3):
    from torchmetrics import BLEUScore
    print("=> Calculating BLEU Score...")
    metric = BLEUScore(n_gram=4, smooth=True)
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch_idx, (imgs, captions) in enumerate(tqdm(loader, desc="BLEU Eval", leave=False)):
            if batch_idx >= limit_batches:
                break
            imgs = imgs.to(device)
            for i in range(imgs.shape[0]):
                img = imgs[i].unsqueeze(0)
                generated = _generate(model, img, vocab, device, use_beam_search, beam_width)
                preds.append(" ".join(generated))
                truth_words = []
                for token_idx in captions[i]:
                    word = vocab.itos[token_idx.item()]
                    if word == "<EOS>":
                        break
                    if word not in ("<SOS>", "<PAD>"):
                        truth_words.append(word)
                targets.append([" ".join(truth_words)])
    score = metric(preds, targets)
    print(f"=> BLEU-4 Score: {score.item():.4f}")
    model.train()
    return score.item()


def get_attention_map(model, image_tensor, vocab, device, max_length=50):
    model.eval()
    with torch.no_grad():
        encoder_out = model.encoderCNN(image_tensor)
        num_pixels = encoder_out.size(1)
        grid_size = int(num_pixels ** 0.5)
        h, c = model.decoderRNN.init_hidden_state(encoder_out)
        word = torch.tensor([vocab.stoi["<SOS>"]], device=device)
        words, alphas = [], []
        for _ in range(max_length):
            embeddings = model.decoderRNN.embed(word)
            context, alpha = model.decoderRNN.attention(encoder_out, h)
            h, c = model.decoderRNN.lstm_cell(torch.cat((embeddings, context), dim=1), (h, c))
            h_norm = model.decoderRNN.layer_norm(h)
            preds = model.decoderRNN.fc(h_norm)
            predicted = preds.argmax(dim=1)
            token = vocab.itos[predicted.item()]
            if token == "<EOS>":
                break
            if token != "<SOS>":
                words.append(token)
                alphas.append(alpha.view(grid_size, grid_size).cpu())
            word = predicted
    model.train()
    return words, alphas
"""))

nb1.cells.append(md(r"""## 6. Training loop (from `train.py`)

Config defaults below are tuned for a **free-tier Colab GPU** session — see the time budget note at the top of this notebook. Feel free to bump `num_epochs` back up if you have Colab Pro / more session time; early stopping (`patience`) will cut things off automatically once val loss stops improving either way."""))

nb1.cells.append(code(r"""def get_teacher_forcing_ratio(epoch, num_epochs, start=1.0, end=0.7):
    if num_epochs <= 1:
        return start
    progress = min(epoch / (num_epochs - 1), 1.0)
    return start + (end - start) * progress


def run_epoch(model, loader, criterion, device, vocab, optimizer=None,
              teacher_forcing_ratio=1.0, grad_clip=None,
              doubly_stochastic_lambda=0.0, writer=None, epoch=0,
              step_offset=0, tqdm_disable=False):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    loop = loader if tqdm_disable else tqdm(loader, leave=True)
    total_loss = 0.0
    step = step_offset

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for imgs, captions in loop:
            imgs = imgs.to(device)
            captions = captions.to(device)

            predictions, alphas = model(imgs, captions,
                teacher_forcing_ratio=teacher_forcing_ratio if is_train else 1.0)

            targets = captions[:, 1:]
            loss = criterion(predictions.reshape(-1, predictions.shape[-1]), targets.reshape(-1))

            if doubly_stochastic_lambda > 0:
                loss = loss + doubly_stochastic_lambda * ((1. - alphas.sum(dim=1)) ** 2).mean()

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                if grad_clip is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                if writer is not None:
                    writer.add_scalar("Training/Batch_Loss", loss.item(), step)
                step += 1

            total_loss += loss.item()
            if not tqdm_disable:
                loop.set_description(f"Epoch [{epoch + 1}]" + (" Train" if is_train else " Val"))
                loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(loader)
    return avg_loss, step


def train(config=None):
    default_config = {
        "experiment_name": "ResNet101_Attn_LSTM_Colab",
        "seed": 42,
        "learning_rate": 3e-4,
        "batch_size": 32,
        "embed_size": 256,
        "hidden_size": 512,
        "attention_dim": 256,
        "dropout": 0.5,
        "num_epochs": 20,          # reduced from 100 for a Colab session
        "save_model": True,
        "num_workers": 2,          # Colab CPU cores are limited
        "freq_threshold": 5,
        "teacher_forcing_start": 1.0,
        "teacher_forcing_end": 0.7,
        "unfreeze_encoder_epoch": 12,
        "grad_clip": 5.0,
        "doubly_stochastic_lambda": 1.0,
        "optimizer": "Adam",
        "patience": 5,
        "bleu_every_n_epochs": 5,
        "load_model": False,
        "checkpoint_path": None,
        "tqdm_disable": False,
        "data_root": "/content/caption_data/Images",
        "annotation_file": "/content/caption_data/captions.txt",
    }
    if config:
        for key, value in config.items():
            default_config[key] = value
    config = default_config

    seed_everything(config["seed"])
    experiment = Experiment(config["experiment_name"], config)
    writer = SummaryWriter(log_dir=experiment.logs_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading data...")
    train_transform = get_transforms(train=True)
    val_transform = get_transforms(train=False)

    train_loader, val_loader, test_loader, vocab = get_loaders(
        root_folder=config["data_root"],
        annotation_file=config["annotation_file"],
        train_transform=train_transform,
        val_transform=val_transform,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        freq_threshold=config["freq_threshold"],
    )
    torch.save(vocab, os.path.join(experiment.dir, "vocab.pth"))
    print(f"Vocab size: {len(vocab)}")

    model = CNNtoRNN(
        embed_size=config["embed_size"], hidden_size=config["hidden_size"],
        vocab_size=len(vocab), attention_dim=config["attention_dim"],
        dropout=config["dropout"], fine_tune_encoder=False,
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi["<PAD>"])

    if config["optimizer"] == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    elif config["optimizer"] == "Adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=config["learning_rate"])
    elif config["optimizer"] == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=config["learning_rate"])
    else:
        print(f"Warning: Unknown optimizer {config['optimizer']}, defaulting to Adam.")
        optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    step = 0
    start_epoch = 0

    if config.get("load_model", False) and config.get("checkpoint_path"):
        if os.path.exists(config["checkpoint_path"]):
            checkpoint = torch.load(config["checkpoint_path"], map_location=device)
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            start_epoch = checkpoint["epoch"] + 1
            print(f"=> Resuming from Epoch {start_epoch}")
        else:
            print(f"=> Checkpoint {config['checkpoint_path']} not found, starting fresh.")

    best_val_loss = float("inf")
    patience_counter = 0
    best_bleu_score = 0.0
    encoder_unfrozen = False
    history = {"train_loss": [], "val_loss": [], "bleu": {}}

    for epoch in range(start_epoch, config["num_epochs"]):
        if (config["unfreeze_encoder_epoch"] is not None
                and epoch >= config["unfreeze_encoder_epoch"] and not encoder_unfrozen):
            print(f"=> Unfreezing encoder's last conv block at epoch {epoch + 1}")
            model.unfreeze_encoder()
            encoder_unfrozen = True

        tf_ratio = get_teacher_forcing_ratio(epoch, config["num_epochs"],
            start=config["teacher_forcing_start"], end=config["teacher_forcing_end"])

        print(f"Epoch [{epoch + 1}/{config['num_epochs']}] (teacher_forcing={tf_ratio:.2f}) Training...")
        avg_train_loss, step = run_epoch(model, train_loader, criterion, device, vocab,
            optimizer=optimizer, teacher_forcing_ratio=tf_ratio, grad_clip=config["grad_clip"],
            doubly_stochastic_lambda=config["doubly_stochastic_lambda"], writer=writer,
            epoch=epoch, step_offset=step, tqdm_disable=config["tqdm_disable"])
        print(f"Average Train Loss: {avg_train_loss:.4f}")
        writer.add_scalar("Training/Epoch_Loss", avg_train_loss, epoch)

        print(f"Epoch [{epoch + 1}/{config['num_epochs']}] Validation...")
        avg_val_loss, _ = run_epoch(model, val_loader, criterion, device, vocab, optimizer=None,
            doubly_stochastic_lambda=config["doubly_stochastic_lambda"], epoch=epoch,
            tqdm_disable=config["tqdm_disable"])
        print(f"Average Val Loss: {avg_val_loss:.4f}")
        writer.add_scalar("Validation/Epoch_Loss", avg_val_loss, epoch)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)

        if (epoch + 1) % config["bleu_every_n_epochs"] == 0:
            print("Running BLEU Evaluation...")
            bleu_score = evaluate_bleu(val_loader, model, device, vocab)
            writer.add_scalar("Validation/BLEU_Score", bleu_score, epoch)
            best_bleu_score = max(best_bleu_score, bleu_score)
            history["bleu"][epoch] = bleu_score

        checkpoint = {"epoch": epoch, "state_dict": model.state_dict(),
                      "optimizer": optimizer.state_dict(), "vocab": vocab, "config": config}

        if config["save_model"]:
            torch.save(checkpoint, experiment.get_checkpoint_path(epoch))

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(checkpoint, os.path.join(experiment.weights_dir, "best_model.pth.tar"))
            print("New Best Model Saved!")
        else:
            patience_counter += 1
            print(f"Early Stopping Counter: {patience_counter}/{config['patience']}")
            if patience_counter >= config["patience"]:
                print("Early Stopping Triggered. Stopping Training.")
                break

    print(f"Training Complete. Best Val Loss: {best_val_loss:.4f}, Best BLEU: {best_bleu_score:.2f}")
    return {"best_val_loss": best_val_loss, "best_bleu": best_bleu_score,
            "experiment_dir": experiment.dir, "history": history, "model": model,
            "vocab": vocab, "val_loader": val_loader, "test_loader": test_loader}
"""))

nb1.cells.append(md(r"""## 7. Run training

This is the cell that actually trains. Adjust `num_epochs` / `batch_size` here if you want to experiment — everything else falls back to the defaults above."""))

nb1.cells.append(code(r"""results = train({
    "num_epochs": 20,
    "batch_size": 32,
    "num_workers": 2,
    "patience": 5,
    "bleu_every_n_epochs": 5,
})
"""))

nb1.cells.append(md(r"""## 8. Plot loss curves"""))

nb1.cells.append(code(r"""import matplotlib.pyplot as plt

history = results["history"]
plt.figure(figsize=(8, 5))
plt.plot(history["train_loss"], label="Train Loss")
plt.plot(history["val_loss"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training / Validation Loss")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

if history["bleu"]:
    epochs, scores = zip(*sorted(history["bleu"].items()))
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, scores, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("BLEU-4")
    plt.title("Validation BLEU-4 over training")
    plt.grid(alpha=0.3)
    plt.show()
"""))

nb1.cells.append(md(r"""## 9. Qualitative check + save artifacts to Drive

Prints a couple of example predictions, then zips the whole experiment folder (checkpoints + vocab + config + logs) and copies it to Drive so it survives past this Colab session — you'll need it for `inference.ipynb`."""))

nb1.cells.append(code(r"""device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print_examples(results["model"], device, results["val_loader"].dataset, results["vocab"], n=3)
"""))

nb1.cells.append(code(r"""import shutil

exp_dir = results["experiment_dir"]
zip_base = exp_dir.rstrip("/")
zip_path = shutil.make_archive(zip_base, "zip", exp_dir)
print("Zipped to:", zip_path)

# Copy to Drive (mounts if not already mounted)
if not os.path.exists("/content/drive"):
    from google.colab import drive
    drive.mount('/content/drive')

drive_dest = "/content/drive/MyDrive/" + os.path.basename(zip_path)
shutil.copy(zip_path, drive_dest)
print("Saved experiment artifacts to:", drive_dest)
print("\nCopy this Drive path — you'll set it as CHECKPOINT_ZIP in inference.ipynb.")
"""))

nb1.cells.append(md(r"""---
### Next step
Open **`inference.ipynb`**, set the path to the zip you just saved to Drive (or directly to `best_model.pth.tar` if you keep the experiment folder around), and run it to demonstrate + analyze the model on unseen test images."""))

with open("data_and_training.ipynb", "w") as f:
    nbf.write(nb1, f)

print("Wrote data_and_training.ipynb with", len(nb1.cells), "cells")