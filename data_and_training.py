import os
import re
import json
import random
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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text):
        """
        Lowercases and splits punctuation off as its own token. A regex-based
        split (rather than a literal '.'/',' replace) covers things like
        "beach!" or "dog's" so punctuation doesn't get glued onto words and
        starve the vocabulary / inflate <UNK> usage.
        """
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
        return [
            self.stoi.get(token, self.stoi["<UNK>"])
            for token in tokenized_text
        ]

class FlickrDataset(Dataset):
    def __init__(self, root_dir, imgs, captions, vocab, transform=None):
        self.root_dir = root_dir
        self.imgs = imgs
        self.captions = captions
        self.vocab = vocab
        self.transform = transform
        self._warned_bad_images = 0

    def __len__(self):
        return len(self.imgs)

    def _load_image(self, index):
        img_path = os.path.join(self.root_dir, self.imgs[index])
        return Image.open(img_path).convert("RGB")

    def __getitem__(self, index):
        # The old version used recursive random fallback for unreadable images.
        # If several paths are bad, that can loop for a long time and make epoch 1
        # look frozen. This bounded fallback tries nearby samples and then fails
        # clearly if the dataset paths are actually broken.
        last_error = None
        for offset in range(len(self.imgs)):
            idx = (index + offset) % len(self.imgs)
            caption = self.captions[idx]
            try:
                image = self._load_image(idx)
                break
            except (FileNotFoundError, UnidentifiedImageError, OSError) as e:
                last_error = e
                if self._warned_bad_images < 5:
                    print(f"[FlickrDataset] Skipping unreadable image '{self.imgs[idx]}' ({e}).")
                    self._warned_bad_images += 1
        else:
            raise RuntimeError(
                "No readable images were found. Check that content/Images and captions.txt "
                f"match correctly. Last error: {last_error}"
            )

        if self.transform is not None:
            image = self.transform(image)

        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<EOS>"])

        return image, torch.tensor(numericalized_caption)

def collate_batch_first(batch, pad_idx):
    """
    Pads a batch of (image, caption) pairs. Captions come back batch-first,
    (B, T), matching what the attention decoder expects -- avoids needing a
    .permute() call in the training loop.
    """
    imgs = torch.stack([item[0] for item in batch], dim=0)
    captions = [item[1] for item in batch]
    padded = pad_sequence(captions, batch_first=True, padding_value=pad_idx)
    return imgs, padded


class CaptionCollate:
    """Picklable wrapper around collate_batch_first (needed for num_workers > 0)."""

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
    """
    Parses the "image,caption" CSV-style file. Also drops captions that
    tokenize to zero words or to something unusually long (a handful of
    scraped datasets have a stray garbled/duplicated line) -- outliers like
    that disproportionately stretch every batch's padding and can distort
    the loss for everyone else in the batch.
    """
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
        print(f"[data_loader] Dropped {dropped} caption(s) that were empty "
              f"or longer than {max_caption_length} tokens.")

    return all_imgs, all_captions


def _split_by_image(all_imgs, val_size, test_size, split_seed=42):
    """
    Splits by unique image id (not by caption row) so that all 5 captions
    for a given image stay in the same split -- otherwise the model could
    see an image's phrasing during training and get evaluated on a
    near-duplicate caption for the same image in val/test.
    """
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

def get_loaders(
        root_folder,
        annotation_file,
        transform=None,
        train_transform=None,
        val_transform=None,
        batch_size=64,
        num_workers=0,
        shuffle=True,
        pin_memory=None,
        test_size=0.1,
        val_size=0.1,
        freq_threshold=5,
        max_caption_length=40,
        split_seed=42,
):
    train_transform = train_transform or transform
    val_transform = val_transform or transform

    all_imgs, all_captions = _read_annotations(annotation_file, max_caption_length)
    train_ids, val_ids, test_ids = _split_by_image(all_imgs, val_size, test_size, split_seed)

    train_imgs, train_caps = [], []
    val_imgs, val_caps = [], []
    test_imgs, test_caps = [], []

    for img, cap in zip(all_imgs, all_captions):
        if img in train_ids:
            train_imgs.append(img)
            train_caps.append(cap)
        elif img in val_ids:
            val_imgs.append(img)
            val_caps.append(cap)
        elif img in test_ids:
            test_imgs.append(img)
            test_caps.append(cap)

    print(f"[data_loader] Split: {len(train_ids)} train / {len(val_ids)} val / "
          f"{len(test_ids)} test images "
          f"({len(train_caps)} / {len(val_caps)} / {len(test_caps)} captions)")

    vocab = Vocabulary(freq_threshold)
    vocab.build_vocabulary(train_caps)  # vocab built ONLY from train captions, avoids leakage
    print(f"[data_loader] Vocabulary size: {len(vocab)}")

    train_dataset = FlickrDataset(root_folder, train_imgs, train_caps, vocab, transform=train_transform)
    val_dataset = FlickrDataset(root_folder, val_imgs, val_caps, vocab, transform=val_transform)
    test_dataset = FlickrDataset(root_folder, test_imgs, test_caps, vocab, transform=val_transform)

    pad_idx = vocab.stoi["<PAD>"]
    collate_fn = CaptionCollate(pad_idx)

    # Notebook/Windows stability fix: num_workers=0 avoids the common epoch-1
    # multiprocessing hang. If you run as a .py script on Linux, you can raise it.
    if num_workers is None:
        num_workers = 0
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    loader_kwargs = dict(
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        persistent_workers=(num_workers > 0),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle,
        drop_last=True,  # keeps every training batch a consistent size
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        **loader_kwargs,
    )

    return train_loader, val_loader, test_loader, vocab

class EncoderCNN(nn.Module):
    """
    CNN encoder that outputs a spatial grid of features instead of a single
    pooled vector, so the decoder can attend to different regions of the
    image at each generation step.
    """

    def __init__(self, encoded_image_size=5, fine_tune=False):
        super(EncoderCNN, self).__init__()
        self.encoded_image_size = encoded_image_size

        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # Drop avgpool + fc, keep everything up to the last conv block
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        # Fixed-size spatial grid regardless of input resolution
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

        self.set_fine_tune(fine_tune)

    def forward(self, images):
        features = self.resnet(images)                      # (B, 2048, H, W)
        features = self.adaptive_pool(features)              # (B, 2048, S, S)
        features = features.permute(0, 2, 3, 1)               # (B, S, S, 2048)
        features = features.view(features.size(0), -1, features.size(-1))  # (B, S*S, 2048)
        return features

    def set_fine_tune(self, fine_tune=False, unfreeze_from_block=7):
        """
        Freeze everything by default. When fine_tune=True, unfreeze only the
        deeper conv blocks (index >= unfreeze_from_block in the Sequential),
        which is usually enough to adapt features without destabilizing
        early training or blowing up compute.
        """
        for p in self.resnet.parameters():
            p.requires_grad = False

        if fine_tune:
            children = list(self.resnet.children())
            for block in children[unfreeze_from_block:]:
                for p in block.parameters():
                    p.requires_grad = True

class Attention(nn.Module):
    """
    Bahdanau-style (concat) attention: the encoder features and the decoder's
    previous hidden state are concatenated, projected, passed through tanh,
    then scored -- a softmax over regions (no learned gate).
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.attn_proj = nn.Linear(encoder_dim + decoder_dim, attention_dim)
        self.score_proj = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, encoder_out, decoder_hidden):
        # encoder_out: (B, L, encoder_dim), decoder_hidden: (B, decoder_dim)
        num_pixels = encoder_out.size(1)
        hidden_expanded = decoder_hidden.unsqueeze(1).expand(-1, num_pixels, -1)
        energy = torch.tanh(self.attn_proj(torch.cat((encoder_out, hidden_expanded), dim=2)))
        scores = self.score_proj(energy).squeeze(2)          # (B, L)
        alpha = F.softmax(scores, dim=1)                     # (B, L)
        context = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (B, encoder_dim)
        return context, alpha

class DecoderRNN(nn.Module):
    """
    LSTMCell-based decoder with attention and scheduled sampling.
    """

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
        """
        encoder_out: (B, L, encoder_dim)
        captions:    (B, T) token ids, including <SOS> ... <EOS>
        Returns predictions of shape (B, T-1, vocab_size) and attention
        weights of shape (B, T-1, L), aligned to predict captions[:, 1:].
        """
        batch_size = encoder_out.size(0)
        device = encoder_out.device
        num_steps = captions.size(1) - 1  # predict all tokens after <SOS>
        num_pixels = encoder_out.size(1)

        h, c = self.init_hidden_state(encoder_out)

        predictions = torch.zeros(batch_size, num_steps, self.vocab_size, device=device)
        alphas = torch.zeros(batch_size, num_steps, num_pixels, device=device)

        # First input is always the ground-truth <SOS> token
        input_word = captions[:, 0]

        for t in range(num_steps):
            embeddings = self.embed_dropout(self.embed(input_word))  # (B, embed_size)
            context, alpha = self.attention(encoder_out, h)

            h, c = self.lstm_cell(torch.cat((embeddings, context), dim=1), (h, c))
            h_norm = self.layer_norm(h)
            preds = self.fc(self.dropout(h_norm))                   # (B, vocab_size)

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
        self.decoderRNN = DecoderRNN(
            embed_size=embed_size,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            encoder_dim=encoder_dim,
            attention_dim=attention_dim,
            dropout=dropout,
        )

    def forward(self, images, captions, teacher_forcing_ratio=1.0):
        """
        captions: (B, T) batch-first token ids.
        Returns predictions (B, T-1, vocab_size) and attention weights for
        optional regularization; compare predictions against captions[:, 1:].
        """
        # Speed/memory fix: while the CNN encoder is frozen, do not build a
        # ResNet autograd graph. This makes epoch 1 much less likely to crawl
        # or run out of memory, especially on CPU.
        encoder_trainable = any(p.requires_grad for p in self.encoderCNN.parameters())
        if encoder_trainable:
            encoder_out = self.encoderCNN(images)
        else:
            self.encoderCNN.eval()
            with torch.no_grad():
                encoder_out = self.encoderCNN(images)
        predictions, alphas = self.decoderRNN(encoder_out, captions, teacher_forcing_ratio)
        return predictions, alphas

    def unfreeze_encoder(self):
        self.encoderCNN.set_fine_tune(fine_tune=True)

    @torch.no_grad()
    def caption_image(self, image, vocabulary, max_length=50, device=None):
        device = device or image.device
        device = device or image.device
        self.eval()
        encoder_out = self.encoderCNN(image)  # (1, L, encoder_dim)

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
    def beam_search_caption(self, image, vocabulary, max_length=50, beam_width=3, device=None):
        device = device or image.device
        self.eval()
        encoder_out = self.encoderCNN(image)  # (1, L, encoder_dim)
        h0, c0 = self.decoderRNN.init_hidden_state(encoder_out)

        # Each beam entry: (score, last_word_idx, h, c, sequence)
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
                h_new, c_new = self.decoderRNN.lstm_cell(
                    torch.cat((embeddings, context), dim=1), (h_i, c_i)
                )
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
        result_caption = [
            vocabulary.itos[idx] for idx in best_seq
            if vocabulary.itos[idx] not in ("<EOS>", "<SOS>")
        ]
        self.train()
        return result_caption

def seed_everything(seed=42):
    """Sets the seed for reproducibility across all libraries."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Experiment:
    """
    Manages experiment directories, configs, and logging.
    experiments/
        2026-01-17_18-30-00_ResNet101_Attn_LSTM/
            config.json
            logs/
            weights/
                checkpoint_epoch_1.pth.tar
                best_model.pth.tar
    """

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
    """Small shared helper so print_examples/evaluate_bleu don't duplicate
    the greedy-vs-beam branching logic."""
    if use_beam_search:
        return model.beam_search_caption(img, vocab, beam_width=beam_width, device=device)
    return model.caption_image(img, vocab, device=device)


def print_examples(model, device, dataset, vocab, n=2, use_beam_search=True, beam_width=3):
    """
    Prints a few example predictions vs. ground truth. Defaults to beam
    search since that's what the inference notebook uses -- keeps the
    qualitative checks seen during training representative of real
    inference output.
    """
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


def evaluate_bleu(loader, model, device, vocab, limit_batches=50,
                   use_beam_search=False, beam_width=3):
    """
    Calculates BLEU-4 over (up to) limit_batches batches from `loader`.
    Compares each prediction against the single caption paired with that
    image in the batch -- not the full 5-reference set -- so treat this as
    a relative, epoch-over-epoch progress signal rather than a paper-grade
    BLEU number.

    use_beam_search=False by default here because beam search is slower and
    this runs every few epochs during training; flip it on for a final, more
    representative evaluation pass (e.g. in the inference notebook) at the
    cost of speed.
    """
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
            # captions are batch-first here: (B, T)
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

def get_teacher_forcing_ratio(epoch, num_epochs, start=1.0, end=0.7):
    """
    Linearly decays teacher forcing from `start` to `end` across training.
    """
    if num_epochs <= 1:
        return start
    progress = min(epoch / (num_epochs - 1), 1.0)
    return start + (end - start) * progress


def run_epoch(model, loader, criterion, device, vocab, optimizer=None,
              teacher_forcing_ratio=1.0, grad_clip=None,
              doubly_stochastic_lambda=0.0, writer=None, epoch=0,
              step_offset=0, tqdm_disable=False, max_batches=None):
    """
    Shared logic for one pass over a loader. If `optimizer` is provided,
    runs in training mode with backprop; otherwise runs a no-grad eval pass.

    `max_batches` is optional and useful for a quick smoke test before a full run.
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_batches = len(loader) if max_batches is None else min(len(loader), max_batches)
    loop = loader if tqdm_disable else tqdm(loader, total=total_batches, leave=True, mininterval=1.0)
    total_loss = 0.0
    step = step_offset
    batches_done = 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch_idx, (imgs, captions) in enumerate(loop):
            if max_batches is not None and batch_idx >= max_batches:
                break

            imgs = imgs.to(device, non_blocking=True)
            captions = captions.to(device, non_blocking=True)

            predictions, alphas = model(
                imgs, captions,
                teacher_forcing_ratio=teacher_forcing_ratio if is_train else 1.0
            )

            targets = captions[:, 1:]
            loss = criterion(
                predictions.reshape(-1, predictions.shape[-1]),
                targets.reshape(-1),
            )

            if doubly_stochastic_lambda > 0:
                loss = loss + doubly_stochastic_lambda * ((1. - alphas.sum(dim=1)) ** 2).mean()

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

                if writer is not None:
                    writer.add_scalar("Training/Batch_Loss", loss.item(), step)
                step += 1

            total_loss += loss.item()
            batches_done += 1

            if not tqdm_disable:
                loop.set_description(f"Epoch [{epoch + 1}]" + (" Train" if is_train else " Val"))
                loop.set_postfix(loss=f"{loss.item():.4f}", batch=f"{batches_done}/{total_batches}")

    if batches_done == 0:
        raise RuntimeError("The DataLoader produced 0 batches. Lower batch_size or check dataset split sizes.")

    avg_loss = total_loss / batches_done
    return avg_loss, step

def train(config=None):
    default_config = {
        "experiment_name": "ResNet50_Attn_LSTM_fixed",
        "seed": 42,
        "learning_rate": 3e-4,
        "batch_size": 16,
        "embed_size": 128,
        "hidden_size": 256,
        "attention_dim": 256,
        "dropout": 0.5,
        "num_epochs": 30,
        "save_model": True,
        "num_workers": 0,  # IMPORTANT: safer for Windows/Jupyter; prevents epoch-1 worker hangs
        "freq_threshold": 5,

        # Teacher forcing schedule
        "teacher_forcing_start": 1.0,
        "teacher_forcing_end": 0.7,

        # Encoder fine-tuning: freeze for the first N epochs, then unfreeze
        # the last conv block. Set to None to never fine-tune the encoder.
        "unfreeze_encoder_epoch": 20,

        # Regularization / stability
        "grad_clip": 5.0,
        "doubly_stochastic_lambda": 1.0,

        "optimizer": "Adam",
        "patience": 5,
        "bleu_every_n_epochs": 10,

        "load_model": False,
        "checkpoint_path": None,

        "tqdm_disable": False,
        "pin_memory": None,
        "max_train_batches": None,  # set e.g. 20 for a quick smoke test
        "max_val_batches": None,
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
    base_dir = os.path.abspath(os.path.join(os.getcwd(), "content"))
    if not os.path.isdir(base_dir):
        base_dir = os.path.abspath(os.path.join(os.getcwd(), "..", "content"))

    root_folder = os.path.join(base_dir, "Images")
    annotation_file = os.path.join(base_dir, "captions.txt")

    if not os.path.isdir(root_folder) or not os.path.exists(annotation_file):
        raise FileNotFoundError(
            f"Expected dataset at '{root_folder}' and '{annotation_file}'. "
            "Update the paths if your folder is named differently."
        )

    train_transform = get_transforms(train=True)
    val_transform = get_transforms(train=False)

    train_loader, val_loader, test_loader, vocab = get_loaders(
        root_folder=root_folder,
        annotation_file=annotation_file,
        train_transform=train_transform,
        val_transform=val_transform,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
        freq_threshold=config["freq_threshold"],
    )
    torch.save(vocab, os.path.join(experiment.dir, "vocab.pth"))
    print(f"Vocab size: {len(vocab)}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")

    # Fail fast before epoch 1 if the dataset/loader has a path or multiprocessing issue.
    sample_imgs, sample_caps = next(iter(train_loader))
    print(f"Sanity batch loaded: images={tuple(sample_imgs.shape)}, captions={tuple(sample_caps.shape)}")

    model = CNNtoRNN(
        embed_size=config["embed_size"],
        hidden_size=config["hidden_size"],
        vocab_size=len(vocab),
        attention_dim=config["attention_dim"],
        dropout=config["dropout"],
        fine_tune_encoder=False,
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
        # Two-phase fine-tuning: unfreeze the encoder's deepest block once
        # the decoder has had time to converge on frozen features.
        if (config["unfreeze_encoder_epoch"] is not None
                and epoch >= config["unfreeze_encoder_epoch"]
                and not encoder_unfrozen):
            print(f"=> Unfreezing encoder's last conv block at epoch {epoch + 1}")
            model.unfreeze_encoder()
            encoder_unfrozen = True

        tf_ratio = get_teacher_forcing_ratio(
            epoch, config["num_epochs"],
            start=config["teacher_forcing_start"],
            end=config["teacher_forcing_end"],
        )

        print(f"Epoch [{epoch + 1}/{config['num_epochs']}] "
              f"(teacher_forcing={tf_ratio:.2f}) Training...")
        avg_train_loss, step = run_epoch(
            model, train_loader, criterion, device, vocab,
            optimizer=optimizer,
            teacher_forcing_ratio=tf_ratio,
            grad_clip=config["grad_clip"],
            doubly_stochastic_lambda=config["doubly_stochastic_lambda"],
            writer=writer, epoch=epoch, step_offset=step,
            tqdm_disable=config["tqdm_disable"],
            max_batches=config["max_train_batches"],
        )
        print(f"Average Train Loss: {avg_train_loss:.4f}")
        writer.add_scalar("Training/Epoch_Loss", avg_train_loss, epoch)

        print(f"Epoch [{epoch + 1}/{config['num_epochs']}] Validation...")
        avg_val_loss, _ = run_epoch(
            model, val_loader, criterion, device, vocab,
            optimizer=None,
            doubly_stochastic_lambda=config["doubly_stochastic_lambda"],
            epoch=epoch, tqdm_disable=config["tqdm_disable"],
            max_batches=config["max_val_batches"],
        )
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

        # --- Save: checkpoint + best model -------------------------------
        checkpoint = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "vocab": vocab,
            "config": config,
        }

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
    return {
        "best_val_loss": best_val_loss,
        "best_bleu": best_bleu_score,
        "experiment_dir": experiment.dir,
        "history": history,
        "model": model,
        "vocab": vocab,
        "val_loader": val_loader,
        "test_loader": test_loader,
    }


if __name__ == "__main__":
    results = train({
        # Stable notebook defaults. On Windows/Jupyter, keep num_workers=0.
        "num_workers": 0,
        "batch_size": 16,

        # For a quick test, uncomment these two lines first:
        # "max_train_batches": 20,
        # "max_val_batches": 5,
        # "num_epochs": 1,
    })
    print(f"Model artifacts saved to: {results['experiment_dir']}")

