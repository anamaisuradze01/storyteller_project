# data_loader.py
import os
import re
import random

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from PIL import Image, UnidentifiedImageError
import torchvision.transforms as transforms


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
        Lowercases and splits punctuation off as its own token. The old
        version only handled '.' and ',' with a literal string replace,
        which left things like "beach!" or "dog's" glued together as a
        single (rare) token -- that inflates <UNK> usage and starves the
        vocabulary of the actual word. A small regex-based split covers the
        common punctuation Flickr8k captions actually contain.
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
            # A handful of images in scraped Flickr-style datasets are
            # missing or corrupted. Rather than crashing the whole training
            # run over one bad file, fall back to a different sample so the
            # DataLoader keeps going -- and let it be visible in logs once,
            # not silently swallowed forever.
            print(f"[FlickrDataset] Skipping unreadable image '{self.imgs[index]}' ({e}); "
                  f"substituting a different sample.")
            fallback_index = random.randrange(len(self.imgs))
            return self.__getitem__(fallback_index)

        if self.transform is not None:
            image = self.transform(image)

        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<EOS>"])

        return image, torch.tensor(numericalized_caption)


def collate_batch_first(batch, pad_idx):
    """
    Pads a batch of (image, caption) pairs. Captions come back batch-first,
    (B, T), matching what the attention decoder in model.py expects --
    avoids needing a .permute() call in the training loop.
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
        batch_size=32,
        num_workers=2,
        shuffle=True,
        pin_memory=True,
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

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=num_workers,
        shuffle=shuffle, pin_memory=pin_memory, collate_fn=collate_fn,
        drop_last=True,  # keeps every training batch a consistent size
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, num_workers=num_workers,
        shuffle=False, pin_memory=pin_memory, collate_fn=collate_fn,
        persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, num_workers=num_workers,
        shuffle=False, pin_memory=pin_memory, collate_fn=collate_fn,
        persistent_workers=num_workers > 0,
    )

    return train_loader, val_loader, test_loader, vocab


if __name__ == "__main__":
    pass