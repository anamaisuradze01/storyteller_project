# utils.py
import os
import json
import random
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm


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
    search since that's what inference_old.py uses -- keeps the qualitative
    checks you see during training representative of real inference output.
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

    use_beam_search=False by default here because beam search is slower
    and this runs every few epochs during training; flip it on for a
    final, more representative evaluation pass (e.g. in the inference
    notebook) at the cost of speed.
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


def get_attention_map(model, image_tensor, vocab, device, max_length=50):
    """
    Runs greedy decoding while keeping the per-step attention weights, so
    you can visualize which image regions the model looked at for each
    generated word -- useful "success vs. failure case" material for the
    inference notebook's Analysis section, since the new model.py is
    attention-based and this wasn't possible with the old pooled-vector
    encoder.

    Returns (words, alphas) where alphas is a list of (S, S) attention
    grids, one per generated word (S = encoder's spatial grid size).
    """
    model.eval()
    with torch.no_grad():
        encoder_out = model.encoderCNN(image_tensor)  # (1, L, encoder_dim)
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