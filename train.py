# train.py
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data_loader import get_loaders, get_transforms
from model import CNNtoRNN
from utils import evaluate_bleu, Experiment, seed_everything


def get_teacher_forcing_ratio(epoch, num_epochs, start=1.0, end=0.7):
    """
    Linearly decays teacher forcing from `start` to `end` across training.
    Early on, the model leans on ground-truth tokens to learn basic word
    order and vocabulary. Later, it's increasingly forced to condition on
    its own predictions, which better matches what happens at inference
    time and tends to produce more robust captions.
    """
    if num_epochs <= 1:
        return start
    progress = min(epoch / (num_epochs - 1), 1.0)
    return start + (end - start) * progress


def run_epoch(model, loader, criterion, device, vocab, optimizer=None,
              teacher_forcing_ratio=1.0, grad_clip=None,
              doubly_stochastic_lambda=0.0, writer=None, epoch=0,
              step_offset=0, tqdm_disable=False):
    """
    Shared logic for one pass over a loader. If `optimizer` is provided,
    runs in training mode with backprop; otherwise runs a no-grad eval pass.
    Returns (average_loss, next_step).
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    loop = loader if tqdm_disable else tqdm(loader, leave=True)
    total_loss = 0.0
    step = step_offset

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for imgs, captions in loop:
            imgs = imgs.to(device)
            # data_loader's collate_fn now returns captions batch-first,
            # (B, T), matching what the model expects directly.
            captions = captions.to(device)

            predictions, alphas = model(
                imgs, captions,
                teacher_forcing_ratio=teacher_forcing_ratio if is_train else 1.0
            )

            targets = captions[:, 1:]  # everything after <SOS>
            loss = criterion(
                predictions.reshape(-1, predictions.shape[-1]),
                targets.reshape(-1),
            )

            if doubly_stochastic_lambda > 0:
                # Encourages the attention weights for each pixel to sum to
                # ~1 across the whole generated sequence (Show, Attend and
                # Tell regularization) -- keeps attention from collapsing
                # onto a few regions and tends to sharpen captions.
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
        "experiment_name": "ResNet101_Attn_LSTM_v2",
        "seed": 42,
        "learning_rate": 3e-4,
        "batch_size": 32,
        "embed_size": 256,
        "hidden_size": 512,
        "attention_dim": 256,
        "dropout": 0.5,
        "num_epochs": 100,
        "save_model": True,
        "num_workers": 6,
        "freq_threshold": 5,

        # Teacher forcing schedule
        "teacher_forcing_start": 1.0,
        "teacher_forcing_end": 0.7,

        # Encoder fine-tuning: freeze for the first N epochs, then unfreeze
        # the last conv block. Set to None to never fine-tune the encoder.
        "unfreeze_encoder_epoch": 10,

        # Regularization / stability
        "grad_clip": 5.0,
        "doubly_stochastic_lambda": 1.0,

        "optimizer": "Adam",
        "patience": 5,
        "bleu_every_n_epochs": 5,

        "load_model": False,
        "checkpoint_path": None,

        "tqdm_disable": False,
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
        root_folder="caption_data/Images",
        annotation_file="caption_data/captions.txt",
        train_transform=train_transform,
        val_transform=val_transform,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        freq_threshold=config["freq_threshold"],
    )
    torch.save(vocab, os.path.join(experiment.dir, "vocab.pth"))
    print(f"Vocab size: {len(vocab)}")

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
        )
        print(f"Average Train Loss: {avg_train_loss:.4f}")
        writer.add_scalar("Training/Epoch_Loss", avg_train_loss, epoch)

        print(f"Epoch [{epoch + 1}/{config['num_epochs']}] Validation...")
        avg_val_loss, _ = run_epoch(
            model, val_loader, criterion, device, vocab,
            optimizer=None,
            doubly_stochastic_lambda=config["doubly_stochastic_lambda"],
            epoch=epoch, tqdm_disable=config["tqdm_disable"],
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
    train()