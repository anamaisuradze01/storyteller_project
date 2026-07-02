# inference_old.py
import os

import torch
import torchvision.transforms as transforms
from PIL import Image

from model import CNNtoRNN


class CaptionModel:
    """
    Bundles everything generate_caption() needs (the trained model, its
    vocabulary, the matching image transform, and device) behind a single
    object, so the public inference function can keep the exact signature
    required by the assignment: generate_caption(image_path, model).
    """

    def __init__(self, model, vocab, transform, device, use_beam_search=True, beam_width=3):
        self.model = model
        self.vocab = vocab
        self.transform = transform
        self.device = device
        self.use_beam_search = use_beam_search
        self.beam_width = beam_width

    @classmethod
    def from_checkpoint(cls, checkpoint_path, device=None, use_beam_search=True, beam_width=3):
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"=> Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        config = checkpoint["config"]
        vocab = checkpoint["vocab"]

        print(f"   Experiment: {config['experiment_name']}")
        print(f"   Epoch: {checkpoint['epoch']}")
        print(f"   Vocab size: {len(vocab)}")

        model = CNNtoRNN(
            embed_size=config["embed_size"],
            hidden_size=config["hidden_size"],
            vocab_size=len(vocab),
            attention_dim=config.get("attention_dim", 256),
            dropout=config.get("dropout", 0.5),
        ).to(device)

        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        return cls(model, vocab, transform, device, use_beam_search, beam_width)


def generate_caption(image_path: str, model: "CaptionModel") -> str:
    """
    Takes a path to an image and returns a generated caption string.

    `model` is a CaptionModel (see CaptionModel.from_checkpoint), which
    bundles the trained network, vocab, transform, and device.
    """
    image = Image.open(image_path).convert("RGB")
    image_tensor = model.transform(image).unsqueeze(0).to(model.device)

    if model.use_beam_search:
        tokens = model.model.beam_search_caption(
            image_tensor, model.vocab,
            beam_width=model.beam_width, device=model.device
        )
    else:
        tokens = model.model.caption_image(
            image_tensor, model.vocab, device=model.device
        )

    return " ".join(tokens)


def generate_captions_for_dir(image_dir, model, extensions=(".jpg", ".jpeg", ".png")):
    """
    Convenience helper for the inference notebook: runs generate_caption
    over every image in a directory. Returns {filename: caption}.
    Useful for building the "successful vs. failure case" comparison.
    """
    results = {}
    for fname in sorted(os.listdir(image_dir)):
        if fname.lower().endswith(extensions):
            path = os.path.join(image_dir, fname)
            results[fname] = generate_caption(path, model)
    return results


if __name__ == "__main__":
    CHECKPOINT_FILE = "experiments/<your_run>/weights/best_model.pth.tar"
    IMAGE_FILE = "test_image.jpg"

    if not os.path.exists(CHECKPOINT_FILE):
        print(f"Checkpoint not found at {CHECKPOINT_FILE} -- update the path and rerun.")
    else:
        caption_model = CaptionModel.from_checkpoint(CHECKPOINT_FILE)

        if not os.path.exists(IMAGE_FILE):
            print(f"Image {IMAGE_FILE} not found -- update the path and rerun.")
        else:
            caption = generate_caption(IMAGE_FILE, caption_model)
            print("Caption:")
            print(caption)