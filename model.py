# model.py
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet101_Weights


class EncoderCNN(nn.Module):
    """
    CNN encoder that outputs a spatial grid of features instead of a single
    pooled vector, so the decoder can attend to different regions of the
    image at each generation step.
    """

    def __init__(self, encoded_image_size=8, fine_tune=False):
        super(EncoderCNN, self).__init__()
        self.encoded_image_size = encoded_image_size

        resnet = models.resnet101(weights=ResNet101_Weights.DEFAULT)
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
    then scored. This is a different formulation from a gated additive-sum
    attention -- no learned sigmoid gate here, just a softmax over regions.
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
        Returns predictions (B, T-1, vocab_size), targets to compare against
        (captions[:, 1:]), and attention weights for optional regularization.
        """
        encoder_out = self.encoderCNN(images)
        predictions, alphas = self.decoderRNN(encoder_out, captions, teacher_forcing_ratio)
        return predictions, alphas

    def unfreeze_encoder(self):
        self.encoderCNN.set_fine_tune(fine_tune=True)

    @torch.no_grad()
    def caption_image(self, image, vocabulary, max_length=50, device="cuda"):
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
    def beam_search_caption(self, image, vocabulary, max_length=50, beam_width=3, device="cuda"):
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