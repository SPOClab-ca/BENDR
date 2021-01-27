import copy
import mne
import parse
import tqdm

import torch
import torch.nn.functional as F
import numpy as np

from torch import nn
from math import ceil
from pathlib import Path

from dn3.trainable.processes import StandardClassification, BaseProcess
from dn3.trainable.models import StrideClassifier, Classifier
from dn3.trainable.layers import Flatten, Permute
from dn3.utils import DN3ConfigException


class LinearHeadBENDR(Classifier):

    @property
    def num_features_for_classification(self):
        return self.encoder_h * self.pool_length

    def features_forward(self, x):
        x = self.encoder(x)
        x = self.enc_augment(x)
        x = self.summarizer(x)
        return self.extended_classifier(x)

    def __init__(self, targets, samples, channels, encoder_h=512, projection_head=False,
                 enc_do=0.1, feat_do=0.4, pool_length=4, mask_p_t=0.01, mask_p_c=0.005, mask_t_span=0.05,
                 mask_c_span=0.1, classifier_layers=1):
        if classifier_layers < 1:
            self.pool_length = pool_length
            self.encoder_h = 3 * encoder_h
        else:
            self.pool_length = pool_length // classifier_layers
            self.encoder_h = encoder_h
        super().__init__(targets, samples, channels)

        self.encoder = ConvEncoderBENDR(channels, encoder_h=encoder_h, projection_head=projection_head, dropout=enc_do)
        encoded_samples = self.encoder.downsampling_factor(samples)

        mask_t_span = mask_t_span if mask_t_span > 1 else int(mask_t_span * encoded_samples)
        # Important for short things like P300
        mask_t_span = 0 if encoded_samples < 2 else mask_t_span
        mask_c_span = mask_c_span if mask_c_span > 1 else int(mask_c_span * encoder_h)

        self.enc_augment = EncodingAugment(encoder_h, mask_p_t, mask_p_c, mask_c_span=mask_c_span,
                                           mask_t_span=mask_t_span)
        tqdm.tqdm.write(self.encoder.description(None, samples) + " | {} pooled".format(pool_length))
        self.summarizer = nn.AdaptiveAvgPool1d(pool_length)

        classifier_layers = [self.encoder_h * self.pool_length for i in range(classifier_layers)] if \
            not isinstance(classifier_layers, (tuple, list)) else classifier_layers
        classifier_layers.insert(0, 3 * encoder_h * pool_length)
        self.extended_classifier = nn.Sequential(Flatten())
        for i in range(1, len(classifier_layers)):
            self.extended_classifier.add_module("ext-classifier-{}".format(i), nn.Sequential(
                nn.Linear(classifier_layers[i - 1], classifier_layers[i]),
                nn.Dropout(feat_do),
                nn.ReLU(),
                nn.BatchNorm1d(classifier_layers[i]),
            ))

    def load_encoder(self, encoder_file, freeze=False, strict=True):
        self.encoder.load(encoder_file, strict=strict)
        self.encoder.freeze_features(not freeze)
        print("Loaded {}".format(encoder_file))

    def load_pretrained_modules(self, encoder_file, contextualizer_file, strict=False, freeze_encoder=True):
        self.load_encoder(encoder_file, strict=strict, freeze=freeze_encoder)
        self.enc_augment.init_from_contextualizer(contextualizer_file)


class BENDRClassification(Classifier):

    @property
    def num_features_for_classification(self):
        return self.encoder_h

    def features_forward(self, *x):
        encoded = self.encoder(x[0])

        if self.trial_embeddings is not None and len(x) > 1:
            embeddings = self.trial_embeddings(x[-1])
            encoded += embeddings.unsqueeze(-1).expand_as(encoded)

        context = self.contextualizer(encoded)
        # return self.projection_mlp(context[:, :, 0])
        # return nn.functional.adaptive_max_pool1d(context, output_size=1)
        return context[:, :, -1]

    def __init__(self, targets, samples, channels, encoder_h=512, contextualizer_hidden=3076, projection_head=False,
                 new_projection_layers=0, dropout=0., trial_embeddings=None, layer_drop=0, keep_layers=None,
                 mask_p_t=0.01, mask_p_c=0.005, mask_t_span=0.1, mask_c_span=0.1, multi_gpu=False):
        self.encoder_h = encoder_h
        self.contextualizer_hidden = contextualizer_hidden
        super().__init__(targets, samples, channels)

        encoder = ConvEncoderBENDR(channels, encoder_h=encoder_h, dropout=dropout, projection_head=projection_head)
        encoded_samples = encoder.downsampling_factor(samples)

        mask_t_span = mask_t_span if mask_t_span > 1 else int(mask_t_span * encoded_samples)
        mask_c_span = mask_c_span if mask_c_span > 1 else int(mask_c_span * encoder_h)
        contextualizer = BENDRContextualizer(encoder_h, hidden_feedforward=contextualizer_hidden, finetuning=True,
                                                  mask_p_t=mask_p_t, mask_p_c=mask_p_c, layer_drop=layer_drop,
                                                  mask_c_span=mask_c_span, dropout=dropout,
                                                  mask_t_span=mask_t_span)

        self.encoder = nn.DataParallel(encoder) if multi_gpu else encoder
        self.contextualizer = nn.DataParallel(contextualizer) if multi_gpu else contextualizer

        tqdm.tqdm.write(encoder.description(sequence_len=samples))

        self.projection_mlp = nn.Sequential()
        for p in range(1, new_projection_layers + 1):
            self.projection_mlp.add_module("projection-{}".format(p), nn.Sequential(
                nn.Linear(encoder_h, encoder_h),
                nn.Dropout(dropout),
                nn.BatchNorm1d(encoder_h),
                nn.GELU(),
            ))
        self.trial_embeddings = nn.Embedding(trial_embeddings, encoder_h, scale_grad_by_freq=True) \
            if trial_embeddings is not None else trial_embeddings

    def load_encoder(self, encoder_file, freeze=False, strict=True):
        self.encoder.load(encoder_file, strict=strict)
        self.encoder.freeze_features(unfreeze=not freeze)

    def load_contextualizer(self, contextualizer_file, freeze=False, strict=True):
        self.contextualizer.load(contextualizer_file, strict=strict)
        self.contextualizer.freeze_features(unfreeze=not freeze)

    def load_pretrained_modules(self, encoder_file, contextualizer_file, freeze_encoder=False,
                                freeze_contextualizer=False, freeze_position_conv=False,
                                freeze_mask_replacement=True, strict=False):
        self.load_encoder(encoder_file, freeze=freeze_encoder, strict=strict)
        self.load_contextualizer(contextualizer_file, freeze=freeze_contextualizer, strict=strict)
        self.contextualizer.mask_replacement.requires_grad = freeze_mask_replacement
        if freeze_position_conv:
            for p in self.contextualizer.relative_position.parameters():
                p.requires_grad = False


class RefinedBENDR(StrideClassifier):

    @property
    def num_features_for_classification(self):
        return self.encoder_h

    def features_forward(self, *x):
        encoded = self.encoder(x[0])

        if self.trial_embeddings is not None and len(x) > 1:
            embeddings = self.trial_embeddings(x[-1])
            encoded += embeddings.unsqueeze(-1).expand_as(encoded)

        context = self.contextualizer(encoded)
        return self.projection_mlp(context)

    def __init__(self, targets, samples, channels, encoder_h=768, contextualizer_hidden=1024, projection_head=True,
                 new_projection_layers=2, dropout=0.1, trial_embeddings=None, stride_width=4,
                 mask_p_t=0.05, mask_p_c=0.005, mask_c_span=0.1, mask_t_span=0.25):
        self.encoder_h = encoder_h
        self.contextualizer_hidden = contextualizer_hidden
        super().__init__(targets, samples, channels, stride_width=stride_width)
        self.encoder = ConvEncoderBENDR(channels, encoder_h=encoder_h, dropout=dropout, projection_head=projection_head)
        encoded_samples = self.encoder.downsampling_factor(samples)
        self.contextualizer = BENDRContextualizer(encoder_h, hidden_feedforward=contextualizer_hidden, finetuning=True,
                                                  mask_p_t=mask_p_t, mask_p_c=mask_p_c,
                                                  mask_c_span=int(mask_c_span * encoder_h),
                                                  mask_t_span=int(mask_t_span * encoded_samples))
        tqdm.tqdm.write(self.encoder.description(sequence_len=samples))

        self.projection_mlp = nn.Sequential()
        for p in range(1, new_projection_layers + 1):
            self.projection_mlp.add_module("projection-{}".format(p), nn.Sequential(
                nn.Conv1d(encoder_h, encoder_h, 1),
                nn.Dropout2d(dropout),
                nn.BatchNorm1d(encoder_h),
                nn.GELU(),
            ))
        self.trial_embeddings = nn.Embedding(trial_embeddings, encoder_h, scale_grad_by_freq=True) \
            if trial_embeddings is not None else trial_embeddings

    def load_encoder(self, encoder_file, freeze=False, strict=True):
        self.encoder.load(encoder_file, strict=strict)
        self.encoder.freeze_features(unfreeze=not freeze)

    def load_contextualizer(self, contextualizer_file, freeze=False, strict=True):
        self.contextualizer.load(contextualizer_file, strict=strict)
        self.contextualizer.freeze_features(unfreeze=not freeze)
        self.contextualizer.mask_replacement.requires_grad = False
        for p in self.contextualizer.relative_position.parameters():
            p.requires_grad = False

    def load_pretrained_modules(self, encoder_file, contextualizer_file, freeze_encoder=True,
                                freeze_contextualizer=False, strict=True):
        self.load_encoder(encoder_file, freeze=freeze_encoder, strict=strict)
        self.load_contextualizer(contextualizer_file, freeze=freeze_contextualizer, strict=strict)


def _make_span_from_seeds(seeds, span, total=None):
    inds = list()
    for seed in seeds:
        for i in range(seed, seed + span):
            if total is not None and i >= total:
                break
            elif i not in inds:
                inds.append(int(i))
    return np.array(inds)


def _make_mask(shape, p, total, span, allow_no_inds=False):
    # num_mask_spans = np.sum(np.random.rand(total) < p)
    # num_mask_spans = int(p * total)
    mask = torch.zeros(shape, requires_grad=False, dtype=torch.bool)

    for i in range(shape[0]):
        mask_seeds = list()
        while not allow_no_inds and len(mask_seeds) == 0 and p > 0:
            mask_seeds = np.nonzero(np.random.rand(total) < p)[0]

        mask[i, _make_span_from_seeds(mask_seeds, span, total=total)] = True

    return mask


class BendingCollegeWav2Vec(BaseProcess):
    """
    A more wav2vec 2.0 style of constrastive self-supervision, more inspired-by than exactly like it.
    """
    def __init__(self, encoder, context_fn, mask_rate=0.1, mask_span=6, learning_rate=0.01, temp=0.5,
                 permuted_encodings=False, permuted_contexts=False, enc_feat_l2=0.001, multi_gpu=False,
                 l2_weight_decay=1e-4, unmasked_negative_frac=0.25, encoder_grad_frac=1.0,
                 num_negatives=100, **kwargs):
        self.predict_length = mask_span
        self._enc_downsample = encoder.downsampling_factor
        if multi_gpu:
            encoder = nn.DataParallel(encoder)
            context_fn = nn.DataParallel(context_fn)
        if encoder_grad_frac < 1:
            encoder.register_backward_hook(lambda module, in_grad, out_grad:
                                           tuple(encoder_grad_frac * ig for ig in in_grad))
        super(BendingCollegeWav2Vec, self).__init__(encoder=encoder, context_fn=context_fn,
                                                    loss_fn=nn.CrossEntropyLoss(), lr=learning_rate,
                                                    l2_weight_decay=l2_weight_decay,
                                                    metrics=dict(Accuracy=self._contrastive_accuracy,
                                                                 Mask_pct=self._mask_pct), **kwargs)
        self.best_metric = None
        self.mask_rate = mask_rate
        self.mask_span = mask_span
        self.temp = temp
        self.permuted_encodings = permuted_encodings
        self.permuted_contexts = permuted_contexts
        self.beta = enc_feat_l2
        self.start_token = getattr(context_fn, 'start_token', None)
        self.unmasked_negative_frac = unmasked_negative_frac
        self.num_negatives = num_negatives

    def description(self, sequence_len):
        encoded_samples = self._enc_downsample(sequence_len)
        desc = "{} samples | mask span of {} at a rate of {} => E[masked] ~= {}".format(
            encoded_samples, self.mask_span, self.mask_rate,
            int(encoded_samples * self.mask_rate * self.mask_span))
        return desc

    def _generate_negatives(self, z):
        """Generate negative samples to compare each sequence location against"""
        batch_size, feat, full_len = z.shape
        z_k = z.permute([0, 2, 1]).reshape(-1, feat)
        with torch.no_grad():
            # candidates = torch.arange(full_len).unsqueeze(-1).expand(-1, self.num_negatives).flatten()
            negative_inds = torch.randint(0, full_len-1, size=(batch_size, full_len * self.num_negatives))
            # From wav2vec 2.0 implementation, I don't understand
            # negative_inds[negative_inds >= candidates] += 1

            for i in range(1, batch_size):
                negative_inds[i] += i * full_len

        z_k = z_k[negative_inds.view(-1)].view(batch_size, full_len, self.num_negatives, feat)
        return z_k, negative_inds

    def _calculate_similarity(self, z, c, negatives):
        c = c[..., 1:].permute([0, 2, 1]).unsqueeze(-2)
        z = z.permute([0, 2, 1]).unsqueeze(-2)

        # In case the contextualizer matches exactly, need to avoid divide by zero errors
        negative_in_target = (c == negatives).all(-1)
        targets = torch.cat([c, negatives], dim=-2)

        logits = F.cosine_similarity(z, targets, dim=-1) / self.temp
        if negative_in_target.any():
            logits[1:][negative_in_target] = float("-inf")

        return logits.view(-1, logits.shape[-1])

    def forward(self, *inputs):
        z = self.encoder(inputs[0])

        if self.permuted_encodings:
            z = z.permute([1, 2, 0])

        unmasked_z = z.clone()

        batch_size, feat, samples = z.shape

        if self._training:
            mask = _make_mask((batch_size, samples), self.mask_rate, samples, self.mask_span)
        else:
            mask = torch.zeros((batch_size, samples), requires_grad=False, dtype=torch.bool)
            half_avg_num_seeds = max(1, int(samples * self.mask_rate * 0.5))
            if samples <= self.mask_span * half_avg_num_seeds:
                raise ValueError("Masking the entire span, pointless.")
            mask[:, _make_span_from_seeds((samples // half_avg_num_seeds) * np.arange(half_avg_num_seeds).astype(int),
                                              self.mask_span)] = True

        c = self.context_fn(z, mask)

        # Select negative candidates and generate labels for which are correct labels
        negatives, negative_inds = self._generate_negatives(z)

        # Prediction -> batch_size x predict_length x predict_length
        logits = self._calculate_similarity(unmasked_z, c, negatives)
        return logits, z, mask

    @staticmethod
    def _mask_pct(inputs, outputs):
        return outputs[2].float().mean().item()

    @staticmethod
    def _contrastive_accuracy(inputs, outputs):
        logits = outputs[0]
        labels = torch.zeros(logits.shape[0], device=logits.device, dtype=torch.long)
        return StandardClassification._simple_accuracy([labels], logits)

    def calculate_loss(self, inputs, outputs):
        logits = outputs[0]
        labels = torch.zeros(logits.shape[0], device=logits.device, dtype=torch.long)
        # Note the loss_fn here integrates the softmax as per the normal classification pipeline (leveraging logsumexp)
        return self.loss_fn(logits, labels) + self.beta * outputs[1].pow(2).mean()


class _BENDREncoder(nn.Module):
    def __init__(self, in_features, encoder_h=256,):
        super().__init__()
        self.in_features = in_features
        self.encoder_h = encoder_h

    def load(self, filename, strict=True):
        state_dict = torch.load(filename)
        self.load_state_dict(state_dict, strict=strict)

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def freeze_features(self, unfreeze=False):
        for param in self.parameters():
            param.requires_grad = unfreeze


class ConvEncoderBENDR(_BENDREncoder):
    def __init__(self, in_features, encoder_h=256, enc_width=(3, 2, 2, 2, 2, 2),
                 dropout=0., projection_head=False, enc_downsample=(3, 2, 2, 2, 2, 2)):
        super().__init__(in_features, encoder_h)
        self.encoder_h = encoder_h
        if not isinstance(enc_width, (list, tuple)):
            enc_width = [enc_width]
        if not isinstance(enc_downsample, (list, tuple)):
            enc_downsample = [enc_downsample]
        assert len(enc_downsample) == len(enc_width)

        # Centerable convolutions make life simpler
        enc_width = [e if e % 2 else e+1 for e in enc_width]
        self._downsampling = enc_downsample
        self._width = enc_width

        self.encoder = nn.Sequential()
        for i, (width, downsample) in enumerate(zip(enc_width, enc_downsample)):
            self.encoder.add_module("Encoder_{}".format(i), nn.Sequential(
                nn.Conv1d(in_features, encoder_h, width, stride=downsample, padding=width // 2),
                nn.Dropout2d(dropout),
                nn.GroupNorm(encoder_h // 2, encoder_h),
                nn.GELU(),
            ))
            in_features = encoder_h

        if projection_head:
            self.encoder.add_module("projection-1", nn.Sequential(
                nn.Conv1d(in_features, in_features, 1),
                nn.Dropout2d(dropout*2),
                nn.GroupNorm(in_features // 2, in_features),
                nn.GELU()
            ))

    def description(self, sfreq=None, sequence_len=None):
        widths = list(reversed(self._width))[1:]
        strides = list(reversed(self._downsampling))[1:]

        rf = self._width[-1]
        for w, s in zip(widths, strides):
            rf = rf if w == 1 else (rf - 1) * s + 2 * (w // 2)

        desc = "Receptive field: {} samples".format(rf)
        if sfreq is not None:
            desc += ", {:.2f} seconds".format(rf / sfreq)

        ds_factor = np.prod(self._downsampling)
        desc += " | Downsampled by {}".format(ds_factor)
        if sfreq is not None:
            desc += ", new sfreq: {:.2f} Hz".format(sfreq / ds_factor)
        desc += " | Overlap of {} samples".format(rf - ds_factor)
        if sequence_len is not None:
            desc += " | {} encoded samples/trial".format(sequence_len // ds_factor)
        return desc

    def downsampling_factor(self, samples):
        for factor in self._downsampling:
            samples = ceil(samples / factor)
        return samples

    def forward(self, x):
        return self.encoder(x)


# FIXME this is redundant with part of the contextualizer
class EncodingAugment(nn.Module):
    def __init__(self, in_features, mask_p_t=0.1, mask_p_c=0.01, mask_t_span=6, mask_c_span=64, dropout=0.1,
                 position_encoder=25):
        super().__init__()
        self.mask_replacement = torch.nn.Parameter(torch.zeros(in_features), requires_grad=True)
        self.p_t = mask_p_t
        self.p_c = mask_p_c
        self.mask_t_span = mask_t_span
        self.mask_c_span = mask_c_span
        transformer_dim = 3 * in_features

        conv = nn.Conv1d(in_features, in_features, position_encoder, padding=position_encoder // 2, groups=16)
        nn.init.normal_(conv.weight, mean=0, std=2 / transformer_dim)
        nn.init.constant_(conv.bias, 0)
        conv = nn.utils.weight_norm(conv, dim=2)
        self.relative_position = nn.Sequential(conv, nn.GELU())

        self.input_conditioning = nn.Sequential(
            Permute([0, 2, 1]),
            nn.LayerNorm(in_features),
            nn.Dropout(dropout),
            Permute([0, 2, 1]),
            nn.Conv1d(in_features, transformer_dim, 1),
        )

    def forward(self, x, mask_t=None, mask_c=None):
        bs, feat, seq = x.shape

        if self.training:
            if mask_t is None and self.p_t > 0 and self.mask_t_span > 0:
                mask_t = _make_mask((bs, seq), self.p_t, x.shape[-1], self.mask_t_span)
            if mask_c is None and self.p_c > 0 and self.mask_c_span > 0:
                mask_c = _make_mask((bs, feat), self.p_c, x.shape[1], self.mask_c_span)

        if mask_t is not None:
            x.transpose(2, 1)[mask_t] = self.mask_replacement
        if mask_c is not None:
            x[mask_c] = 0

        x = self.input_conditioning(x + self.relative_position(x))
        return x

    def init_from_contextualizer(self, filename):
        state_dict = torch.load(filename)
        self.load_state_dict(state_dict, strict=False)
        for param in self.parameters():
            param.requires_grad = False
        print("Initialized mask embedding and position encoder from ", filename)


class _Hax(nn.Module):
    """T-fixup assumes self-attention norms are removed"""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class BENDRContextualizer(nn.Module):

    def __init__(self, in_features, hidden_feedforward=3076, heads=8, layers=8, dropout=0.15, activation='gelu',
                 position_encoder=25, layer_drop=0.0, mask_p_t=0.1, mask_p_c=0.004, mask_t_span=6, mask_c_span=64,
                 start_token=-5, finetuning=False):
        super().__init__()

        self.dropout = dropout
        self.in_features = in_features
        self._transformer_dim = in_features * 3

        encoder = nn.TransformerEncoderLayer(d_model=in_features * 3, nhead=heads, dim_feedforward=hidden_feedforward,
                                             dropout=dropout, activation=activation)
        encoder.norm1 = _Hax()
        encoder.norm2 = _Hax()

        self.norm = nn.LayerNorm(self._transformer_dim)

        # self.norm_layers = nn.ModuleList([copy.deepcopy(norm) for _ in range(layers)])
        self.transformer_layers = nn.ModuleList([copy.deepcopy(encoder) for _ in range(layers)])
        self.layer_drop = layer_drop
        self.p_t = mask_p_t
        self.p_c = mask_p_c
        self.mask_t_span = mask_t_span
        self.mask_c_span = mask_c_span
        self.start_token = start_token
        self.finetuning = finetuning

        # Initialize replacement vector with 0's
        self.mask_replacement = torch.nn.Parameter(torch.normal(0, in_features**(-0.5), size=(in_features,)),
                                                   requires_grad=True)

        self.position_encoder = position_encoder > 0
        if position_encoder:
            conv = nn.Conv1d(in_features, in_features, position_encoder, padding=position_encoder // 2, groups=16)
            nn.init.normal_(conv.weight, mean=0, std=2 / self._transformer_dim)
            nn.init.constant_(conv.bias, 0)
            conv = nn.utils.weight_norm(conv, dim=2)
            self.relative_position = nn.Sequential(conv, nn.GELU())

        self.input_conditioning = nn.Sequential(
            Permute([0, 2, 1]),
            nn.LayerNorm(in_features),
            nn.Dropout(dropout),
            Permute([0, 2, 1]),
            nn.Conv1d(in_features, self._transformer_dim, 1),
            Permute([2, 0, 1]),
        )

        self.output_layer = nn.Conv1d(self._transformer_dim, in_features, 1)
        self.apply(self.init_bert_params)

    def init_bert_params(self, module):
        if isinstance(module, nn.Linear):
            # module.weight.data.normal_(mean=0.0, std=0.02)
            nn.init.xavier_uniform_(module.weight.data)
            if module.bias is not None:
                module.bias.data.zero_()
            # Tfixup
            module.weight.data = 0.67 * len(self.transformer_layers) ** (-0.25) * module.weight.data

        # if isinstance(module, nn.Conv1d):
        #     # std = np.sqrt((4 * (1.0 - self.dropout)) / (self.in_features * self.in_features))
        #     # module.weight.data.normal_(mean=0.0, std=std)
        #     nn.init.xavier_uniform_(module.weight.data)
        #     module.bias.data.zero_()

    def forward(self, x, mask_t=None, mask_c=None):
        bs, feat, seq = x.shape
        if self.training and self.finetuning:
            if mask_t is None and self.p_t > 0:
                mask_t = _make_mask((bs, seq), self.p_t, x.shape[-1], self.mask_t_span)
            if mask_c is None and self.p_c > 0:
                mask_c = _make_mask((bs, feat), self.p_c, x.shape[1], self.mask_c_span)

        if mask_t is not None:
            x.transpose(2, 1)[mask_t] = self.mask_replacement
        if mask_c is not None:
            x[mask_c] = 0

        if self.position_encoder:
            x = x + self.relative_position(x)
        x = self.input_conditioning(x)

        if self.start_token is not None:
            in_token = self.start_token * torch.ones((1, 1, 1), requires_grad=True).to(x.device).expand([-1, *x.shape[1:]])
            x = torch.cat([in_token, x], dim=0)

        for layer in self.transformer_layers:
            if not self.training or torch.rand(1) > self.layer_drop:
                x = layer(x)

        return self.output_layer(x.permute([1, 2, 0]))

    def freeze_features(self, unfreeze=False, finetuning=False):
        for param in self.parameters():
            param.requires_grad = unfreeze
        if self.finetuning or finetuning:
            self.mask_replacement.requires_grad = False

    def load(self, filename, strict=True):
        state_dict = torch.load(filename)
        self.load_state_dict(state_dict, strict=strict)

    def save(self, filename):
        torch.save(self.state_dict(), filename)


class LoaderERPBCI:
    """
    The dataset from https://physionet.org/content/erpbci/1.0.0/ required a customized solution.

    I've put it in an object so that the solution is somewhat self-contained.
    """
    MAX_ACCEPTABLE_FLASHES = 144
    SOA = 0.15
    TOTAL_RUN_TIME_S = int(MAX_ACCEPTABLE_FLASHES * SOA)
    STIM_CHANNEL = 'STI 014'

    @staticmethod
    def _get_target_and_crop(raw):
        target_char = parse.search('#Tgt{}_', raw.annotations[0]['description'])[0]

        # Find the first speller flash (it isn't consistently at the second or even nth index for that matter)
        start_off = 0
        while len(raw.annotations[start_off]['description']) > 6 and start_off < len(raw.annotations):
            start_off += 1
        assert start_off < len(raw.annotations) - 1
        start_t = raw.annotations[start_off]['onset']
        end_t = start_t + LoaderERPBCI.TOTAL_RUN_TIME_S
        # Operates in-place
        raw.crop(start_t, end_t, include_tmax=False)
        return target_char

    @staticmethod
    def _make_blank_stim(raw):
        info = mne.create_info([LoaderERPBCI.STIM_CHANNEL], raw.info['sfreq'], ['stim'])
        stim_raw = mne.io.RawArray(np.zeros((1, len(raw.times))), info)
        raw.add_channels([stim_raw], force_update_info=True)

    @classmethod
    def __call__(cls, path: Path):
        # Data has to be preloaded to add events to it, swap edf for fif if haven't offline processed first
        # run = mne.io.read_raw_edf(str(path), preload=True)
        run = mne.io.read_raw_fif(str(path), preload=True)
        if len(run.annotations) == 0:
            raise DN3ConfigException
        cls._make_blank_stim(run)
        target_letter = cls._get_target_and_crop(run)
        events, occurrences = mne.events_from_annotations(run, lambda a: int(target_letter in a) + 1)
        run.add_events(events, stim_channel=cls.STIM_CHANNEL)
        return run
