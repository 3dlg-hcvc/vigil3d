"""
The code in this file is used to build a data structure which can be used to determine how the loss is computed between
predicted features and text tokens.

Given:
* a feature prediction per point in a 3D space (tensor of size [num_point_groups, feature_dim])
* original sentence (raw string)
* tokenized sentence (list of tokens)
* mapping of word indices (in the original sentence) to semantic label indices in the ground truth
* a ground truth segmentation mask from points to semantic label indices (tensor of size [num_points, ])

We assume currently that there are only two scenarios:
1. There are no token annotations, i.e. we only have the target object in the text.
2. We have annotations for every object referred to in the text (e.g. object detection annotations).

Output:
* a ground truth label of size [num_point_groups, num_tokens] for which [i, j] is 1 if the j-th token corresponds to an 
  object including the i-th point group and 0 otherwise
* a mask of size [num_point_groups, num_tokens] for which [i, j] is 1 if the loss should be computed for the i-th point
    group and the j-th token and 0 otherwise. Any object which has an annotation should have a mask of 1 for the 
    corresponding tokens. Additionally, any clear added tokens, such as punctuation, should have a mask of 1.
"""

import re

import torch
import torch.nn.functional as F

TARGET_TOKEN = "<|enddoftext|>"
NO_OBJ_TOKEN = "<|startoftext|>"
SPECIAL_TOKENS = [TARGET_TOKEN, NO_OBJ_TOKEN]


def split_sentence(text: str) -> list[str]:
    words = re.findall(r"<\|\w+\|>|\d+|[a-z]+|\'[a-z]+|[^\w\s]+", text.lower())
    return ["<|startoftext|>", *[f"{w}</w>" if not re.match(r"<\|\w+\|>", w) else w for w in words], "<|endoftext|>"]


def map_words_to_tokens(words: list[str], tokens: list[str], has_target: bool) -> dict[int, list[int]]:
    word_idx = 0
    start_token_idx = 0
    token_idx = 0
    build_token = ""
    mapping = {}
    while word_idx < len(words) and token_idx < len(tokens):
        build_token += tokens[token_idx]

        if build_token == words[word_idx]:
            mapping[word_idx] = [start_token_idx, token_idx + 1]
            token_idx += 1
            start_token_idx = token_idx
            word_idx += 1
            build_token = ""
        else:
            token_idx += 1

    if build_token != "":
        raise ValueError(f"Could not find a match for all words in the text: {words=}, {tokens=}")

    return mapping


def downsample_label(
    label: torch.Tensor, point_mapping: torch.Tensor, batch_size: int, num_groups: int, group_size: int
) -> torch.Tensor:
    """
    Downsample original segmentation mask label from all points to point groups.

    Any point cluster which has at least one point corresponding to the object of interest will be
    considered as a positive example.

    :param label: original segmentation mask tensor of shape [num_points, ]
    :param point_mapping: indices of points corresponding to each cluster, of shape [batch_size * num_groups * group_size, ]
    :param batch_size: number of batches
    :param num_groups: number of point groups/clusters
    :param group_size: number of points associated with each cluster
    :return: tensor of shape [batch_size, num_groups]
    """
    label = label[point_mapping].view(batch_size, num_groups, group_size).contiguous()
    out = (label.sum(dim=-1) > 0).to(dtype=torch.long)  # hard voting; [batch_size, num_groups]
    return out


def compute_label_and_mask(
    label: torch.Tensor,
    text_features: list[str],
    text: list[str],
    tokens: list[list[str]],
    class_lookup: list[dict[int, int]],
    has_target: list[bool],
    num_groups: int,
    max_text_length: int = 77,
):
    """
    Compute target labels for loss computation.

    Here, we have two considerations:
    1. For the ground truth segmentation label, we need to map our labels for all points to the corresponding
        clusters.
    2. Our text output is per-token, so we need to convert our mapping of words-to-objects to tokens-to-objects.

    :param label: tensor of size [batch_size * num_points, ] corresponding to a segmentation mask over points of
    classes corresponding to different tokens. 0 means no mapping, and 1, etc. refer to each of the objects for
    which we have known annotations relevant to the text.
    :param text: list of strings corresponding to the text input for a batch
    :param tokens: _description_
    :param point_mapping: tensor of size [batch_size * num_groups * group_size, ] corresponding to a flat index of
    points for each cluster
    :param class_lookup: list of mappings of word indices to semantic labels in label
    :return: tuple of
    * labels (0 or 1) for each of the point clusters, corresponding to whehter that point cluster should be usedd in
    loss calculation.
    a positive example (within the GT) for that text token
    * mask (0 or 1) for each of the point clusters, corresponding to whether that point cluster should be considered
    """
    batch_size = len(text)

    # 1. initialize label and mask tensors
    # point_text_label = torch.zeros((batch_size, num_groups, max_text_length), dtype=torch.int, device=label.device)
    # mask = torch.zeros((batch_size, num_groups, max_text_length), dtype=torch.int, device=label.device)
    # sem_idx_to_token_all = []

    text_feature_indices = []
    for batch_idx in range(len(text)):
        cl = {v: k for k, v in class_lookup[batch_idx].items()}
        text_feature_indices.append(batch_idx * max_text_length + cl[0])
        text_feature_indices.append(batch_idx * max_text_length + cl[1])
    text_feature_indices = torch.tensor(text_feature_indices, dtype=torch.int64, device=text_features.device)

    text_features_flat = text_features.view(batch_size * max_text_length, -1)
    text_features_flat = text_features_flat[text_feature_indices, :]
    text_features = text_features_flat.view(batch_size, 2, -1)
    return text_features

    # for batch_idx in range(len(text)):
    #     # 2. construct mapping from words to tokens
    #     words = split_sentence(text[batch_idx])
    #     mapping = map_words_to_tokens(words, tokens[batch_idx], has_target[batch_idx])

    #     # 3. construct label and mask tensors
    #     # As of now, we expect only two options here:
    #     #   A. No object annotations, only target object in text
    #     #   B. Object annotations for all objects in text
    #     cl = class_lookup[batch_idx] if class_lookup[batch_idx] is not None else {}
    #     fully_annotated = len(cl) > (2 if has_target[batch_idx] else 1)
    #     sem_idx_to_token = defaultdict(list)
    #     breakpoint()

    #     for word_idx, [start_token, end_token] in mapping.items():
    #         if words[word_idx] == NO_OBJ_TOKEN:
    #             point_text_label[batch_idx, label[batch_idx] == 0, start_token:end_token] = 1
    #             if fully_annotated:
    #                 mask[batch_idx, :, start_token:end_token] = 1
    #             else:
    #                 mask[batch_idx, :, start_token:end_token] = 1
    #                 # mask[batch_idx, label[batch_idx] != 0, start_token:end_token] = 1

    #             sem_idx_to_token[cl[word_idx]].extend(list(range(start_token, end_token)))
    #         elif word_idx in cl:
    #             point_text_label[batch_idx, label[batch_idx] == cl[word_idx], start_token:end_token] = 1
    #             mask[batch_idx, :, start_token:end_token] = 1
    #             sem_idx_to_token[cl[word_idx]].extend(list(range(start_token, end_token)))
    #         elif re.match(r"[^\w\d\s]", words[word_idx]):  # match added tokens such as punctuation
    #             # enable loss but set all labels to 0
    #             mask[batch_idx, :, start_token:end_token] = 1

    #     sem_idx_to_token_all.append(sem_idx_to_token)

    return point_text_label, mask, sem_idx_to_token_all


# def compute_prediction(raw_pred: torch.Tensor, token_groupings: list[dict[int, list[int]]]) -> torch.Tensor:
def compute_prediction(raw_pred: torch.Tensor) -> torch.Tensor:
    """
    Compute the prediction tensor from the raw prediction tensor and the token groupings.

    :param raw_pred: tensor of size [batch_size, num_point_groups, max_length] representing the cosine alignments
    between features and tokens
    :param token_groupings: list of dictionaries of lists of token indices, where the outermost list corresponds to each sample
    in the batch, and each sublist is a list of token indices corresponding to the same object.
    :return: tensor of size [num_point_groups, len(token_groupings)] for which each value is the index + 1 of the token group
    """
    predictions = torch.argmax(raw_pred, dim=-1)
    probabilities = F.softmax(raw_pred, dim=-1)
    # batch_size = raw_pred.size(0)
    # num_point_groups = raw_pred.size(1)
    # predictions = torch.zeros((batch_size, num_point_groups), dtype=torch.int)
    # probabilities = []
    # for batch_idx in range(batch_size):
    #     prob = torch.zeros((num_point_groups, len(token_groupings[batch_idx]))).type_as(raw_pred)
    #     for sem_idx, token_indices in token_groupings[batch_idx].items():
    #         prob[:, sem_idx] = raw_pred[batch_idx, :, token_indices].mean(dim=-1)

    #     predictions[batch_idx, :] = prob.argmax(dim=-1)
    #     prob = F.softmax(prob, dim=-1)

    #     probabilities.append(prob)

    return predictions, probabilities
