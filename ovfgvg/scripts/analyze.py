"""
analyze.py
-----------
Main analysis script for datasets.
"""

import json
import logging
import os
import re
import traceback
from collections import Counter
from dataclasses import asdict, dataclass, field, fields
from itertools import chain
from omegaconf import DictConfig
from typing import Self

import hydra
import spacy
from nltk.corpus import wordnet as wn
from tqdm import tqdm

# from ovfgvg.data.analysis.scenegraph import GroundingParser, InvalidSceneGraphError, SceneGraph
from ovfgvg.data.analysis.directparse import GroundingParser, InvalidSceneGraphError
from ovfgvg.data.types import SceneCollection
from ovfgvg.utils import setup_environment

NUM_UPOS_TAGS = 18  # https://universaldependencies.org/u/pos/ (17 UPOS tokens + custom START token)
UPOS_NGRAMS = [2, 3]
OBJECT_SYNSETS = [
    "object.n.01",
    "thing.n.04",
    "item.n.03",
    "article.n.02",
    "artifact.n.01",
    "doodad.n.01",
    "target.n.01",
]
OBJECT_SYNONYMS = set(
    map(lambda word: word.replace("_", " "), chain.from_iterable([wn.synset(s).lemma_names() for s in OBJECT_SYNSETS]))
)
LATEX_NONE_MARK = "\\myxmark"
LATEX_SOME_MARK = "\\mycheckmark"
LATEX_MANY_MARK = "\\doublecheckmark"


@dataclass
class Metrics:
    # metric counts by prompts
    num_prompts: int = 0
    num_valid_prompts: int = 0
    num_unique_tokens: int = 0
    num_sentences: int = 0
    num_words: int = 0

    num_prompts_zero_target: int = 0
    num_prompts_single_target: int = 0
    num_prompts_multiple_target: int = 0

    num_target_ref_generic: int = 0
    num_target_ref_categorical: int = 0
    num_target_ref_fine_grained: int = 0
    num_target_not_first_np: int = 0
    num_coreferences: int = 0
    num_negation: int = 0
    num_attribute_type: dict = field(
        default_factory=lambda: Counter(
            color=0, size=0, shape=0, number=0, material=0, texture=0, function=0, style=0, text_label=0, state=0
        )
    )
    num_attribute_type_unique: dict = field(
        default_factory=lambda: Counter(
            color=0, size=0, shape=0, number=0, material=0, texture=0, function=0, style=0, text_label=0, state=0
        )
    )
    num_relationship_type: dict = field(
        default_factory=lambda: Counter(
            near=0, far=0, viewpoint_dependent=0, vertical=0, contain=0, arrangement=0, comparison=0, ordinal=0
        )
    )
    num_relationship_type_unique: dict = field(
        default_factory=lambda: Counter(
            near=0, far=0, viewpoint_dependent=0, vertical=0, contain=0, arrangement=0, comparison=0, ordinal=0
        )
    )
    num_anchor_type_single: int = 0
    num_anchor_type_multiple: int = 0
    num_anchor_type_non_object: int = 0
    num_anchor_type_viewpoint: int = 0

    # metric counts averaged across prompts
    num_attributes_total: int = 0
    num_attributes_target: int = 0
    num_attributes_anchors: int = 0
    num_relationships_total: int = 0
    num_relationships_target: int = 0
    num_relationships_anchors: int = 0
    num_objects: int = 0
    num_bigrams: int = 0
    num_unique_bigrams_sem: int = 0
    num_unique_ngrams_upos: dict = field(default_factory=lambda: Counter({"2": 0, "3": 0}))
    num_unique_bigrams_dep: int = 0

    def __getattribute__(self, name):
        if name.startswith("m_ave_"):
            match name[6:]:
                case "objects_anchor":
                    return (self.num_objects - self.num_prompts) / self.num_prompts if self.num_prompts > 0 else 0
                case _:
                    return getattr(self, f"num_{name[6:]}") / self.num_prompts if self.num_prompts > 0 else 0
        elif name.startswith("m_percent_"):
            match name[10:]:
                case "unique_bigrams_sem":
                    return self.num_unique_bigrams_sem / self.num_bigrams if self.num_bigrams > 0 else 0
                case "unique_bigrams_dep":
                    return self.num_unique_bigrams_dep / self.num_bigrams if self.num_bigrams > 0 else 0
                case "unique_bigrams_upos":
                    return self.num_unique_ngrams_upos["2"] / (NUM_UPOS_TAGS**2)
                case "unique_trigrams_upos":
                    return self.num_unique_ngrams_upos["3"] / (NUM_UPOS_TAGS**3)

            # num_attribute_type
            match = re.match(r"m_percent_attribute_type(?:\.|_)(\w+)", name)
            if match:
                attr_type = match.group(1)
                return self.num_attribute_type[attr_type] / self.num_prompts if self.num_prompts > 0 else 0

            # num_relationship_type
            match = re.match(r"m_percent_relationship_type(?:\.|_)(\w+)", name)
            if match:
                rel_type = match.group(1)
                return self.num_relationship_type[rel_type] / self.num_prompts if self.num_prompts > 0 else 0

            # final case
            return getattr(self, f"num_{name[10:]}") / self.num_prompts if self.num_prompts > 0 else 0
        else:
            return super().__getattribute__(name)

    def __add__(self, other: Self | dict) -> Self:
        sum_metrics = Metrics()
        for f in fields(sum_metrics):
            if isinstance(other, Metrics):
                setattr(sum_metrics, f.name, getattr(self, f.name) + getattr(other, f.name))
            else:
                other_data = other.get(f.name, getattr(sum_metrics, f.name))
                if isinstance(other_data, dict) and not isinstance(other_data, Counter):
                    other_data = Counter(other_data)
                setattr(sum_metrics, f.name, getattr(self, f.name) + other_data)

        return sum_metrics

    @classmethod
    def from_json(self, data: dict) -> Self:
        out = Metrics()  # for counters and other variables that don't like being directly initialized
        out += {k: v for k, v in data.items() if k in asdict(out)}
        return out

    def to_json(self):
        data = asdict(self)
        existing_keys = set(data.keys())
        for key in dir(self):
            if key.startswith("m_") and key not in existing_keys:
                data[key] = getattr(self, key)

        # patch counters
        for key in data:
            if isinstance(getattr(self, key), Counter):
                data[key] = dict(getattr(self, key))
        return data

    def to_latex_row(self, dataset_name: str, columns: list[str]):
        output = [dataset_name]
        for col in columns:
            value = getattr(self, col)
            if col.startswith("m_percent_") and "grams" not in col:
                if value < 0.05:
                    output.append(LATEX_NONE_MARK)
                elif value < 0.2:
                    output.append(LATEX_SOME_MARK)
                else:
                    output.append(LATEX_MANY_MARK)
            else:
                output.append(f"{value:.2f}")
        return " & ".join(output) + " \\\\"


def has_coreferences(doc):
    found_first_nsubj = False  # exclude first nsubj
    for token in doc:
        if token.dep_ == "nsubj" and not found_first_nsubj:
            found_first_nsubj = True
            continue

        if (
            token.pos_ == "PRON"
            and token.text.lower() not in {"you", "your", "yours", "yourself", "yourselves"}
            and token.tag_ not in {"WDT", "WP", "EX", "NN"}
        ):
            return True
    return False


@hydra.main(version_base=None, config_path="../../config", config_name="analysis_config")
def main(cfg: DictConfig):
    # 1. setup environment
    setup_environment(cfg.env)
    nlp = spacy.load("en_core_web_md")
    # coref = nlp.add_pipe("experimental_coref")
    # coref.initialize
    sg_parser = GroundingParser(cfg.parser_model)
    if os.path.exists(os.path.join(cfg.data.source, cfg.split, cfg.output_name)):
        with open(os.path.join(cfg.data.source, cfg.split, cfg.output_name), "r") as f:
            results_cache = json.load(f)
        if "seed" in results_cache:
            if cfg.shuffle and cfg.skip_existing and cfg.env.seed != results_cache["seed"]:
                raise ValueError(
                    f"Seed mismatch: {cfg.env.seed} (config) != {results_cache['seed']} (cache). "
                    "Please ensure that the seed is consistent across runs if you are continuing a job and shuffling "
                    "prompts."
                )
    else:
        results_cache = {"seed": cfg.env.seed, "descriptions": {}, "metrics": {}}
    seen = set()
    for prompt_data in results_cache["descriptions"].values():
        seen.add((str(prompt_data["prompt_id"]), prompt_data["text"]))

    # 2. initialize metrics
    # total_metrics = Metrics.from_json(results_cache.get("metrics", {}))
    total_metrics = Metrics.from_json({})

    # 3. load prompts
    scenes = SceneCollection(cfg.data.dataset_name, cfg.data.source, splits=[cfg.split])

    # 4. iterate through prompts
    bigrams_sem = set()
    ngrams_upos = {str(n): set() for n in UPOS_NGRAMS}
    bigrams_dep = set()
    vocab = set()
    num_failed = 0
    if cfg.use_existing_only:

        def iterable():
            for prompt_data in results_cache["descriptions"].values():
                yield scenes.get_annotation_by_id(cfg.split, prompt_data["prompt_id"])

    else:

        def iterable():
            return scenes.iter_annotations(cfg.split, count=cfg.num_prompts, shuffle=cfg.shuffle)

    # iterate through prompts
    for idx, prompt in tqdm(enumerate(iterable())):
        try:
            prompt_id = str(prompt.id)
            metrics = Metrics()
            metrics.num_prompts = 1

            # 4a. parse description
            doc = nlp(prompt.text)  # linguistic features
            if not cfg.skip_existing or (prompt_id, prompt.text) not in seen:
                print("--------------------")
                print(f"Prompt: {prompt.text}\n")
                graph = sg_parser.parse(prompt.text, id=prompt_id)  # scene graph
            else:
                graph = results_cache["descriptions"][prompt_id]["scene_graph"]

            metrics.num_valid_prompts = 1

            # 4b. compute metrics
            # num targets
            if prompt.entities is not None:
                metrics.num_prompts_zero_target = 1 if len(prompt.entities[0].ids) == 0 else 0
                metrics.num_prompts_single_target = 1 if len(prompt.entities[0].ids) == 1 else 0
                metrics.num_prompts_multiple_target = 1 if len(prompt.entities[0].ids) > 1 else 0

            # sentence diversity
            metrics.num_sentences = len(list(doc.sents))
            metrics.num_words = len(prompt.text.split())
            prev_vocab_size = len(vocab)
            vocab |= set(prompt.text.split())
            bigrams_sem_count = len(bigrams_sem)
            ngrams_upos_count = {str(n): len(ngrams_upos[str(n)]) for n in UPOS_NGRAMS}
            bigrams_dep_count = len(bigrams_dep)
            sentences = list(doc.sents)
            for sent_idx, sentence in enumerate(sentences):
                sem_tokens = [token.text for token in sentence]
                start_token = "START" if sent_idx == 0 else sentences[sent_idx - 1][-1].pos_
                pos_tokens = [start_token, *[token.pos_ for token in sentence]]
                for idx in range(len(sem_tokens)):
                    if idx < len(sem_tokens) - 1:
                        bigrams_sem.add((sem_tokens[idx], sem_tokens[idx + 1]))
                        bigrams_dep.add((doc[idx].dep_, doc[idx + 1].dep_))  # TODO: give this its own loop

                for idx in range(len(pos_tokens)):
                    for n in UPOS_NGRAMS:
                        if idx < len(pos_tokens) - n + 1:
                            ngram = tuple(pos_tokens[idx + i] for i in range(n))
                            ngrams_upos[str(n)].add(ngram)
            metrics.num_unique_tokens += len(vocab) - prev_vocab_size
            metrics.num_bigrams = len(doc) - 1
            metrics.num_unique_bigrams_sem = len(bigrams_sem) - bigrams_sem_count
            for n in UPOS_NGRAMS:
                metrics.num_unique_ngrams_upos[str(n)] = len(ngrams_upos[str(n)]) - ngrams_upos_count[str(n)]
            metrics.num_unique_bigrams_dep = len(bigrams_dep) - bigrams_dep_count

            # object metrics
            match graph["target_reference_type"]:
                case "generic":
                    metrics.num_target_ref_generic = 1
                case "categorical":
                    metrics.num_target_ref_categorical = 1
                case "fine-grained":
                    metrics.num_target_ref_fine_grained = 1

            metrics.num_target_not_first_np = 0 if graph["first_noun"] else 1
            metrics.num_objects = len(graph["objects"])

            # coreferences
            if has_coreferences(doc):
                metrics.num_coreferences = 1

            # attribute metrics
            for attr in graph.get("attributes", []):
                if attr["object_id"] in graph["target"]:
                    metrics.num_attributes_target += len(attr["attributes"])
                else:
                    metrics.num_attributes_anchors += len(attr["attributes"])
            metrics.num_attributes_total = metrics.num_attributes_target + metrics.num_attributes_anchors

            all_attr_types = []
            for attr_type in graph.get("num_attribute_type", []):
                if graph["num_attribute_type"][attr_type]["exists"]:
                    metrics.num_attribute_type[attr_type] = 1
                    all_attr_types.append(attr_type)
            if len(all_attr_types) == 1:
                metrics.num_attribute_type_unique[all_attr_types[0]] = 1

            # relationship metrics
            for rel in graph.get("relationships", []):
                if set(rel["subject_id"]) & set(graph["target"]) or set(rel["recipient_id"]) & set(graph["target"]):
                    metrics.num_relationships_target += 1
                else:
                    metrics.num_relationships_anchors += 1
            metrics.num_relationships_total = metrics.num_relationships_target + metrics.num_relationships_anchors
            all_rel_types = []
            for rel_type in graph.get("num_relationship_type", []):
                if graph["num_relationship_type"][rel_type]["exists"]:
                    metrics.num_relationship_type[rel_type] = 1
                    all_rel_types.append(rel_type)
            if len(all_rel_types) == 1:
                metrics.num_relationship_type_unique[all_rel_types[0]] = 1

            metrics.num_anchor_type_single = 1 if graph["anchors"]["single"] else 0
            metrics.num_anchor_type_multiple = 1 if graph["anchors"]["multiple"] else 0
            metrics.num_anchor_type_non_object = 1 if graph["anchors"]["non_object"] else 0
            metrics.num_anchor_type_viewpoint = 1 if graph["anchors"]["viewpoint"] else 0

            # negation metrics
            for token in doc:
                if token.dep_ == "neg" or token in {
                    "no",
                    "neither",
                    "nor",
                    "nothing",
                    "nowhere",
                    "nobody",
                    "none",
                    # "contrast",
                    # "contrasted",
                }:
                    metrics.num_negation = 1
                    break

            assert (
                prompt_id not in results_cache["descriptions"]
                or results_cache["descriptions"][prompt_id]["text"] == prompt.text
            )
            results_cache["descriptions"][prompt_id] = {
                "prompt_id": prompt_id,
                "text": prompt.text,
                "scene_graph": graph,
                "metrics": metrics.to_json(),
            }
            total_metrics += metrics
            seen.add((prompt_id, prompt.text))
        except Exception as e:
            logging.error(f"Error processing prompt: {prompt.text}")
            logging.error(f"Error message: {e}")
            traceback.print_exc()
            num_failed += 1

    results_cache["metrics"] = total_metrics.to_json()
    results_cache["usage"] = {
        "inp_tokens": sg_parser.inp_tokens_used,
        "out_tokens": sg_parser.out_tokens_used,
        "num_requests": sg_parser.num_requests,
        "cost": sg_parser.get_cost(),
        "num_prompts_failed": num_failed,
    }

    # generate table entry
    results_cache["latex"] = total_metrics.to_latex_row(cfg.data.dataset_name, cfg.latex_columns)
    print("LaTeX code:")
    print(results_cache["latex"])

    with open(os.path.join(cfg.data.source, cfg.split, cfg.output_name), "w") as f:
        json.dump(results_cache, f, indent=4)

    if num_failed > 0:
        logging.error(f"Failed to process {num_failed} prompts.")
    else:
        logging.info("All prompts processed successfully.")
    logging.info(f"Saved results to {os.path.join(cfg.data.source, cfg.split, cfg.output_name)}")


if __name__ == "__main__":
    main()
