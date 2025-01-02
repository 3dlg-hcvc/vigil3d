"""
filter.py
---------
Functions to prefilter a dataset.
"""


class BaseFilter:
    def __call__(self, scene_ids: list[str], metadata) -> tuple[list[str], dict]:
        raise NotImplementedError


class FilterByNumTargets(BaseFilter):
    def __init__(self, target_types: list[str]):
        self.target_types = target_types

    def __call__(self, scene_ids: list[str], metadata) -> tuple[list[str], dict]:
        filtered_scene_ids = set()
        filtered_metadata = {"grounding": []}
        for prompt in metadata["grounding"]:
            if prompt["scene_id"] in scene_ids and prompt["metadata"]["target_type"] in self.target_types:
                filtered_scene_ids.add(prompt["scene_id"])
                filtered_metadata["grounding"].append(prompt)
        return filtered_scene_ids, filtered_metadata
