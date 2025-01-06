import json
import logging
import os
from typing import Optional

from openai import OpenAI

OBJECT_PROMPT = """You are given a description of an object that someone is supposed to find in a scene. Similar to the Visual Genome dataset, we would like to identify the objects, attributes, and relationships in the following text:
"{}"

Please first return a list of the objects in the scene in a JSON format:
{{
  "success": boolean,
  "objects": [
    {{
      "id": id of object,
      "name": name of object as string,
    }}
  ],
  "target": list of ids of object,
  "target_reference_type": "generic", "categorical", or "fine-grained",
  "first_noun": boolean
}}

The object IDs should start with 0 and increment.

The name of the object should be lowercase (except for proper nouns) and sufficient to define the object class, if specified. Attributes should not be included in the class name. For instance, "a big, red apple" has the object name of "apple", and a "rectangular washing machine" has the object name of "washing machine". Parts of objects should be included as separate objects. For instance, "a chair with four legs" has objects "chair" and "legs".

If the target object is implied in the text but not explicitly named, such as in, "Where can I keep food cold?" the target object name should just be specified as "object".

The target ID should be the list of IDs of the objects that the user is supposed to find. For example, if the prompt is, "This is the toolbox on the shelf", the target ID should be the ID of the toolbox.

target_reference_type should be specified as the most specific term used to name the object:
* generic - object is referred to by a general term such as "object", "thing", or "item", or if the object is implied but not explicitly named.
* categorical - object is referred to by a category which is more specific than a generic reference but not specific to a particular object class. This includes references such as "appliance", "seat", "container", "display", "machine", or "device".
* fine-grained - object is referred to by a specific object class, from which it should be easy to infer what the object is and how it should be used.

If the prompt is not a description of an object in a scene, set "success" to False and ignore the rest of the output. Otherwise, set it to True.

If the target object is the first noun phrase mentioned in the description, set "first_noun" to True. Otherwise, set it to False.
"""

ATTRIBUTES_PROMPT = """You are given a description of an object that someone is supposed to find in a scene. Your goal is to calculate statistics about the attributes used to describe objects in the description:
"{}"

The parsed list of objects is as follows:
{}

The parsed list of relationships is as follows:
{}

Return the output in a JSON format according to the following format:
{{
  "num_attribute_type": {{
    "color": {{
        "exists": True if attribute is found in prompt or False otherwise,
        "explanation": list of attributes identified, or empty if none
    }},
    "size": {{
        "exists": True if attribute is found in prompt or False otherwise,
        "explanation": list of attributes identified, or empty if none
    }},
    "shape": {{
        "exists": True if attribute is found in prompt or False otherwise,
        "explanation": list of attributes identified, or empty if none
    }},
    "number": {{
        "exists": True if attribute is found in prompt or False otherwise,
        "explanation": list of attributes identified, or empty if none
    }},
    "material": {{
        "exists": True if attribute is found in prompt or False otherwise,
        "explanation": list of attributes identified, or empty if none
    }},
    "texture": {{
        "exists": True if attribute is found in prompt or False otherwise,
        "explanation": list of attributes identified, or empty if none
    }},
    "function": {{
        "exists": True if attribute is found in prompt or False otherwise,
        "explanation": list of attributes identified, or empty if none
    }},
    "style": {{
        "exists": True if attribute is found in prompt or False otherwise,
        "explanation": list of attributes identified, or empty if none
    }},
    "text_label": {{
        "exists": True if attribute is found in prompt or False otherwise,
        "explanation": list of attributes identified, or empty if none
    }},
    "state": {{
        "exists": True if attribute is found in prompt or False otherwise,
        "explanation": list of attributes identified, or empty if none
    }}
  }},
  "attributes": [
    {{
      "object_id": id of object,
      "attributes": list of attributes
    }}
  ]
}}

Attributes are any descriptors that help distinguish an object from others. The name of an object does NOT count as an attribute.

Each attribute type is defined as follows:

color: Any attribute which describes the color properties of an object. Examples include red, blue, black, light, dark, monocolor, or colorful.

size: Any attribute which describes the size of an object. Examples include big, small, large, larger, tall, long, short, or medium. You should exclude cases where the height of an object is described to capture vertical position rather than size.

shape: Any attribute which describes the shape or form of an object. Examples include round, square, rectangular, or circular.

number: Any attribute which describes the quantity of an object. Examples include "two chairs". This does not include cases where the number is used to describe the relative order of the object, or cases where "one" is used as a pronoun to refer to the object.

material: Any attribute which describes what an object is made of. Examples include wood, metal, plastic, or glass. If the attribute describes the texture but not what the object is actually made of, e.g. metallic, then it should count as a texture attribute rather than a material.

texture: Any attribute which describes the texture of an object. Examples include smooth, rough, soft, metallic, or comfy.

function: Any attribute which describes what an object can be used for or the function it performs in a space. Examples include a chair for sitting or a lamp that makes the space warm and welcoming. The name of the object does not count as a function.

style: Any attribute which describes the style of an object or the effect of its presence in the space. Examples include modern, vintage, antique, futuristic, luxurious, or industrial, or describing its prominent or subtle presence in a room.

text_label: Any attribute which describes text that can be found on an object. Examples include "fragile" on a box or "exit" on a door.

state: Any attribute which describes the state of an object, which can be changed. Examples include "unopened" to describe a jar, "broken", or "drying" to describe clothes hanging on a rack.

You should also include a list of each of the attributes for each object in the scene.
"""


RELATIONSHIPS_PROMPT = """You are given a description of an object that someone is supposed to find in a scene. Your goal is to analyze the prompt for information about the relationships used to describe objects in the description:
"{}"

The parsed list of objects is as follows:
{}

Return the output in a JSON format according to the following format:
{{
  "relationships": [
    {{
        "name": name of relationship as string as it appears in the prompt,
        "subject_id": list of ids of objects which are the subject of the relationship,
        "recipient_id": list of ids of objects which is the recipient of the relationship
    }}
  ],
  "num_relationship_type": {{
    "near": {{
      "exists": True if relationship is found in prompt or False otherwise,
      "explanation": list of relationships identified, or empty if none
    }},
    "far": {{
      "exists": True if relationship is found in prompt or False otherwise,
      "explanation": list of relationships identified, or empty if none
    }},
    "viewpoint_dependent": {{
      "exists": True if relationship is found in prompt or False otherwise,
      "explanation": list of relationships identified, or empty if none
    }},
    "vertical": {{
      "exists": True if relationship is found in prompt or False otherwise,
      "explanation": list of relationships identified, or empty if none
    }},
    "contain": {{
      "exists": True if relationship is found in prompt or False otherwise,
      "explanation": list of relationships identified, or empty if none
    }},
    "arrangement": {{
      "exists": True if relationship is found in prompt or False otherwise,
      "explanation": list of relationships identified, or empty if none
    }},
    "ordinal": {{
      "exists": True if relationship is found in prompt or False otherwise,
      "explanation": list of relationships identified, or empty if none
    }},
    "comparison": {{
      "exists": True if relationship is found in prompt or False otherwise,
      "explanation": list of relationships identified, or empty if none
    }}
  }},
  "anchors": {{
    "single": True if at least one of the anchor objects is a single object otherwise False,
    "multiple": True if at least one of the anchor objects represents multiple objects otherwise False,
    "non_object": True if at least one of the anchor objects represents a region or room otherwise False
    "viewpoint": True if one of the relationships requires a specific viewpoint otherwise False
  }}
}}

A relationship compares an object(s) or region to another object(s) or region. Relationships should capture objects in the scene and not hypothetical objects. The name of the relationship should be the word or phrase used in the description to describe the relationship. If a noun in the description is a part of an object rather than a distinct object, then it should be counted as a part rather than an object. If a recipient is relative to the speaker of the description, use the ID of the "<speaker>" object.

Each relationship type is defined as follows:

near: Any relationship which describes one object in proximity of another. Examples include near, next to, nearby, adjacent, close to, proximate, amidst, among, covered, or contact relationships (against, leaning on, on, hanging on, supported by, attached to).

far: Any relationship which describes one object far away from another. Examples include far from, opposite, across from, and distant from.

viewpoint_dependent: Any relationship which can only be identified based on a canonical reference frame of the object or the viewpoint of the speaker. This includes left, right, in front of, facing, behind, or any cardinal direction.

vertical: Any relationship which describes one object above or below another. Examples include above, below, on top of, under, underneath, or vertical support relationships (e.g., an object on another).

contain: Any relationship which describes one object contained within another or some part that belongs to another. Examples include in, inside, within, with, has, or have.

arrangement: Any relationship which describes one object as part of an ordered arrangement. Examples include "between", "surrounded by", "row of", "column of", "stack of", or "pile of" other objects. You should exclude "amidst", "among", "nearby" or other non-structured relationships.

ordinal: Any relationship which describes the numerical position of an object in a spatial order or array. Examples include first, 2nd, middle, last. You should exclude cases of an object being the closest, leftmost, rightmost, or equivalent.

comparison: Any relationship which compares properties of different objects and requires identifying which one is more or less, or the most or least, of something. Examples include taller, tallest, shorter, greenest, closest, furthest, or same as.

In the explanation, each relationship should be given as a list of [subject, relationship, recipient].

Lastly, indicate the following:
1. If any of the subjects or recipients of a relationship, excluding the target, is a single object, set "single" to True
2. If any of the subjects or recipients of a relationship, excluding the target, represents multiple objects, set "multiple" to True. Examples include "the table is surrounded by six chairs", "the car is in between the shovel and the desk", "the book is the third one on the shelf", or "the chair is the one closest to the door".
3. If any of the subjects or recipients of a relationship is a region or room, set "non_object" to True. Examples include "the shelf in the center of the room", "the microwave in the kitchen", or "the books in the area around the couch".
4. If finding the target is dependent on a specific viewpoint in the scene, set "viewpoint" to True. Examples include "the leftmost wall" or "the window on your right."
"""


class InvalidSceneGraphError(Exception):
    pass


class GroundingParser:
    """
    Parse an unstructured scene description into a structured scene specification using an LLM-based method.

    Use of this module requires an OPENAI_API_KEY environment variable.

    See https://platform.openai.com/docs/models for additional models supported by OpenAI.
    """

    cost_per_1m = {
        "gpt-4o": [5, 15],
        "gpt-4o-2024-08-06": [2.5, 10],
        "gpt-4o-2024-05-13": [5, 15],
        "gpt-4o-mini": [0.15, 0.6],
        "gpt-4o-mini-2024-07-18": [0.15, 0.6],
    }

    def __init__(self, model_name: str = "gpt-4o-mini", **kwargs):
        self.model = model_name

        try:
            api_key = os.environ["OPENAI_API_KEY"]
        except KeyError:
            raise EnvironmentError(
                "Expected an OpenAI API key in order to use the LLMSceneParser. Please set OPENAI_API_KEY and "
                "try again."
            )
        self.client = OpenAI(api_key=api_key)
        self.inp_tokens_used = 0
        self.out_tokens_used = 0
        self.num_requests = 0

    def get_cost(self) -> float:
        """Return projected api cost in dollars"""
        if self.model not in self.cost_per_1m:
            raise ValueError(f"No cached API usage costs for {self.model}.")
        return (
            self.inp_tokens_used * self.cost_per_1m[self.model][0]
            + self.out_tokens_used * self.cost_per_1m[self.model][1]
        ) / 1e6

    def make_json_request(self, prompt: str, description: str, *args) -> dict:
        inp = prompt.format(description, *args)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an assistant helping a user generate a semantic scene graph from a text description of a visual grounding prompt.",
                },
                {"role": "user", "content": inp},
            ],
            response_format={"type": "json_object"},
        )

        raw_output = response.choices[0].message.content
        self.inp_tokens_used += response.usage.prompt_tokens
        self.out_tokens_used += response.usage.completion_tokens
        self.num_requests += 1
        try:
            output_json = json.loads(raw_output)
        except json.JSONDecodeError:
            raise ValueError(f"Failed to parse scene graph response from OpenAI as JSON:\n{raw_output}")

        return output_json

    def _parse_objects(self, objects: dict) -> str:
        pretty_objects = []
        for obj in objects["objects"]:
            pretty_objects.append(f'object_id: {obj["id"]}, name: "{obj["name"]}"')
        return "\n".join(pretty_objects)

    def _parse_relationships(self, relationships: dict) -> str:
        pretty_relationships = []
        for rel in relationships["relationships"]:
            if isinstance(rel, str):
                breakpoint()
            pretty_relationships.append(
                f'name: "{rel["name"]}", subject_id: {rel["subject_id"]}, recipient_id: {rel["recipient_id"]}'
            )
        return "\n".join(pretty_relationships)

    def parse(self, description: str, id: Optional[str] = None) -> dict:
        logging.debug(f"Parsing scene graph from text input: {description}")
        if "<p>" in description:
            pass

        # inp = self.prompt.format(description)

        # parse objects
        objects = self.make_json_request(OBJECT_PROMPT, description)
        if not objects["success"]:
            raise InvalidSceneGraphError(f"Unable to parse as a visual grounding prompt: {description}")
        max_id = max(obj["id"] for obj in objects["objects"])
        objects["objects"].append({"id": max_id + 1, "name": "<speaker>"})

        # parse relationships
        pretty_objects = self._parse_objects(objects)
        relationships = self.make_json_request(RELATIONSHIPS_PROMPT, description, pretty_objects)

        # parse attributes
        pretty_relationships = self._parse_relationships(relationships)
        attributes = self.make_json_request(ATTRIBUTES_PROMPT, description, pretty_objects, pretty_relationships)
        output_json = {**objects, **relationships, **attributes}
        return output_json


if __name__ == "__main__":
    import pprint

    parser = GroundingParser(model_name="gpt-4o")
    descriptions = [
        "This trash can is on the left. It is green.",
        "Next to the bookshelf is a rectangular coffee table.",
        "This is a tall plant which brightens the room.",
        "The plastic box is on top of two colorful books.",
        "The chair has four legs.",
        "The box is labeled 'Eat Me'.",
        "It is a modern, two-seater sofa with orange pillows on top and four pink legs.",
        "This is the shelf in the room.",
        "This is the shelf in the room. There is a bright blue, rectangular box next to it and four thick books on top.",
        "This is the stainless steel sink in between the purple cow and the pink llama.",
    ]
    for description in descriptions:
        print(f"Description: {description}")
        objects = parser.make_json_request(OBJECT_PROMPT, description)
        max_id = max(obj["id"] for obj in objects["objects"])
        objects["objects"].append({"id": max_id + 1, "name": "<speaker>"})
        pprint.pprint(objects)
        pretty_objects = parser._parse_objects(objects)
        relationships = parser.make_json_request(RELATIONSHIPS_PROMPT, description, pretty_objects)
        pprint.pprint(relationships)
        pretty_relationships = parser._parse_relationships(relationships)
        attributes = parser.make_json_request(ATTRIBUTES_PROMPT, description, pretty_objects, pretty_relationships)
        pprint.pprint(attributes)
        print("-----")
