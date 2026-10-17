"""Prospective exact-reward/on-policy bridge contract; no historical rescoring."""

import json
import re

from platform_hybrid.experiments.samestack_protocol import exact_number, last_boxed_content

PROTOCOL_ID = "skyrl-tinker-exact-onpolicy-v2"


def _equal_number(left, right):
    a, b = exact_number(left), exact_number(right)
    return a is not None and b is not None and a == b


def gsm8k_reward(response, item):
    if not isinstance(response, str):
        return 0.0
    candidate = response.strip()
    if "####" in candidate:
        candidate = candidate.rsplit("####", 1)[1].strip()
    else:
        match = re.fullmatch(r"(?:the\s+)?answer\s+is\s+(.+)", candidate, re.I)
        if match:
            candidate = match[1].strip()
    return float(_equal_number(candidate, item.get("answer")))


def math_reward(response, item):
    if not isinstance(response, str) or not isinstance(item.get("answer"), str):
        return 0.0
    predicted = last_boxed_content(response)
    gold = last_boxed_content(item["answer"])
    # Symbolic equivalence is deliberately unsupported, never partial-string matched.
    return float(_equal_number(predicted, gold))


def _unique_object(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError("duplicate JSON key")
        obj[key] = value
    return obj


def tool_reward(response, item):
    try:
        parsed = json.loads(response, object_pairs_hook=_unique_object)
        if not isinstance(parsed, dict) or set(parsed) - {
            "tool",
            "name",
            "arguments",
            "parameters",
        }:
            return 0.0
        if "tool" in parsed and "name" in parsed and parsed["tool"] != parsed["name"]:
            return 0.0
        if (
            "arguments" in parsed
            and "parameters" in parsed
            and parsed["arguments"] != parsed["parameters"]
        ):
            return 0.0
        arguments = parsed.get("arguments", parsed.get("parameters"))
        if not isinstance(arguments, dict):
            return 0.0
        same_args = json.dumps(arguments, sort_keys=True, allow_nan=False) == json.dumps(
            item["args"], sort_keys=True, allow_nan=False
        )
        return float(parsed.get("tool", parsed.get("name")) == item["tool"] and same_args)
    except (ValueError, TypeError, KeyError):
        return 0.0


def refresh_sampler(client, completed_step):
    checkpoint = client.save_weights_for_sampler(name=f"sampler_{completed_step}").result()
    return client.create_sampling_client(model_path=checkpoint.path), checkpoint.path


def load_unique_yaml(stream):
    import yaml

    class UniqueKeyLoader(yaml.SafeLoader):
        pass

    def mapping(loader, node, deep=False):
        loader.flatten_mapping(node)
        result = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if key in result:
                raise ValueError(f"duplicate YAML key: {key}")
            result[key] = loader.construct_object(value_node, deep=deep)
        return result

    UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
    loader = UniqueKeyLoader(stream)
    try:
        return loader.get_single_data()
    finally:
        loader.dispose()
