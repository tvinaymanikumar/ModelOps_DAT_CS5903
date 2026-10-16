#!/usr/bin/env python3
"""Evaluate flattened prediction JSON files against flattened ground-truth JSON files.

The script prompts for folders when command-line options are omitted.

Modes:
  schema     Exact normalized key matching + known aliases only.
  schemafree Exact/alias matching, then one-to-one semantic key matching.
  auto       Same staged matching as schemafree; useful when you are unsure.

Base requirements: Python 3.10+ (standard library only).
Optional, recommended for schema-free matching and semantic similarity:
  pip install sentence-transformers

One <filename>_evaluation.json is written for each ground-truth JSON file.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# Add common aliases here, or use --aliases aliases.json with the format:
# {"canonical_key": ["alias_1", "alias_2"]}
DEFAULT_ALIASES = {
    "patient_name": ["name_of_patient", "patients_name", "pt_name", "patient_full_name", "name_patient"],
    "age": ["patient_age", "age_years", "age_of_patient", "years_old"],
    "sex": ["gender", "patient_sex", "patient_gender"],
    "date_of_admission": ["admission_date", "admit_date", "date_admitted"],
    "date_of_discharge": ["discharge_date", "date_discharged"],
    "hospital_name": ["hospital", "name_of_hospital", "facility_name", "medical_center"],
    "account_number": ["account_no", "acc_no", "a_c_no", "ac_no", "account_num"],
    "cheque_number": ["cheque_no", "check_number", "check_no", "chq_no"],
    "amount_in_words": ["amount_words", "legal_amount"],
    "amount_in_figures": ["amount_figures", "courtesy_amount", "amount_numeric", "numeric_amount"],
    "payee_name": ["payee", "pay_to", "beneficiary_name"],
    "invoice_number": ["invoice_no", "inv_no", "bill_number", "bill_no"],
    "invoice_date": ["date_of_invoice", "bill_date"],
    "company": ["company_name", "vendor", "vendor_name", "seller", "store_name"],
    "address": ["company_address", "vendor_address", "store_address"],
    "total": ["total_amount", "grand_total", "amount_due", "net_total"],
    "document_type": ["doc_type", "document_class", "form_type", "document_category"],
}

CONFLICT_PAIRS = [
    ("admission", "discharge"), ("start", "end"), ("from", "to"),
    ("issue", "due"), ("billing", "shipping"), ("sender", "receiver"),
    ("payer", "payee"), ("debit", "credit"), ("first", "last"),
    ("birth", "death"), ("subtotal", "total"), ("words", "figures"),
    ("min", "max"),
]

WRAPPER_KEYS = {
    "fields", "data", "extracted_fields", "result", "results", "output",
    "key_value_pairs", "form", "extraction",
}
PAIR_FIELDS = [("key", "value"), ("question", "answer"), ("field", "value"), ("label", "text")]
EMPTY_VALUES = {"", "n/a", "none", "null", "nil", "-", "--"}
DOCUMENT_TYPE_KEYS = {"document_type", "doc_type", "document_class", "form_type", "document_category"}


def norm_key(key: Any) -> str:
    """Normalize key case, punctuation, and spaces while preserving word boundaries."""
    text = unicodedata.normalize("NFKC", str(key))
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    return re.sub(r"[^0-9A-Za-z]+", "_", text).strip("_").lower()


def build_alias_lookup(aliases: Dict[str, List[str]]) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for canonical, variants in aliases.items():
        c = norm_key(canonical)
        lookup[c] = c
        for alias in variants:
            lookup[norm_key(alias)] = c
    return lookup


def canonical_key_path(key: str, alias_lookup: Dict[str, str]) -> str:
    """Canonicalize key aliases in every segment while preserving nested/list path separators."""
    parts = re.split(r"([.\[\]])", str(key))
    result = []
    for part in parts:
        if part in {".", "[", "]"}:
            result.append(part)
        elif part == "":
            result.append(part)
        else:
            normalized = norm_key(part)
            result.append(alias_lookup.get(normalized, normalized))
    return "".join(result)


def leaf_key(key: str) -> str:
    """Return the field label without its parent group path.

    Example: 'Personal Information.Full Name' -> 'Full Name'.
    This helps compare nested ground truth to flat model output.
    """
    text = str(key)
    # Strip a trailing list index, but preserve the label before it.
    text = re.sub(r"\[\d+\]$", "", text)
    return text.rsplit(".", 1)[-1]


def canonical_leaf_key(key: str, alias_lookup: Dict[str, str]) -> str:
    leaf = norm_key(leaf_key(key))
    return alias_lookup.get(leaf, leaf)


def expand_single_choice_prediction(
    gt: Dict[str, str], pred: Dict[str, str]
) -> Tuple[Dict[str, str], List[Dict[str, str]]]:
    """Expand a scalar choice into one-hot fields if GT clearly encodes radio/checkbox options.

    Example GT:
      'Membership.Regular (3 month)': 'Yes',
      'Membership.Gold (6 month)': 'No'
    Prediction:
      'membership': 'Regular (3 month)'
    The evaluation view becomes Yes/No values for the option fields. The raw
    prediction is retained separately in the output report.
    Expansion only occurs for groups containing at least two binary-valued
    options, with exactly one positive GT option and an unambiguous predicted
    option label. Otherwise the original prediction is left unchanged.
    """
    output = dict(pred)
    derivations: List[Dict[str, str]] = []
    yes_tokens = {"yes", "true", "1", "checked", "selected", "present"}
    no_tokens = {"no", "false", "0", "unchecked", "unselected", "absent"}

    groups: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for gt_key, gt_value in gt.items():
        if "." not in gt_key:
            continue
        parent, option = gt_key.rsplit(".", 1)
        value_norm = normalize_value(gt_value)
        if value_norm in yes_tokens or value_norm in no_tokens:
            groups[parent].append((gt_key, option))

    consumed_pred_keys = set()
    for parent, options in groups.items():
        if len(options) < 2:
            continue
        group_gt_values = [normalize_value(gt[k]) for k, _ in options]
        if not all(v in yes_tokens | no_tokens for v in group_gt_values):
            continue
        # Require ground truth to describe exactly one selected option.
        if sum(v in yes_tokens for v in group_gt_values) != 1:
            continue
        parent_norm = norm_key(parent)
        candidates = [
            key for key in output
            if key not in consumed_pred_keys
            and (norm_key(key) == parent_norm or norm_key(leaf_key(key)) == norm_key(leaf_key(parent)))
        ]
        if len(candidates) != 1:
            continue
        pred_parent_key = candidates[0]
        chosen_value = normalize_value(output[pred_parent_key])
        chosen_options = [
            (gt_key, option) for gt_key, option in options
            if normalize_value(option) == chosen_value
        ]
        if len(chosen_options) != 1:
            # Allow harmless option-label punctuation variation as a secondary comparison.
            def compact(text: str) -> str:
                return re.sub(r"[^0-9a-z]+", "", unicodedata.normalize("NFKC", text).lower())
            chosen_options = [
                (gt_key, option) for gt_key, option in options
                if compact(option) == compact(scalar_to_text(output[pred_parent_key]))
            ]
        if len(chosen_options) != 1:
            continue

        chosen_gt_key = chosen_options[0][0]
        yes_repr = next(gt[k] for k, _ in options if normalize_value(gt[k]) in yes_tokens)
        no_repr = next(gt[k] for k, _ in options if normalize_value(gt[k]) in no_tokens)
        del output[pred_parent_key]
        consumed_pred_keys.add(pred_parent_key)
        for gt_key, option in options:
            derived_value = yes_repr if gt_key == chosen_gt_key else no_repr
            output[gt_key] = derived_value
            derivations.append({
                "source_prediction_key": pred_parent_key,
                "source_prediction_value": scalar_to_text(pred[pred_parent_key]),
                "derived_ground_truth_option_key": gt_key,
                "derived_prediction_value": scalar_to_text(derived_value),
                "reason": "single-choice prediction expanded to binary options because ground truth defines a one-of-many group",
            })

    return output, derivations


def unwrap_wrappers(data: Any) -> Any:
    """Remove common one-key wrappers, e.g. {\"fields\": {...}}."""
    while isinstance(data, dict) and len(data) == 1:
        key, value = next(iter(data.items()))
        if norm_key(key) in WRAPPER_KEYS and isinstance(value, (dict, list)):
            data = value
        else:
            break
    return data


def scalar_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def convert_pair_list(data: Any) -> Any:
    """Convert lists such as [{\"key\": ..., \"value\": ...}] to a mapping."""
    data = unwrap_wrappers(data)
    if isinstance(data, list) and data and all(isinstance(item, dict) for item in data):
        for key_field, value_field in PAIR_FIELDS:
            if all(key_field in item and value_field in item for item in data):
                output: Dict[str, Any] = {}
                for item in data:
                    base = str(item[key_field])
                    name, number = base, 2
                    while name in output:
                        name = f"{base}_{number}"
                        number += 1
                    output[name] = item[value_field]
                return output
    return data


def flatten_json(obj: Any, prefix: str = "") -> Dict[str, str]:
    """Flatten nested dictionaries to dotted paths and return all leaf values as strings."""
    output: Dict[str, str] = {}
    if isinstance(obj, dict):
        if not obj and prefix:
            output[prefix] = ""
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            output.update(flatten_json(value, path))
    elif isinstance(obj, list):
        if all(not isinstance(item, (dict, list)) for item in obj):
            output[prefix or "__root__"] = " ".join(scalar_to_text(item) for item in obj)
        else:
            if not obj and prefix:
                output[prefix] = ""
            for index, item in enumerate(obj):
                path = f"{prefix}[{index}]" if prefix else f"[{index}]"
                output.update(flatten_json(item, path))
    else:
        output[prefix or "__root__"] = scalar_to_text(obj)
    return output


def json_to_flat_dict(path: Path) -> Dict[str, str]:
    text = path.read_text(encoding="utf-8-sig").strip()
    # Allow model outputs wrapped in Markdown code fences.
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    parsed = json.loads(text)
    parsed = convert_pair_list(parsed)
    flat = flatten_json(parsed)
    return flat


def normalize_value(value: Any) -> str:
    """Case-insensitive comparison that ignores all whitespace, as requested."""
    text = unicodedata.normalize("NFKC", scalar_to_text(value)).lower()
    text = re.sub(r"\s+", "", text)
    if text in EMPTY_VALUES:
        return ""
    return text


def readable_value(value: Any) -> str:
    """Normalization for embeddings: lowercase and collapse, rather than remove, spaces."""
    text = unicodedata.normalize("NFKC", scalar_to_text(value)).lower()
    return re.sub(r"\s+", " ", text).strip()


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (ca != cb),
            ))
        previous = current
    return previous[-1]


def anls_score(gt_value: str, pred_value: str, tau: float = 0.5) -> float:
    if not gt_value and not pred_value:
        return 1.0
    denominator = max(len(gt_value), len(pred_value))
    if denominator == 0:
        return 1.0
    normalized_distance = levenshtein(gt_value, pred_value) / denominator
    return 1.0 - normalized_distance if normalized_distance < tau else 0.0


def keys_conflict(gt_key: str, pred_key: str) -> bool:
    gt_tokens, pred_tokens = set(norm_key(gt_key).split("_")), set(norm_key(pred_key).split("_"))
    for first, second in CONFLICT_PAIRS:
        if (first in gt_tokens and second in pred_tokens and first not in pred_tokens and second not in gt_tokens):
            return True
        if (second in gt_tokens and first in pred_tokens and second not in pred_tokens and first not in gt_tokens):
            return True
    return False


class TextSimilarity:
    """Optional MiniLM similarity; gracefully falls back to lexical string similarity."""

    def __init__(self, enabled: bool = True, model_name: str = "all-MiniLM-L6-v2"):
        self.model = None
        self.cache: Dict[str, Any] = {}
        self.load_error: Optional[str] = None
        if enabled:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
                self.model = SentenceTransformer(model_name)
            except Exception as exc:  # optional dependency or model download can fail
                self.load_error = str(exc)

    @property
    def using_embeddings(self) -> bool:
        return self.model is not None

    def _encode(self, texts: List[str]) -> Dict[str, Any]:
        missing = list(dict.fromkeys(t for t in texts if t not in self.cache))
        if missing:
            embeddings = self.model.encode(missing, normalize_embeddings=True, show_progress_bar=False)
            for text, vector in zip(missing, embeddings):
                self.cache[text] = vector
        return {text: self.cache[text] for text in texts}

    def cosine(self, left: str, right: str) -> float:
        a_text, b_text = readable_value(left), readable_value(right)
        if not a_text and not b_text:
            return 1.0
        if self.model is not None:
            vectors = self._encode([a_text, b_text])
            a, b = vectors[a_text], vectors[b_text]
            # SentenceTransformer normalized embeddings; dot product is cosine similarity.
            score = float(sum(float(x) * float(y) for x, y in zip(a, b)))
            return max(-1.0, min(1.0, score))
        return SequenceMatcher(None, norm_key(a_text), norm_key(b_text)).ratio()


def align_keys(
    gt: Dict[str, str], pred: Dict[str, str], alias_lookup: Dict[str, str],
    similarity: TextSimilarity, mode: str, threshold: float,
) -> Tuple[List[Tuple[str, str, str, float]], List[str], List[str]]:
    """Return one-to-one aligned (GT key, predicted key, method, score), missing GT and extra pred keys.

    Matching order:
      1. Full normalized path / aliases.
      2. Unique normalized leaf labels / aliases (handles nested GT vs flat prediction).
      3. Semantic or lexical similarity of leaf labels for remaining schema-free keys.
    """
    remaining_gt = list(gt.keys())
    remaining_pred = list(pred.keys())
    matches: List[Tuple[str, str, str, float]] = []

    # Stage 1: match complete normalized paths and aliases.
    pred_by_canonical: Dict[str, List[str]] = defaultdict(list)
    for pred_key in remaining_pred:
        pred_by_canonical[canonical_key_path(pred_key, alias_lookup)].append(pred_key)

    for gt_key in list(remaining_gt):
        candidates = pred_by_canonical.get(canonical_key_path(gt_key, alias_lookup), [])
        candidate = next((key for key in candidates if key in remaining_pred), None)
        if candidate is not None:
            method = "normalized_exact" if norm_key(gt_key) == norm_key(candidate) else "known_alias_path"
            matches.append((gt_key, candidate, method, 1.0))
            remaining_gt.remove(gt_key)
            remaining_pred.remove(candidate)

    # Stage 2: match field labels while ignoring group prefixes, but only when unique
    # on both sides. Ambiguous leaf labels are deliberately not guessed here.
    gt_leaf_canon = {key: canonical_leaf_key(key, alias_lookup) for key in remaining_gt}
    pred_leaf_canon = {key: canonical_leaf_key(key, alias_lookup) for key in remaining_pred}
    gt_counts = Counter(gt_leaf_canon.values())
    pred_counts = Counter(pred_leaf_canon.values())
    for gt_key in list(remaining_gt):
        canon = gt_leaf_canon[gt_key]
        candidates = [p for p in remaining_pred if pred_leaf_canon[p] == canon]
        if len(candidates) == 1 and gt_counts[canon] == 1 and pred_counts[canon] == 1:
            pred_key = candidates[0]
            raw_leaf_match = norm_key(leaf_key(gt_key)) == norm_key(leaf_key(pred_key))
            method = "leaf_normalized_exact" if raw_leaf_match else "leaf_known_alias"
            matches.append((gt_key, pred_key, method, 1.0))
            remaining_gt.remove(gt_key)
            remaining_pred.remove(pred_key)

    # Stage 3: semantic/lexical assignment on leaf labels rather than full dotted paths.
    semantic_enabled = mode in {"schemafree", "auto"}
    if semantic_enabled and remaining_gt and remaining_pred:
        candidates: List[Tuple[float, int, int]] = []
        for i, gt_key in enumerate(remaining_gt):
            for j, pred_key in enumerate(remaining_pred):
                if keys_conflict(leaf_key(gt_key), leaf_key(pred_key)):
                    continue
                score = similarity.cosine(leaf_key(gt_key), leaf_key(pred_key))
                candidates.append((score, i, j))
        candidates.sort(key=lambda item: item[0], reverse=True)
        used_gt, used_pred = set(), set()
        for score, i, j in candidates:
            if score < threshold:
                break
            if i in used_gt or j in used_pred:
                continue
            method = "semantic_embedding_leaf" if similarity.using_embeddings else "fuzzy_lexical_leaf_fallback"
            matches.append((remaining_gt[i], remaining_pred[j], method, float(score)))
            used_gt.add(i)
            used_pred.add(j)
        remaining_gt = [key for i, key in enumerate(remaining_gt) if i not in used_gt]
        remaining_pred = [key for j, key in enumerate(remaining_pred) if j not in used_pred]

    return matches, remaining_gt, remaining_pred

def percent(numerator: int | float, denominator: int | float) -> float:
    return round(100.0 * numerator / denominator, 2) if denominator else 0.0


def document_type_result(gt: Dict[str, str], pred: Dict[str, str], matches: List[Tuple[str, str, str, float]]) -> Optional[Dict[str, Any]]:
    gt_type_keys = [key for key in gt if norm_key(key.split(".")[-1]) in DOCUMENT_TYPE_KEYS]
    if not gt_type_keys:
        return None
    match_by_gt = {g: p for g, p, _, _ in matches}
    key = gt_type_keys[0]
    if key not in match_by_gt:
        return {"field": key, "correct": False, "accuracy_pct": 0.0, "status": "ground-truth type field has no aligned prediction"}
    pred_key = match_by_gt[key]
    correct = normalize_value(gt[key]) == normalize_value(pred[pred_key])
    return {"field": key, "predicted_key": pred_key, "correct": correct, "accuracy_pct": 100.0 if correct else 0.0, "status": "evaluated"}


def evaluate_one(
    filename: str, gt: Dict[str, str], pred: Dict[str, str], pred_for_eval: Dict[str, str],
    mode: str, alias_lookup: Dict[str, str], similarity: TextSimilarity, threshold: float,
    anls_tau: float, status: str, prediction_error: Optional[str] = None,
    derivations: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    derivations = derivations or []
    raw_pred = pred
    pred = pred_for_eval
    matches, missing_gt_keys, extra_pred_keys = align_keys(gt, pred, alias_lookup, similarity, mode, threshold)
    matched_by_gt = {g: (p, method, score) for g, p, method, score in matches}

    correct_fields = 0
    true_positive = false_positive = false_negative = 0
    anls_values: List[float] = []
    semantic_values: List[float] = []
    matched_details: List[Dict[str, Any]] = []
    incorrect_values: List[Dict[str, Any]] = []
    aligned_predictions: Dict[str, str] = {}

    # Put all matched predictions under the ground-truth key to create directly comparable flattened maps.
    for gt_key, pred_key, method, key_score in matches:
        gt_value, pred_value = gt[gt_key], pred[pred_key]
        gt_norm, pred_norm = normalize_value(gt_value), normalize_value(pred_value)
        exact = gt_norm == pred_norm
        if exact:
            correct_fields += 1
            if gt_norm:
                true_positive += 1
        else:
            if gt_norm:
                false_negative += 1
            if pred_norm:
                false_positive += 1

        if gt_norm:
            anls_values.append(anls_score(gt_norm, pred_norm, anls_tau))
            if similarity.using_embeddings:
                semantic_values.append(max(0.0, similarity.cosine(gt_value, pred_value)))
        aligned_predictions[gt_key] = pred_value

        detail = {
            "ground_truth_key": gt_key,
            "predicted_key": pred_key,
            "key_match_method": method,
            "key_match_score": round(key_score, 4),
            "ground_truth_value": gt_value,
            "predicted_value": pred_value,
            "normalized_ground_truth_value": gt_norm,
            "normalized_predicted_value": pred_norm,
            "value_exact_match": exact,
            "anls": round(anls_values[-1], 4) if gt_norm else None,
        }
        if similarity.using_embeddings and gt_norm:
            detail["value_semantic_similarity_cosine"] = round(semantic_values[-1], 4)
        matched_details.append(detail)
        if not exact:
            incorrect_values.append({
                "ground_truth_key": gt_key,
                "predicted_key": pred_key,
                "ground_truth_value": gt_value,
                "predicted_value": pred_value,
                "normalized_ground_truth_value": gt_norm,
                "normalized_predicted_value": pred_norm,
                "reason": "values differ after case-insensitive and whitespace-insensitive normalization",
            })

    missing_from_predictions = []
    for gt_key in missing_gt_keys:
        gt_value = gt[gt_key]
        gt_norm = normalize_value(gt_value)
        if gt_norm:
            false_negative += 1
            anls_values.append(anls_score(gt_norm, "", anls_tau))
            if similarity.using_embeddings:
                semantic_values.append(0.0)
        missing_from_predictions.append({
            "key": gt_key,
            "value": gt_value,
            "normalized_value": gt_norm,
            "reason": "no corresponding predicted key could be aligned",
        })
        # Add an empty placeholder to the aligned flat map so its expected fields
        # line up with the ground truth. The missing_from_predictions list still
        # distinguishes an absent key from a key that was explicitly predicted empty.
        aligned_predictions.setdefault(gt_key, "")

    extra_in_predictions = []
    for pred_key in extra_pred_keys:
        pred_value = pred[pred_key]
        pred_norm = normalize_value(pred_value)
        if pred_norm:
            false_positive += 1
        extra_in_predictions.append({
            "key": pred_key,
            "value": pred_value,
            "normalized_value": pred_norm,
            "reason": "predicted key has no corresponding ground-truth key",
        })
        aligned_name = pred_key
        if aligned_name in aligned_predictions:
            suffix = 2
            while f"{pred_key}__extra_{suffix}" in aligned_predictions:
                suffix += 1
            aligned_name = f"{pred_key}__extra_{suffix}"
        aligned_predictions[aligned_name] = pred_value

    total_gt_fields = len(gt)
    matched_fields = len(matches)
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    document_exact = (
        len(missing_gt_keys) == 0
        and len(extra_pred_keys) == 0
        and all(normalize_value(gt[g]) == normalize_value(pred[p]) for g, p, _, _ in matches)
    )
    doc_type = document_type_result(gt, pred, matches)

    key_method_counts = dict(Counter(method for _, _, method, _ in matches))
    metrics: Dict[str, Any] = {
        "total_ground_truth_fields": total_gt_fields,
        "total_prediction_fields": len(raw_pred),
        "matched_key_count": matched_fields,
        "correct_fields": correct_fields,
        "field_level_extraction_accuracy_pct": percent(correct_fields, total_gt_fields),
        "matched_field_value_exact_match_pct": percent(
            sum(1 for g, p, _, _ in matches if normalize_value(gt[g]) == normalize_value(pred[p])), matched_fields
        ),
        "document_exact_match": bool(document_exact),
        "document_exact_match_pct": 100.0 if document_exact else 0.0,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision_pct": round(100 * precision, 2),
        "recall_pct": round(100 * recall, 2),
        "f1_pct": round(100 * f1, 2),
        "anls_pct": round(100 * sum(anls_values) / len(anls_values), 2) if anls_values else 0.0,
        "anls_fields_scored": len(anls_values),
        "semantic_similarity_pct": round(100 * sum(semantic_values) / len(semantic_values), 2) if semantic_values else None,
        "semantic_similarity_fields_scored": len(semantic_values),
        "document_type_classification": doc_type,
        "key_match_methods": key_method_counts,
    }

    # Keep the per-file report deliberately compact and consistent for every dataset.
    # Raw predictions remain unchanged in predictions_flattened; structural
    # interpretations are only applied to the evaluation view.
    report: Dict[str, Any] = {
        "file": filename,
        "status": status,
        "prediction_error": prediction_error,
        "ground_truth_flattened": gt,
        "predictions_flattened": raw_pred,
        "evaluation": {
            "mode": mode,
            "normalization": {
                "keys": "case-insensitive; whitespace/punctuation normalized; nested parent prefixes ignored when unique leaf labels match; common aliases used",
                "values": "Unicode NFKC, lowercase, all whitespace removed; punctuation preserved",
                "semantic_key_matching": mode in {"schemafree", "auto"},
                "semantic_key_threshold": threshold,
                "semantic_backend": "all-MiniLM-L6-v2" if similarity.using_embeddings else ("lexical fallback; review unmatched/different key labels" if mode in {"schemafree", "auto"} else "not used in schema mode"),
                "anls_threshold_tau": anls_tau,
                "single_choice_options": "when the ground truth defines one positive option in a binary option group, a matching selected-option prediction may be evaluated as Yes/No values",
            },
            "metrics": metrics,
            "incorrect_values": incorrect_values,
            "key_match_methods": key_method_counts,
        },
        "missing_from_predictions": [
            {"key": item["key"], "ground_truth_value": item["value"]}
            for item in missing_from_predictions
        ],
        "extra_in_predictions": [
            {"key": item["key"], "predicted_value": item["value"]}
            for item in extra_in_predictions
        ],
    }
    return report


def read_alias_file(path: Optional[str]) -> Dict[str, List[str]]:
    aliases = {key: list(value) for key, value in DEFAULT_ALIASES.items()}
    if not path:
        return aliases
    custom = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(custom, dict):
        raise ValueError("Aliases JSON must be an object mapping canonical keys to lists of aliases.")
    for canonical, values in custom.items():
        if not isinstance(values, list):
            raise ValueError(f"Alias list for {canonical!r} must be a JSON list.")
        aliases.setdefault(str(canonical), [])
        aliases[str(canonical)] = list(dict.fromkeys(aliases[str(canonical)] + [str(v) for v in values]))
    return aliases


def ask_if_missing(value: Optional[str], prompt: str) -> str:
    if value and value.strip():
        return value.strip().strip('"')
    return input(prompt).strip().strip('"')


# def find_prediction_file(gt_path: Path, gt_root: Path, pred_root: Path, stem_index: Dict[str, List[Path]]) -> Optional[Path]:
#     relative = gt_path.relative_to(gt_root)
#     exact_path = pred_root / relative
#     if exact_path.is_file():
#         return exact_path
#     # Fallback: same basename (unique only) if directory structures differ.
#     candidates = stem_index.get(gt_path.stem, [])
#     return candidates[0] if len(candidates) == 1 else None
def find_prediction_file(
    gt_path: Path,
    gt_root: Path,
    pred_root: Path,
    stem_index: Dict[str, List[Path]]
) -> Optional[Path]:

    relative = gt_path.relative_to(gt_root)

    # Ground truth: 1.json
    # Prediction:   1_qwen_only.json
    prediction_name = f"{gt_path.stem}_qwen_only.json"

    # Preserve the ground-truth subfolder structure.
    exact_path = pred_root / relative.parent / prediction_name

    if exact_path.is_file():
        return exact_path

    # Fallback: search by the expected prediction filename.
    candidates = stem_index.get(
        f"{gt_path.stem}_qwen_only", []
    )

    return candidates[0] if len(candidates) == 1 else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate per-file prediction JSON against ground-truth JSON.")
    parser.add_argument("--gt_dir", help="Folder containing ground-truth .json files")
    parser.add_argument("--pred_dir", help="Folder containing prediction .json files")
    parser.add_argument("--out_dir", help="Folder where per-file evaluation JSON will be saved")
    parser.add_argument("--mode", choices=["schema", "schemafree", "auto"], help="Key-alignment mode")
    parser.add_argument("--threshold", type=float, default=0.80, help="Similarity threshold for schema-free key matching (default: 0.80)")
    parser.add_argument("--anls_tau", type=float, default=0.5, help="ANLS cutoff threshold (default: 0.5)")
    parser.add_argument("--aliases", help="Optional JSON file containing additional key aliases")
    parser.add_argument("--no_embeddings", action="store_true", help="Do not attempt to load all-MiniLM-L6-v2")
    args = parser.parse_args()

    try:
        gt_dir = Path(ask_if_missing(args.gt_dir, "Enter the ground-truth JSON folder path: ")).expanduser().resolve()
        pred_dir = Path(ask_if_missing(args.pred_dir, "Enter the predictions JSON folder path: ")).expanduser().resolve()
        out_dir = Path(ask_if_missing(args.out_dir, "Enter the output folder path: ")).expanduser().resolve()
        mode = args.mode or input("Mode [schema / schemafree / auto] (default auto): ").strip().lower() or "auto"
        if mode not in {"schema", "schemafree", "auto"}:
            raise ValueError("Mode must be schema, schemafree, or auto.")
        if not gt_dir.is_dir():
            raise NotADirectoryError(f"Ground-truth folder does not exist: {gt_dir}")
        if not pred_dir.is_dir():
            raise NotADirectoryError(f"Predictions folder does not exist: {pred_dir}")
        if not 0.0 <= args.threshold <= 1.0:
            raise ValueError("--threshold must be between 0 and 1.")
        if not 0.0 < args.anls_tau <= 1.0:
            raise ValueError("--anls_tau must be greater than 0 and at most 1.")

        aliases = read_alias_file(args.aliases)
        alias_lookup = build_alias_lookup(aliases)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Collect input files before writing outputs and skip any output JSONs if output is nested in GT.
        gt_files = []
        for path in gt_dir.rglob("*.json"):
            try:
                path.resolve().relative_to(out_dir)
                continue
            except ValueError:
                pass
            gt_files.append(path)
        gt_files.sort()
        if not gt_files:
            raise FileNotFoundError(f"No ground-truth .json files found under: {gt_dir}")

        pred_stem_index: Dict[str, List[Path]] = defaultdict(list)
        for path in pred_dir.rglob("*.json"):
            pred_stem_index[path.stem].append(path)

        wants_semantics = mode in {"schemafree", "auto"} and not args.no_embeddings
        similarity = TextSimilarity(enabled=wants_semantics)
        if wants_semantics and not similarity.using_embeddings:
            print("[WARNING] all-MiniLM-L6-v2 is unavailable. Key matching will use aliases plus a lexical fallback, and semantic value similarity will be null.")
            if similarity.load_error:
                print(f"          Details: {similarity.load_error[:300]}")
        elif similarity.using_embeddings:
            print("Loaded all-MiniLM-L6-v2 for semantic key alignment and supplementary value similarity.")

        saved = 0
        dataset_reports: List[Dict[str, Any]] = []
        for gt_path in gt_files:
            relative = gt_path.relative_to(gt_dir)
            pred_path = find_prediction_file(gt_path, gt_dir, pred_dir, pred_stem_index)
            status = "ok"
            prediction_error = None
            try:
                gt_flat = json_to_flat_dict(gt_path)
            except Exception as exc:
                error_report = {
                    "file": str(relative), "status": "invalid_ground_truth_json", "error": str(exc),
                    "note": "This file could not be evaluated because its ground-truth JSON could not be loaded or flattened."
                }
                destination = out_dir / relative.parent / f"{gt_path.stem}_evaluation.json"
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(json.dumps(error_report, indent=2, ensure_ascii=False), encoding="utf-8")
                dataset_reports.append(error_report)
                saved += 1
                print(f"[ERROR] {relative}: invalid ground truth JSON ({exc})")
                continue

            pred_flat: Dict[str, str] = {}
            if pred_path is None:
                status = "missing_prediction_file"
                prediction_error = "No prediction JSON file with the same relative path or unique matching basename was found."
            else:
                try:
                    pred_flat = json_to_flat_dict(pred_path)
                except Exception as exc:
                    status = "invalid_prediction_json"
                    prediction_error = f"Could not load prediction JSON: {exc}"

            pred_eval_flat, derivations = expand_single_choice_prediction(gt_flat, pred_flat)
            report = evaluate_one(
                filename=str(relative), gt=gt_flat, pred=pred_flat, pred_for_eval=pred_eval_flat,
                mode=mode, alias_lookup=alias_lookup, similarity=similarity,
                threshold=args.threshold, anls_tau=args.anls_tau,
                status=status, prediction_error=prediction_error, derivations=derivations,
            )
            destination = out_dir / relative.parent / f"{gt_path.stem}_evaluation.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
            dataset_reports.append(report)
            saved += 1
            m = report["evaluation"]["metrics"]
            print(
                f"[OK] {relative} -> {destination.name} | "
                f"FLA={m['field_level_extraction_accuracy_pct']}% | "
                f"F1={m['f1_pct']}% | ANLS={m['anls_pct']}% | "
                f"missing={len(report['missing_from_predictions'])} | "
                f"extra={len(report['extra_in_predictions'])} | "
                f"incorrect_values={len(report['evaluation']['incorrect_values'])}"
            )

        # Dataset-level micro summary across all valid per-file evaluation reports.
        valid_reports = [r for r in dataset_reports if isinstance(r.get("evaluation"), dict) and isinstance(r["evaluation"].get("metrics"), dict)]
        metric_rows = [r["evaluation"]["metrics"] for r in valid_reports]
        total_gt_fields = sum(m.get("total_ground_truth_fields", 0) for m in metric_rows)
        total_correct_fields = sum(m.get("correct_fields", 0) for m in metric_rows)
        total_tp = sum(m.get("true_positive", 0) for m in metric_rows)
        total_fp = sum(m.get("false_positive", 0) for m in metric_rows)
        total_fn = sum(m.get("false_negative", 0) for m in metric_rows)
        total_matched = sum(m.get("matched_key_count", 0) for m in metric_rows)
        total_exact_matches = sum(bool(m.get("document_exact_match", False)) for m in metric_rows)
        total_anls_fields = sum(m.get("anls_fields_scored", 0) for m in metric_rows)
        total_anls_sum = sum((m.get("anls_pct", 0.0) / 100.0) * m.get("anls_fields_scored", 0) for m in metric_rows)
        precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
        recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        summary = {
            "mode": mode,
            "documents_found": len(gt_files),
            "documents_evaluated": len(valid_reports),
            "documents_with_invalid_ground_truth": sum(r.get("status") == "invalid_ground_truth_json" for r in dataset_reports),
            "documents_with_missing_or_invalid_predictions": sum(
                r.get("status") in {"missing_prediction_file", "invalid_prediction_json"} for r in valid_reports
            ),
            "total_ground_truth_fields": total_gt_fields,
            "total_correct_fields": total_correct_fields,
            "field_level_extraction_accuracy_pct": percent(total_correct_fields, total_gt_fields),
            "matched_key_count": total_matched,
            "matched_field_value_exact_match_pct": percent(total_correct_fields, total_matched),
            "documents_exact_match": total_exact_matches,
            "document_exact_match_pct": percent(total_exact_matches, len(valid_reports)),
            "true_positive": total_tp,
            "false_positive": total_fp,
            "false_negative": total_fn,
            "precision_pct": round(100 * precision, 2),
            "recall_pct": round(100 * recall, 2),
            "f1_pct": round(100 * f1, 2),
            "anls_pct": round(100 * total_anls_sum / total_anls_fields, 2) if total_anls_fields else 0.0,
            "anls_fields_scored": total_anls_fields,
            "semantic_backend": "all-MiniLM-L6-v2" if similarity.using_embeddings else (
                "unavailable; lexical fallback for unmatched schema-free keys" if mode in {"schemafree", "auto"}
                else "not used in schema mode"
            ),
            "per_file_reports": [r.get("file") for r in dataset_reports],
            "note": "Aggregate metrics are micro-averaged over fields, except document exact-match which is document-level. Review semantic/fuzzy key alignments for ambiguous schema-free fields.",
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

        print(f"\nFinished. Wrote {saved} per-file evaluation JSON file(s) to: {out_dir}")
        print(f"Dataset summary: {out_dir / 'summary.json'}")
        print(f"Aggregate FLA={summary['field_level_extraction_accuracy_pct']}% | F1={summary['f1_pct']}% | ANLS={summary['anls_pct']}% | Document EM={summary['document_exact_match_pct']}%")
        print("Review semantic key matches in each JSON before trusting schema-free scores, especially if the lexical fallback was used.")
        return 0
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nCancelled by user.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
