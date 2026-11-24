"""Key-value extraction metrics (ported from Metric_new.py, the team's evaluator).

Reports only:
1. Combined Key-Value Precision
2. Combined Key-Value Recall
3. Combined Key-Value F1
4. Whole-form Normalized Edit Distance similarity (NED)
5. Strict Form Accuracy

Combined key-value metrics:
- A key-value pair is a TP only when its key is aligned and its value is exactly
  equal after domain-aware value normalization.
- Incorrect aligned values count as one FP and one FN.
- Extra predicted keys count as FP; missing ground-truth keys count as FN.
- Dataset precision/recall/F1 are micro-averaged across documents.

NED: 1 - Levenshtein(canonical GT form, canonical prediction form) / max(lengths),
in percent; the dataset value is macro-averaged over documents.

Strict Form Accuracy: a document is correct only if every GT key is matched,
there are no extra predicted keys, and every value matches after NFKC,
lowercasing and whitespace collapsing.

Key alignment mode: "schema" (alias / exact matches only), "schemafree" or
"auto" (also lexical matching above KEY_THRESHOLD).
"""
import json
import re
import unicodedata
from collections import Counter
from datetime import date
from decimal import Decimal, InvalidOperation
from difflib import SequenceMatcher
from pathlib import Path

KEY_THRESHOLD = 0.80   # lexical key threshold (as in the script's default)

DEFAULT_ALIASES = {
    "patient_name": ["name_of_patient", "patients_name", "pt_name", "patient_full_name"],
    "age": ["patient_age", "age_years", "age_of_patient", "years_old"],
    "sex": ["gender", "patient_sex", "patient_gender"],
    "date_of_admission": ["admission_date", "admit_date", "date_admitted"],
    "date_of_discharge": ["discharge_date", "date_discharged"],
    "hospital_name": ["hospital", "name_of_hospital", "facility_name", "medical_center"],
    "account_number": ["account_no", "acc_no", "a_c_no", "ac_no", "account_num"],
    "cheque_number": ["cheque_no", "check_number", "check_no", "chq_no"],
    "amount_in_words": ["amount_words", "legal_amount"],
    "amount_in_figures": ["amount_figures", "courtesy_amount", "amount_numeric"],
    "payee_name": ["payee", "pay_to", "beneficiary_name"],
    "invoice_number": ["invoice_no", "inv_no", "bill_number", "bill_no"],
    "invoice_date": ["date_of_invoice", "bill_date"],
    "company": ["company_name", "vendor", "vendor_name", "seller", "store_name"],
    "address": ["company_address", "vendor_address", "store_address"],
    "total": ["total_amount", "grand_total", "amount_due", "net_total"],
    "document_type": ["doc_type", "document_class", "form_type", "document_category"],
}
WRAPPERS = {"fields", "data", "extracted_fields", "result", "results", "output",
            "key_value_pairs", "form", "extraction"}
PAIR_FIELDS = [("key", "value"), ("question", "answer"), ("field", "value"), ("label", "text")]
CONFLICT_PAIRS = [("admission", "discharge"), ("start", "end"), ("from", "to"),
                  ("issue", "due"), ("billing", "shipping"), ("sender", "receiver"),
                  ("payer", "payee"), ("debit", "credit"), ("first", "last"),
                  ("birth", "death"), ("subtotal", "total"), ("words", "figures"),
                  ("min", "max")]


def norm_key(x):
    s = unicodedata.normalize("NFKC", str(x))
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    return re.sub(r"[^0-9A-Za-z]+", "_", s).strip("_").casefold()


def leaf_key(k):
    return str(k).rsplit(".", 1)[-1].replace("]", "").split("[")[0]


def make_alias_lookup(aliases):
    result = {}
    for canonical, variants in aliases.items():
        c = norm_key(canonical)
        result[c] = c
        for alias in variants:
            result[norm_key(alias)] = c
    return result


def canonical_path(key, lookup):
    parts = re.split(r"([.\[\]])", str(key))
    return "".join(
        part if part in {".", "[", "]", ""} else lookup.get(norm_key(part), norm_key(part))
        for part in parts
    )


def scalar(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def canonical_number(value):
    try:
        number = Decimal(str(value))
        if not number.is_finite():
            return str(value)
        result = format(number.normalize(), "f")
        if "." in result:
            result = result.rstrip("0").rstrip(".")
        return "0" if result in {"", "-0"} else result
    except (InvalidOperation, ValueError):
        return str(value)


def normalize_value(value):
    s = unicodedata.normalize("NFKC", scalar(value)).strip().casefold()
    s = re.sub(r"\s+", " ", s)
    if s in {"", "n/a", "na", "none", "null", "nil", "-", "--"}:
        return ""
    if s in {"m", "male"}:
        return "male"
    if s in {"f", "female"}:
        return "female"
    age = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(?:years?|yrs?|y\.?o\.?)", s)
    if age:
        return canonical_number(age.group(1))
    dm = re.fullmatch(r"(\d{1,2})[/-](\d{1,2})[/-](\d{2}|\d{4})", s)
    if dm:
        day, month, year = map(int, dm.groups())
        if year < 100:
            year += 2000 if year <= 68 else 1900
        try:
            return date(year, month, day).isoformat()
        except ValueError:
            pass
    numeric = re.sub(r"^(?:rm|rs\.?|inr|usd|₹|\$)\s*", "", s).strip()
    compact = numeric.replace(",", "").replace(" ", "")
    if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", compact):
        return canonical_number(compact)
    return re.sub(r"\s+", "", s)


def strict_text(value):
    """Only NFKC + lowercase + whitespace collapsing for Strict Form Accuracy."""
    s = unicodedata.normalize("NFKC", scalar(value)).casefold()
    return re.sub(r"\s+", " ", s).strip()


def unwrap(data):
    while isinstance(data, dict) and len(data) == 1:
        key, value = next(iter(data.items()))
        if norm_key(key) in WRAPPERS and isinstance(value, (dict, list)):
            data = value
        else:
            break
    return data


def flatten(obj, prefix=""):
    out = {}
    if isinstance(obj, dict):
        obj = unwrap(obj)
        for key, value in obj.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            out.update(flatten(value, child))
        if not obj and prefix:
            out[prefix] = ""
    elif isinstance(obj, list):
        if obj and all(not isinstance(x, (dict, list)) for x in obj):
            out[prefix or "__root__"] = " ".join(scalar(x) for x in obj)
        elif obj and all(isinstance(x, dict) for x in obj):
            for kf, vf in PAIR_FIELDS:
                if all(kf in item and vf in item for item in obj):
                    for i, item in enumerate(obj):
                        base = str(item[kf]).strip() or f"item_{i+1}"
                        name, n = base, 2
                        while name in out:
                            name, n = f"{base}_{n}", n + 1
                        out[name] = scalar(item[vf])
                    return out
            for i, item in enumerate(obj):
                out.update(flatten(item, f"{prefix}[{i}]" if prefix else f"item_{i}"))
        elif not obj and prefix:
            out[prefix] = ""
        else:
            for i, item in enumerate(obj):
                out.update(flatten(item, f"{prefix}[{i}]" if prefix else f"item_{i}"))
    else:
        out[prefix or "__root__"] = scalar(obj)
    return out


def load_flat(path):
    raw = path.read_text(encoding="utf-8-sig").strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.I)
    return flatten(json.loads(raw))


def has_conflicting_semantics(a, b):
    ta, tb = set(norm_key(a).split("_")), set(norm_key(b).split("_"))
    return any((x in ta and y in tb) or (y in ta and x in tb) for x, y in CONFLICT_PAIRS)


def align_keys(gt, pred, alias_lookup, mode, threshold):
    gt_left, pred_left, matches = list(gt), list(pred), []
    pred_groups = {}
    for p in pred_left:
        pred_groups.setdefault(canonical_path(p, alias_lookup), []).append(p)
    for g in list(gt_left):
        candidates = pred_groups.get(canonical_path(g, alias_lookup), [])
        p = next((x for x in candidates if x in pred_left), None)
        if p is not None:
            matches.append((g, p))
            gt_left.remove(g)
            pred_left.remove(p)

    gs = {g: alias_lookup.get(norm_key(leaf_key(g)), norm_key(leaf_key(g))) for g in gt_left}
    ps = {p: alias_lookup.get(norm_key(leaf_key(p)), norm_key(leaf_key(p))) for p in pred_left}
    gc, pc = Counter(gs.values()), Counter(ps.values())
    for g in list(gt_left):
        sig = gs[g]
        candidates = [p for p in pred_left if ps[p] == sig]
        if gc[sig] == 1 and pc[sig] == 1 and len(candidates) == 1:
            p = candidates[0]
            matches.append((g, p))
            gt_left.remove(g)
            pred_left.remove(p)

    if mode in {"schemafree", "auto"}:
        scored = []
        for i, g in enumerate(gt_left):
            for j, p in enumerate(pred_left):
                if not has_conflicting_semantics(leaf_key(g), leaf_key(p)):
                    score = SequenceMatcher(None, norm_key(leaf_key(g)),
                                             norm_key(leaf_key(p))).ratio()
                    scored.append((score, i, j))
        used_g, used_p = set(), set()
        for score, i, j in sorted(scored, reverse=True):
            if score < threshold:
                break
            if i not in used_g and j not in used_p:
                matches.append((gt_left[i], pred_left[j]))
                used_g.add(i)
                used_p.add(j)
        gt_left = [x for i, x in enumerate(gt_left) if i not in used_g]
        pred_left = [x for j, x in enumerate(pred_left) if j not in used_p]
    return matches, gt_left, pred_left


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return round(100*p, 2), round(100*r, 2), round(100*f, 2)


def levenshtein(a, b):
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j-1] + 1,
                               previous[j-1] + (ca != cb)))
        previous = current
    return previous[-1]


def canonical_form(flat, alias_lookup):
    items = [{"key": canonical_path(k, alias_lookup), "value": normalize_value(v)}
             for k, v in flat.items()]
    items.sort(key=lambda x: (x["key"], x["value"]))
    return json.dumps(items, ensure_ascii=False, separators=(",", ":"))


def ned_similarity(gt, pred, alias_lookup):
    g, p = canonical_form(gt, alias_lookup), canonical_form(pred, alias_lookup)
    longest = max(len(g), len(p))
    return 100.0 if longest == 0 else round(100 * (1 - levenshtein(g, p) / longest), 2)


def evaluate_one(gt, pred, alias_lookup, mode, threshold):
    matches, missing, extra = align_keys(gt, pred, alias_lookup, mode, threshold)
    tp = fp = fn = 0
    for g, p in matches:
        if normalize_value(gt[g]) == normalize_value(pred[p]):
            tp += 1
        else:
            fp += 1
            fn += 1
    fp += len(extra)
    fn += len(missing)
    precision, recall, f1 = prf(tp, fp, fn)

    # Strict whole-form check: every GT key matched, no extra predicted keys,
    # and every matched value is character-identical after strict_text().
    strict = (not missing and not extra and len(gt) == len(pred) and
              all(strict_text(gt[g]) == strict_text(pred[p]) for g, p in matches))
    ned = ned_similarity(gt, pred, alias_lookup)
    return {
        "combined_kv_precision_pct": precision,
        "combined_kv_recall_pct": recall,
        "combined_kv_f1_pct": f1,
        "normalized_edit_distance_similarity_pct": ned,
        "strict_form_accuracy_pct": 100.0 if strict else 0.0,
        "strict_form_correct": strict,
        "ground_truth_fields": len(gt),
        "prediction_fields": len(pred),
        "matched_key_pairs": len(matches),
        "missing_keys": len(missing),
        "extra_keys": len(extra),
        "incorrect_values": sum(1 for g, p in matches
                                if normalize_value(gt[g]) != normalize_value(pred[p])),
        "tp": tp, "fp": fp, "fn": fn,
    }


def load_aliases(alias_file=None):
    aliases = {k: list(v) for k, v in DEFAULT_ALIASES.items()}
    if alias_file:
        custom = json.loads(Path(alias_file).read_text(encoding="utf-8"))
        for key, values in custom.items():
            if not isinstance(values, list):
                raise ValueError(f"Aliases for {key} must be a JSON list.")
            aliases.setdefault(str(key), [])
            aliases[str(key)].extend(str(x) for x in values)
    return aliases


def evaluate_pair(gt_obj, pred_obj, mode="auto", threshold=KEY_THRESHOLD, alias_lookup=None):
    """Score one document. gt_obj / pred_obj are parsed JSON (nested or flat); pred_obj may be None."""
    lookup = alias_lookup or make_alias_lookup(DEFAULT_ALIASES)
    gt = flatten(gt_obj)
    pred = flatten(pred_obj) if pred_obj else {}
    return evaluate_one(gt, pred, lookup, mode, threshold)


def aggregate_rows(rows):
    """Dataset summary exactly as in Metric_new.py's evaluate_folder()."""
    total_tp = sum(r["tp"] for r in rows)
    total_fp = sum(r["fp"] for r in rows)
    total_fn = sum(r["fn"] for r in rows)
    precision, recall, f1 = prf(total_tp, total_fp, total_fn)
    count = len(rows)
    strict_correct = sum(int(r["strict_form_correct"]) for r in rows)
    ned = [r["normalized_edit_distance_similarity_pct"] for r in rows]
    return {
        "combined_kv_precision_pct": precision,
        "combined_kv_recall_pct": recall,
        "combined_kv_f1_pct": f1,
        "normalized_edit_distance_similarity_pct": round(sum(ned) / len(ned), 2) if ned else 0.0,
        "strict_form_accuracy_pct": round(100 * strict_correct / count, 2) if count else 0.0,
        "strict_form_correct_documents": strict_correct,
        "documents_evaluated": count,
        "combined_kv_tp": total_tp,
        "combined_kv_fp": total_fp,
        "combined_kv_fn": total_fn,
    }


def field_rows(gt_obj, pred_obj, mode="auto", threshold=KEY_THRESHOLD, alias_lookup=None):
    """Per-field outcome for the UI diff: correct / wrong / missing / extra."""
    lookup = alias_lookup or make_alias_lookup(DEFAULT_ALIASES)
    gt = flatten(gt_obj)
    pred = flatten(pred_obj) if pred_obj else {}
    matches, missing, extra = align_keys(gt, pred, lookup, mode, threshold)
    rows = []
    for g, p in matches:
        ok = normalize_value(gt[g]) == normalize_value(pred[p])
        rows.append({"outcome": "correct" if ok else "wrong", "gt_key": g, "gt_value": gt[g],
                     "pred_key": p, "pred_value": pred[p]})
    rows += [{"outcome": "missing", "gt_key": g, "gt_value": gt[g], "pred_key": "", "pred_value": ""}
             for g in missing]
    rows += [{"outcome": "extra", "gt_key": "", "gt_value": "", "pred_key": p, "pred_value": pred[p]}
             for p in extra]
    return rows
