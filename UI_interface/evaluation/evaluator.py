"""Per-document and aggregate scoring.

Every ground-truth field is counted once:
  TP = matched key with a correct value
  FP = matched key with a wrong value, or an extra predicted key
  FN = ground-truth key missing from the prediction

Aggregates are micro-averaged over all fields; document exact match is
counted per document. Metrics that are not computed are None (shown as
"not computed"), never 0.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from evaluation.key_matching import match_keys
from evaluation.metrics import cer, nls, normalize_value, safe_div, value_exact, wer
from validation.json_parser import flatten


@dataclass
class FieldResult:
    gt_key: str
    pred_key: str | None
    gt_value: Any
    pred_value: Any
    outcome: str        # correct | wrong | missing | extra
    anls: float


@dataclass
class DocResult:
    document: str
    tp: int = 0
    fp: int = 0
    fn: int = 0
    n_gt: int = 0
    n_matched: int = 0
    anls_sum: float = 0.0
    fields: list[FieldResult] = field(default_factory=list)
    status: str = ""
    cer: float | None = None
    wer: float | None = None
    line_exact: float | None = None

    @property
    def exact(self) -> bool:
        return self.n_gt > 0 and self.tp == self.n_gt and self.fp == 0

    def summary(self) -> dict[str, Any]:
        p = safe_div(self.tp, self.tp + self.fp)
        r = safe_div(self.tp, self.tp + self.fn)
        return {"document": self.document, "status": self.status, "gt_fields": self.n_gt, "tp": self.tp,
                "fp": self.fp, "fn": self.fn, "fla": safe_div(self.tp, self.n_gt),
                "precision": p, "recall": r, "f1": safe_div(2 * p * r, p + r) if p is not None and r is not None else None,
                "anls": safe_div(self.anls_sum, self.n_gt), "exact": self.exact if self.n_gt else None,
                "cer": self.cer, "wer": self.wer}


def evaluate_document(name: str, gt: dict[str, Any], pred: dict[str, Any] | None, fuzzy_threshold: float = 0.85,
                      anls_threshold: float = 0.5, flatten_separator: str = " - ") -> DocResult:
    gt_flat = flatten(gt, flatten_separator)
    pred_flat = flatten(pred or {}, flatten_separator)
    mapping = match_keys(list(gt_flat), list(pred_flat), fuzzy_threshold)
    doc = DocResult(document=name, n_gt=len(gt_flat), n_matched=len(mapping))
    for gk, gv in gt_flat.items():
        pk = mapping.get(gk)
        if pk is None:
            doc.fn += 1
            doc.fields.append(FieldResult(gk, None, gv, None, "missing", 0.0))
            continue
        pv = pred_flat[pk]
        score = nls(gv, pv, anls_threshold)
        doc.anls_sum += score
        if value_exact(gv, pv):
            doc.tp += 1
            doc.fields.append(FieldResult(gk, pk, gv, pv, "correct", score))
        else:
            doc.fp += 1
            doc.fields.append(FieldResult(gk, pk, gv, pv, "wrong", score))
    used = set(mapping.values())
    for pk, pv in pred_flat.items():
        if pk not in used:
            doc.fp += 1
            doc.fields.append(FieldResult("", pk, None, pv, "extra", 0.0))
    return doc


def evaluate_transcription(name: str, gt_text: str, pred_text: str | None) -> DocResult:
    pred_text = pred_text or ""
    gt_lines = [ln.strip() for ln in gt_text.strip().splitlines() if ln.strip()]
    pred_lines = [ln.strip() for ln in pred_text.strip().splitlines() if ln.strip()]
    matches = sum(1 for i, g in enumerate(gt_lines) if i < len(pred_lines) and pred_lines[i] == g)
    gt_joined, pred_joined = " ".join(gt_lines), " ".join(pred_lines)
    doc = DocResult(document=name, cer=cer(gt_joined, pred_joined), wer=wer(gt_joined, pred_joined),
                    line_exact=safe_div(matches, len(gt_lines)))
    doc.n_gt = 1
    doc.anls_sum = nls(gt_joined, pred_joined, 0.0)
    if normalize_value(gt_joined) == normalize_value(pred_joined):
        doc.tp = 1
    elif pred_joined:
        doc.fp = 1
    else:
        doc.fn = 1
    return doc


def aggregate(docs: list[DocResult], mode: str = "extraction") -> dict[str, Any]:
    tp = sum(d.tp for d in docs)
    fp = sum(d.fp for d in docs)
    fn = sum(d.fn for d in docs)
    n_gt = sum(d.n_gt for d in docs)
    matched = sum(d.n_matched for d in docs)
    p, r = safe_div(tp, tp + fp), safe_div(tp, tp + fn)
    out: dict[str, Any] = {
        "documents": len(docs),
        "gt_fields": n_gt,
        "tp": tp, "fp": fp, "fn": fn,
        "fla": safe_div(tp, n_gt),
        "matched_value_exact_match": safe_div(tp, matched) if mode != "transcription" else None,
        "precision": p,
        "recall": r,
        "f1": safe_div(2 * p * r, p + r) if p is not None and r is not None else None,
        "anls": safe_div(sum(d.anls_sum for d in docs), n_gt),
        "document_exact_match": safe_div(sum(1 for d in docs if d.exact), len(docs)),
        "semantic_similarity": None,       # not computed (matches the report)
        "document_type_accuracy": None,    # not computed
        "cer": None, "wer": None, "line_exact_match": None,
    }
    if mode == "transcription":
        vals = [d for d in docs if d.cer is not None]
        out["cer"] = safe_div(sum(d.cer for d in vals), len(vals))
        out["wer"] = safe_div(sum(d.wer for d in vals), len(vals))
        le = [d.line_exact for d in vals if d.line_exact is not None]
        out["line_exact_match"] = safe_div(sum(le), len(le))
    return out


def doc_fields_table(doc: DocResult) -> list[dict[str, Any]]:
    return [asdict(f) for f in doc.fields]
