"""Explicit submission semantics and ID-based template alignment."""
from pathlib import Path
import numpy as np
import pandas as pd
from .data import read_table
from .metrics import labels_from_proba
from .tracking import frame_fingerprint


def make_submission(result, test, config, path=None):
    p = result.test_predictions
    if p is None or len(p) != len(test) or not np.isfinite(p).all():
        raise ValueError("Test predictions must be finite and match the test row count")
    if result.test_hash != frame_fingerprint(test):
        raise ValueError("Submission test rows/order differ from the prediction input")
    s, ids = config["submission"], config["data"].get("id_columns", [])
    names, kind = s["prediction_columns"], s.get("kind", "value")
    if not names or len(set(names)) != len(names) or set(names) & set(ids):
        raise ValueError("prediction_columns must be unique and distinct from id_columns")
    if result.task == "classification":
        if p.ndim != 2 or p.shape[1] != len(result.classes) or (p < 0).any() or (p > 1).any() or not np.allclose(p.sum(axis=1), 1.):
            raise ValueError("Invalid classification probability matrix")
        if kind == "label":
            values = result.classes[labels_from_proba(p, config["metric"].get("threshold", .5))][:, None]
        elif kind == "probability":
            if len(names) == p.shape[1]:
                values = p
            elif p.shape[1] == 2 and len(names) == 1:
                positive = s.get("positive_class")
                positive = result.classes[1] if positive is None else positive
                matches = np.flatnonzero(result.classes == positive)
                if len(matches) != 1:
                    raise ValueError(f"positive_class must be one of {result.classes.tolist()}")
                values = p[:, matches]
            else:
                raise ValueError("Multiclass probability output needs one named column per class in result.classes order")
        else:
            raise ValueError("Classification submission kind must be label or probability")
    else:
        if kind != "value":
            raise ValueError("Regression submission kind must be value")
        values = p[:, None]
    if values.shape[1] != len(names):
        raise ValueError("Prediction column count disagrees with requested output")
    if ids and (test[ids].isna().any().any() or test.duplicated(ids).any()):
        raise ValueError("Submission identifiers must be nonmissing and unique")
    output = test[ids].reset_index(drop=True).copy()
    output[names] = values
    if s.get("template_path"):
        template = read_table(s["template_path"])
        if len(template) != len(output) or not set(ids + names) <= set(template.columns):
            raise ValueError("Template row count or columns disagree with configuration")
        if ids:
            if template[ids].isna().any().any() or template.duplicated(ids).any():
                raise ValueError("Template identifiers must be nonmissing and unique")
            aligned = template[ids].merge(output, on=ids, how="left", validate="one_to_one", sort=False, indicator=True)
            if not aligned["_merge"].eq("both").all():
                raise ValueError("Template and test identifiers differ")
            template[names] = aligned[names].to_numpy()
        else:
            template[names] = values  # Explicit positional contract when no identifiers exist.
        output = template
    if output[names].isna().any().any():
        raise ValueError("Submission has missing predictions")
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing submission: {path}")
        output.to_csv(path, index=False)
    return output
