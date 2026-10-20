"""Model A, part 1: intent classifier.

TF-IDF (1-2 grams, word + char) -> calibrated Linear SVM.
Five intents: coding / writing / search / reasoning / multimodal.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from . import config

MODEL_PATH = config.MODELS_DIR / "intent_clf.joblib"
METRICS_PATH = config.RESULTS_DIR / "intent_metrics.json"


@dataclass
class IntentResult:
    intent: str
    confidence: float
    probs: dict[str, float]


def build_pipeline() -> Pipeline:
    features = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2)),
    ])
    svm = LinearSVC(C=1.0)
    clf = CalibratedClassifierCV(svm, cv=5, method="sigmoid")
    return Pipeline([("features", features), ("clf", clf)])


PUBLIC_BENCH = config.DATA_DIR / "eval" / "public_bench.json"


def public_calibration_prompts(path=PUBLIC_BENCH) -> pd.DataFrame:
    """Prompts from the public benchmark's CALIBRATION split, labelled with their task type.

    Adding them teaches the classifier what real, third-party prompts look like (IFEval writing
    prompts full of counts and keywords, MBPP with assert lines).  The TEST split is never read here,
    so the public test accuracy stays an unseen-data measurement.
    """
    if not path.exists():
        return pd.DataFrame(columns=["text", "intent", "group", "split"])
    tasks = json.loads(path.read_text())["tasks"]
    rows = [{"text": t["prompt"], "intent": t["intent"], "group": f"public-{t['id']}", "split": "train"}
            for t in tasks if t["split"] == "calibration"]
    return pd.DataFrame(rows)


def train(csv_path=None, test_size: float = 0.2, seed: int = 42, public: bool = True) -> dict:
    csv_path = csv_path or config.DATA_DIR / "intents.csv"
    df = pd.read_csv(csv_path)
    n_team = len(df)
    if public:
        df = pd.concat([df, public_calibration_prompts()], ignore_index=True)
    df = df.drop_duplicates(subset="text").reset_index(drop=True)
    if "split" in df.columns:
        # group-aware split written by scripts/make_intent_dataset.py:
        # paraphrases of a held-out query are held out with it (no leakage)
        tr, te = df[df["split"] == "train"], df[df["split"] == "test"]
        X_tr, y_tr, X_te, y_te = tr["text"], tr["intent"], te["text"], te["intent"]
    else:
        X_tr, X_te, y_tr, y_te = train_test_split(
            df["text"], df["intent"], test_size=test_size, random_state=seed, stratify=df["intent"]
        )
    pipe = build_pipeline()
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    report = classification_report(y_te, pred, output_dict=True, zero_division=0)
    metrics = {
        "n_total": int(len(df)),
        "n_team_prompts": int(n_team),
        "n_public_calibration_prompts": int(len(df) - n_team) if public else 0,
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "accuracy": float(accuracy_score(y_te, pred)),
        "macro_f1": float(f1_score(y_te, pred, average="macro")),
        "per_class": {k: v for k, v in report.items() if k in config.INTENTS},
        "labels": config.INTENTS,
        "confusion_matrix": confusion_matrix(y_te, pred, labels=config.INTENTS).tolist(),
    }
    # for comparison only: a plain random split lets paraphrases of a test query
    # sit in train, which inflates accuracy.  Reported so the gap is visible.
    Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(
        df["text"], df["intent"], test_size=test_size, random_state=seed, stratify=df["intent"])
    metrics["random_split_accuracy_leaky"] = float(accuracy_score(yr_te, build_pipeline().fit(Xr_tr, yr_tr).predict(Xr_te)))
    # refit on everything for deployment
    pipe.fit(df["text"], df["intent"])
    joblib.dump(pipe, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    return metrics


class IntentClassifier:
    def __init__(self, path=MODEL_PATH):
        if not path.exists():
            raise FileNotFoundError(f"{path} missing. Run: python scripts/train_intent.py")
        self.pipe = joblib.load(path)
        self.classes = list(self.pipe.classes_)

    def predict(self, text: str) -> IntentResult:
        probs = self.pipe.predict_proba([text])[0]
        pmap = {c: float(p) for c, p in zip(self.classes, probs)}
        best = max(pmap, key=pmap.get)
        return IntentResult(intent=best, confidence=pmap[best], probs=pmap)
