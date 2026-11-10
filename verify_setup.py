"""Check local setup; --smoke additionally loads models and runs predictions.

This script never installs packages, downloads resources, or starts training.
"""

import argparse
import importlib.metadata as metadata
import importlib.util
import math
from numbers import Integral, Real
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
PROJECT_ROOT = Path(__file__).resolve().parent
LANGUAGES = ("en", "ar", "multi")
MODEL_FILES = (
    "tfidf.pkl", "svm.pkl", "nb.pkl", "tokenizer.pkl", "meta.pkl", "lstm.keras"
)
TRAINING_FILES = (
    "datasets/english/Youtube01-Psy.csv",
    "datasets/english/Youtube02-KatyPerry.csv",
    "datasets/english/Youtube03-LMFAO.csv",
    "datasets/english/Youtube04-Eminem.csv",
    "datasets/english/Youtube05-Shakira.csv",
    "datasets/english/YouTube Comments Dataset with Sentiment Toxicity and Spam Labels (45K Rows).csv",
    "datasets/arabic/Augmented_SpamHamTweets.xlsx",
    "datasets/arabic/Autotweet-v1.0-Labeled-Tweets-Text.txt",
)
SAMPLES = {
    "en": ("English", "Thank you for explaining this topic so clearly."),
    "ar": ("Arabic", "شكراً على هذا الشرح الواضح والمفيد، لقد تعلمت الكثير اليوم."),
}


class Checks:
    def __init__(self):
        self.failures = 0

    def report(self, passed, message):
        print(f"[{'OK' if passed else 'FAIL'}] {message}")
        if not passed:
            self.failures += 1


def check_requirements(filename, checks):
    path = PROJECT_ROOT / filename
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeError) as exc:
        checks.report(False, f"Cannot read {filename}: {exc}")
        return
    count = 0
    for number, raw_line in enumerate(lines, 1):
        line = raw_line.partition("#")[0].strip()
        if not line or line.startswith(("-c ", "--constraint ", "--constraint=")):
            continue
        # The base requirements are always checked before optional training pins.
        if line in ("-r requirements.txt", "--requirement requirements.txt"):
            continue
        match = re.fullmatch(r"([A-Za-z0-9][A-Za-z0-9_.-]*)==([^\s;*]+)", line)
        if not match:
            checks.report(False, f"{filename}:{number}: expected an exact package==version pin")
            continue
        name, expected = match.groups()
        count += 1
        try:
            actual = metadata.version(name)
        except metadata.PackageNotFoundError:
            checks.report(False, f"{name}: not installed; expected {expected}")
            continue
        except Exception as exc:
            checks.report(False, f"Cannot inspect {name}: {exc}")
            continue
        checks.report(actual == expected, f"{name}: installed {actual}; expected {expected}")
    if count == 0:
        checks.report(False, f"{filename}: no direct package pins found")


def check_nltk_resources(checks):
    try:
        metadata.version("nltk")
    except metadata.PackageNotFoundError:
        checks.report(False, "NLTK resources not checked: nltk is not installed")
        return
    try:
        import nltk
    except Exception as exc:
        checks.report(False, f"Cannot import NLTK to check its resources: {exc}")
        return
    missing_packages = []
    for resource, package in (
        ("tokenizers/punkt_tab/english/", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
    ):
        try:
            nltk.data.find(resource)
        except LookupError:
            checks.report(False, f"Missing NLTK resource: {resource}")
            missing_packages.append(package)
        except Exception as exc:
            checks.report(False, f"Cannot check NLTK resource {resource}: {exc}")
        else:
            checks.report(True, f"NLTK resource present: {resource}")
    if missing_packages:
        prefix = "& " if sys.platform == "win32" else ""
        command = f'{prefix}"{sys.executable}" -m nltk.downloader {" ".join(missing_packages)}'
        print(f"Install the missing resources once, then rerun this check:\n  {command}")


def check_files(relative_paths, label, checks):
    missing = []
    for relative_path in relative_paths:
        path = PROJECT_ROOT / relative_path
        try:
            if not path.is_file() or path.stat().st_size == 0:
                missing.append(relative_path)
        except OSError:
            missing.append(relative_path)
    if missing:
        checks.report(False, f"{label}: missing, empty, or inaccessible files:")
        for relative_path in missing:
            print(f"       {relative_path}")
    else:
        checks.report(True, f"{label}: {len(relative_paths)} nonempty files present")


def validate_prediction(result, model):
    if not isinstance(result, tuple) or len(result) != 3:
        raise ValueError("prediction must be a (label, numeric_label, probability) tuple")
    label, numeric_label, probability = result
    if (isinstance(numeric_label, bool) or not isinstance(numeric_label, Integral)
            or numeric_label not in (0, 1)):
        raise ValueError("numeric label must be integer 0 or 1")
    if label != ("bot" if numeric_label == 1 else "human"):
        raise ValueError("text label does not match the numeric label")
    if model == "lstm":
        if (isinstance(probability, bool) or not isinstance(probability, Real)
                or not math.isfinite(probability) or not 0 <= probability <= 1):
            raise ValueError("LSTM probability must be finite and between 0 and 1")
        if numeric_label != int(probability >= 0.5):
            raise ValueError("LSTM label does not match its 0.5 probability threshold")


def run_smoke(checks):
    print("\nInference smoke test, not accuracy evaluation.")
    try:
        spec = importlib.util.spec_from_file_location("detector_verification", PROJECT_ROOT / "main.py")
        if spec is None or spec.loader is None:
            raise ImportError("Could not locate main.py")
        detector = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(detector)
    except Exception as exc:
        checks.report(False, f"Cannot import main.py: {type(exc).__name__}: {exc}")
        return
    passed_predictions = 0
    for language in LANGUAGES:
        try:
            pipeline = detector.load_pipeline(language=language)
        except Exception as exc:
            checks.report(False, f"{language} pipeline load: {type(exc).__name__}: {exc}")
            continue
        checks.report(True, f"{language} pipeline loaded")
        samples = (SAMPLES["en"], SAMPLES["ar"]) if language == "multi" else (SAMPLES[language],)
        for sample_name, comment in samples:
            for model in ("svm", "nb", "lstm"):
                description = f"{language}/{model}, {sample_name} sample"
                try:
                    result = detector.predict_comment(comment, pipeline, model=model)
                    validate_prediction(result, model)
                except Exception as exc:
                    checks.report(False, f"{description}: {type(exc).__name__}: {exc}")
                else:
                    passed_predictions += 1
                    checks.report(True, f"{description}: valid prediction ({result[0]})")
    checks.report(passed_predictions == 12, f"Inference smoke test: {passed_predictions}/12 predictions passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", action="store_true", help="also check training package pins and local dataset files")
    parser.add_argument("--smoke", action="store_true", help="after preflight passes, load all pipelines and run 12 predictions")
    args = parser.parse_args()
    checks = Checks()
    actual_python = ".".join(map(str, sys.version_info[:3]))
    checks.report(sys.version_info[:2] == (3, 11), f"Python {actual_python}; Python 3.11 required (reference: 3.11.9)")
    check_requirements("requirements.txt", checks)
    check_nltk_resources(checks)
    for language in LANGUAGES:
        paths = tuple(f"saved_models/{language}/{name}" for name in MODEL_FILES)
        check_files(paths, f"{language} saved pipeline", checks)
    if args.training:
        check_requirements("requirements-training.txt", checks)
        check_files(TRAINING_FILES, "Local training inputs", checks)
        print("Training checks inspect file presence only; dataset contents and remote availability are not tested.")
    if checks.failures:
        print(f"\n[FAIL] Setup checks found {checks.failures} problem(s). No inference was run.")
        if args.smoke:
            print("[FAIL] Smoke test skipped because preflight did not pass.")
        return 1
    if args.smoke:
        run_smoke(checks)
        if checks.failures:
            print("\n[FAIL] Inference smoke test failed. No training was run.")
            return 1
        print("\n[OK] Inference smoke test passed; this is not an accuracy evaluation.")
    else:
        print("\n[OK] Setup checks passed. Model loading and inference were not tested; use --smoke to check them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
