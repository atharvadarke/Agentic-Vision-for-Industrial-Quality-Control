"""
Comprehensive Inference Benchmark & Accuracy Evaluation
Agentic Vision for Industrial Quality Control

Tests:
  A. Feature Extraction Quality     (VisionEncoder)
  B. Anomaly Detection Accuracy     (AnomalyDetector - confusion matrix)
  C. Inference Speed Benchmarks     (latency per component)
  D. LLM Agent Analysis             (Groq / LLaMA-3.3 on a real anomaly)
  E. End-to-End Pipeline Timing     (image -> verdict)
"""

import os
import sys
import time
import json
import textwrap
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# -- path setup ---------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# -- helpers ------------------------------------------------------------------
SEP  = "=" * 70
SEP2 = "-" * 70

def header(title):
    print(f"\n{SEP}\n  {title}\n{SEP}")

def subheader(title):
    print(f"\n{SEP2}\n  {title}\n{SEP2}")

def ok(msg):   print(f"  [PASS] {msg}")
def fail(msg): print(f"  [FAIL] {msg}")
def info(msg): print(f"  [INFO] {msg}")

# -----------------------------------------------------------------------------
# SAMPLE DATA GENERATION
# -----------------------------------------------------------------------------

SAMPLE_DIR = os.path.join(BASE_DIR, "data", "raw")
os.makedirs(SAMPLE_DIR, exist_ok=True)

def make_normal_part(path, color=(180, 180, 180)):
    """Uniform grey industrial part -- ideal baseline."""
    img = Image.new("RGB", (300, 300), color=color)
    img.save(path)

def make_scratched_part(path):
    """Normal part with diagonal scratch lines -- surface defect."""
    img = Image.new("RGB", (300, 300), color=(180, 180, 180))
    draw = ImageDraw.Draw(img)
    for i in range(5):
        x0 = np.random.randint(10, 280)
        y0 = np.random.randint(10, 280)
        x1 = x0 + np.random.randint(-60, 60)
        y1 = y0 + np.random.randint(-60, 60)
        draw.line([(x0, y0), (x1, y1)], fill=(30, 30, 30), width=2)
    img.save(path)

def make_stained_part(path):
    """Part with a dark blotch -- contamination defect."""
    img = Image.new("RGB", (300, 300), color=(180, 180, 180))
    draw = ImageDraw.Draw(img)
    draw.ellipse([100, 100, 200, 200], fill=(50, 30, 20))
    img.save(path)

def make_corroded_part(path):
    """Random brownish noise across the surface -- corrosion defect."""
    arr = np.random.randint(150, 200, (300, 300, 3), dtype=np.uint8)
    for _ in range(20):
        r = np.random.randint(10, 40)
        cx, cy = np.random.randint(20, 280, 2)
        arr[max(0,cy-r):cy+r, max(0,cx-r):cx+r, 0] = np.random.randint(180, 220)
        arr[max(0,cy-r):cy+r, max(0,cx-r):cx+r, 1] = np.random.randint(80, 120)
        arr[max(0,cy-r):cy+r, max(0,cx-r):cx+r, 2] = np.random.randint(20, 60)
    Image.fromarray(arr).save(path)

def make_missing_component(path):
    """Part with a conspicuous black hole -- missing component defect."""
    img = Image.new("RGB", (300, 300), color=(180, 180, 180))
    draw = ImageDraw.Draw(img)
    draw.rectangle([120, 120, 180, 180], fill=(0, 0, 0))
    img.save(path)

# -----------------------------------------------------------------------------
# SECTION A -- Feature Extraction Quality
# -----------------------------------------------------------------------------

def section_a_feature_extraction(encoder):
    header("A. FEATURE EXTRACTION QUALITY  (VisionEncoder / ResNet-18)")

    images = {
        "normal_grey":  (os.path.join(SAMPLE_DIR, "bench_normal.jpg"),    "Normal"),
        "normal_light": (os.path.join(SAMPLE_DIR, "bench_light.jpg"),     "Normal"),
        "scratched":    (os.path.join(SAMPLE_DIR, "bench_scratch.jpg"),   "Anomaly"),
        "stained":      (os.path.join(SAMPLE_DIR, "bench_stain.jpg"),     "Anomaly"),
        "corroded":     (os.path.join(SAMPLE_DIR, "bench_corrosion.jpg"), "Anomaly"),
        "missing_comp": (os.path.join(SAMPLE_DIR, "bench_missing.jpg"),   "Anomaly"),
    }

    make_normal_part(images["normal_grey"][0],  (180, 180, 180))
    make_normal_part(images["normal_light"][0], (210, 210, 210))
    make_scratched_part(images["scratched"][0])
    make_stained_part(images["stained"][0])
    make_corroded_part(images["corroded"][0])
    make_missing_component(images["missing_comp"][0])

    latencies = []
    feature_vectors = {}

    print()
    for name, (path, label) in images.items():
        t0 = time.perf_counter()
        feat = encoder.extract_features(path)
        elapsed = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed)
        norm = np.linalg.norm(feat)
        feature_vectors[name] = (feat, label)
        info(f"{name:<20} | shape={feat.shape} | L2-norm={norm:.5f} | latency={elapsed:.1f} ms")

    subheader("Pairwise Cosine Similarity (Normal vs Anomaly)")
    normals   = [v for v, lbl in feature_vectors.values() if lbl == "Normal"]
    anomalies = [v for v, lbl in feature_vectors.values() if lbl == "Anomaly"]

    avg_normal_sim = np.dot(normals[0], normals[1])
    cross_sims     = [np.dot(normals[0], a) for a in anomalies]
    avg_cross_sim  = np.mean(cross_sims)

    info(f"Normal-to-Normal cosine similarity  : {avg_normal_sim:.4f}  (expect close to 1.0)")
    info(f"Normal-to-Anomaly cosine similarity : {avg_cross_sim:.4f}  (expect < Normal-to-Normal)")

    if avg_normal_sim > avg_cross_sim:
        ok("Normal pairs MORE similar than Normal-Anomaly pairs -- encoder is discriminative!")
    else:
        fail("Anomaly embeddings not separated from normal -- encoder may need fine-tuning.")

    subheader("Encoding Latency Summary")
    info(f"Min  : {min(latencies):.1f} ms")
    info(f"Max  : {max(latencies):.1f} ms")
    info(f"Mean : {np.mean(latencies):.1f} ms")
    info(f"Std  : {np.std(latencies):.1f} ms")

    return feature_vectors, latencies


# -----------------------------------------------------------------------------
# SECTION B -- Anomaly Detection Accuracy (Confusion Matrix)
# -----------------------------------------------------------------------------

def section_b_anomaly_detection(encoder, detector):
    header("B. ANOMALY DETECTION ACCURACY  (IsolationForest -- Confusion Matrix)")

    subheader("Training IsolationForest on 200 normal embeddings")
    np.random.seed(42)

    normal_img_path = os.path.join(SAMPLE_DIR, "bench_normal.jpg")
    normal_anchor   = encoder.extract_features(normal_img_path)

    normal_train = np.vstack([
        normal_anchor + np.random.normal(0, 0.02, 512)
        for _ in range(200)
    ])
    norms = np.linalg.norm(normal_train, axis=1, keepdims=True)
    normal_train = normal_train / norms

    t0 = time.perf_counter()
    detector.train(normal_train)
    train_time = (time.perf_counter() - t0) * 1000
    info(f"Training time : {train_time:.1f} ms  (200 samples x 512-D)")
    ok("IsolationForest trained successfully.")

    subheader("Building test set (normals + labelled anomalies)")

    test_cases = []

    # 10 jittered normal samples
    for i in range(10):
        v = normal_anchor + np.random.normal(0, 0.015, 512)
        v = v / np.linalg.norm(v)
        test_cases.append((v, "Normal", f"normal_jitter_{i}"))

    # Real images from data/raw (assumed normal since they came from the app)
    for fname in sorted(os.listdir(SAMPLE_DIR)):
        if fname.startswith("part_") and fname.endswith(".jpg"):
            fpath = os.path.join(SAMPLE_DIR, fname)
            feat  = encoder.extract_features(fpath)
            test_cases.append((feat, "Normal", f"real_{fname}"))

    # Defect images
    defect_images = {
        "scratch":      os.path.join(SAMPLE_DIR, "bench_scratch.jpg"),
        "stain":        os.path.join(SAMPLE_DIR, "bench_stain.jpg"),
        "corrosion":    os.path.join(SAMPLE_DIR, "bench_corrosion.jpg"),
        "missing_comp": os.path.join(SAMPLE_DIR, "bench_missing.jpg"),
    }
    for label, path in defect_images.items():
        feat = encoder.extract_features(path)
        test_cases.append((feat, "Anomaly", label))

    # Extreme synthetic outliers
    for scale in [5.0, 10.0, 20.0, 50.0]:
        v = np.ones(512) * scale
        test_cases.append((v, "Anomaly", f"extreme_outlier_{scale}x"))

    subheader("Running inference on all test cases")
    print(f"\n  {'Case':<28} {'True Label':<12} {'Predicted':<12} {'Score':>8}  Correct?")
    print(f"  {'-'*28} {'-'*12} {'-'*12} {'-'*8}  --------")

    results    = []
    infer_times = []

    for feat, true_label, case_name in test_cases:
        t0 = time.perf_counter()
        pred = detector.predict(feat)
        infer_ms = (time.perf_counter() - t0) * 1000
        infer_times.append(infer_ms)

        predicted = pred["status"]
        score     = pred["anomaly_score"]
        correct   = (predicted == true_label)
        results.append((true_label, predicted, correct, score))

        tick = "YES" if correct else "NO "
        print(f"  {case_name[:27]:<28} {true_label:<12} {predicted:<12} {score:>8.4f}  {tick}")

    subheader("Confusion Matrix & Metrics")

    TP = sum(1 for t, p, _, _ in results if t == "Anomaly" and p == "Anomaly")
    TN = sum(1 for t, p, _, _ in results if t == "Normal"  and p == "Normal")
    FP = sum(1 for t, p, _, _ in results if t == "Normal"  and p == "Anomaly")
    FN = sum(1 for t, p, _, _ in results if t == "Anomaly" and p == "Normal")
    total = len(results)

    accuracy    = (TP + TN) / total * 100
    precision   = TP / (TP + FP) * 100 if (TP + FP) > 0 else 0.0
    recall      = TP / (TP + FN) * 100 if (TP + FN) > 0 else 0.0
    f1          = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    specificity = TN / (TN + FP) * 100 if (TN + FP) > 0 else 0.0

    print(f"""
  Confusion Matrix:
  +---------------+---------------+---------------+
  |               | Pred: Normal  | Pred: Anomaly |
  +---------------+---------------+---------------+
  | True: Normal  |  TN = {TN:<8} |  FP = {FP:<8} |
  | True: Anomaly |  FN = {FN:<8} |  TP = {TP:<8} |
  +---------------+---------------+---------------+

  Total test cases : {total}
  Accuracy         : {accuracy:.1f}%
  Precision        : {precision:.1f}%
  Recall           : {recall:.1f}%
  F1-Score         : {f1:.1f}%
  Specificity      : {specificity:.1f}%
""")

    if accuracy >= 80:
        ok(f"Overall accuracy {accuracy:.1f}% -- model is performing well.")
    elif accuracy >= 60:
        info(f"Overall accuracy {accuracy:.1f}% -- acceptable, room for improvement.")
    else:
        fail(f"Overall accuracy {accuracy:.1f}% -- model needs more/better training data.")

    subheader("Inference Latency")
    info(f"Mean : {np.mean(infer_times):.3f} ms / prediction")
    info(f"Min  : {min(infer_times):.3f} ms")
    info(f"Max  : {max(infer_times):.3f} ms")
    info(f"P95  : {np.percentile(infer_times, 95):.3f} ms")

    return {
        "accuracy": accuracy, "precision": precision,
        "recall": recall, "f1": f1, "specificity": specificity,
        "TP": TP, "TN": TN, "FP": FP, "FN": FN,
        "infer_latency_mean_ms": np.mean(infer_times),
    }, test_cases


# -----------------------------------------------------------------------------
# SECTION C -- Anomaly Score Distribution Analysis
# -----------------------------------------------------------------------------

def section_c_score_analysis(detector, test_cases):
    header("C. ANOMALY SCORE DISTRIBUTION ANALYSIS")

    normal_scores  = []
    anomaly_scores = []

    for feat, true_label, _ in test_cases:
        pred = detector.predict(feat)
        s = pred["anomaly_score"]
        if true_label == "Normal":
            normal_scores.append(s)
        else:
            anomaly_scores.append(s)

    if normal_scores:
        info(f"Normal  scores -- mean: {np.mean(normal_scores):.4f}  std: {np.std(normal_scores):.4f}  "
             f"min: {min(normal_scores):.4f}  max: {max(normal_scores):.4f}")
    if anomaly_scores:
        info(f"Anomaly scores -- mean: {np.mean(anomaly_scores):.4f}  std: {np.std(anomaly_scores):.4f}  "
             f"min: {min(anomaly_scores):.4f}  max: {max(anomaly_scores):.4f}")

    sep_gap = (np.mean(normal_scores) - np.mean(anomaly_scores)) if (normal_scores and anomaly_scores) else 0
    info(f"Score gap (Normal mean - Anomaly mean): {sep_gap:.4f}  (higher = better separation)")

    if sep_gap > 0.05:
        ok("Clear score separation -- detector is well-calibrated.")
    elif sep_gap > 0:
        info("Modest separation -- consider more diverse training data.")
    else:
        fail("Negative gap -- check training data quality.")


# -----------------------------------------------------------------------------
# SECTION D -- LLM Agent Analysis on a Real Anomaly
# -----------------------------------------------------------------------------

def section_d_llm_agent():
    header("D. LLM AGENT ANALYSIS  (Groq / LLaMA-3.3-70B on detected anomaly)")

    try:
        from src.agent import run_agent

        anomaly_path  = os.path.join(SAMPLE_DIR, "bench_corrosion.jpg")
        anomaly_score = -0.35

        info(f"Sending anomaly to LLM agent ...")
        info(f"  Image : {os.path.basename(anomaly_path)}")
        info(f"  Score : {anomaly_score:.4f}")
        print()

        t0 = time.perf_counter()
        report = run_agent(anomaly_path, anomaly_score)
        elapsed = (time.perf_counter() - t0) * 1000

        print("  LLM Agent Report:")
        print("  " + SEP2)
        for k, v in report.items():
            if k.startswith("_"):
                continue
            wrapped = textwrap.fill(str(v), width=58, subsequent_indent="                             ")
            print(f"    {k:<26} : {wrapped}")
        print("  " + SEP2)
        info(f"LLM response time: {elapsed:.0f} ms")

        if "_error" in report:
            fail(f"LLM parsing/API error: {report['_error']}")
        else:
            ok("LLM agent returned a valid structured JSON report.")
            ok(f"Defect confirmed  : {report.get('defect_confirmed')}")
            ok(f"Severity score    : {report.get('severity_score')}")

        return elapsed

    except Exception as e:
        fail(f"LLM Agent test failed: {e}")
        import traceback; traceback.print_exc()
        return None


# -----------------------------------------------------------------------------
# SECTION E -- End-to-End Pipeline Timing
# -----------------------------------------------------------------------------

def section_e_e2e_timing(encoder, detector):
    header("E. END-TO-END PIPELINE TIMING  (image -> verdict, no LLM)")

    images = [
        os.path.join(SAMPLE_DIR, "bench_normal.jpg"),
        os.path.join(SAMPLE_DIR, "bench_scratch.jpg"),
        os.path.join(SAMPLE_DIR, "bench_corrosion.jpg"),
        os.path.join(SAMPLE_DIR, "bench_missing.jpg"),
    ]

    print(f"\n  {'Image':<22} {'Verdict':<12} {'E2E (ms)':>10}  {'Encode (ms)':>12}  {'Detect (ms)':>12}")
    print(f"  {'-'*22} {'-'*12} {'-'*10}  {'-'*12}  {'-'*12}")

    e2e_times = []

    for img_path in images:
        t_start = time.perf_counter()

        t0   = time.perf_counter()
        feat = encoder.extract_features(img_path)
        encode_ms = (time.perf_counter() - t0) * 1000

        t0   = time.perf_counter()
        pred = detector.predict(feat)
        detect_ms = (time.perf_counter() - t0) * 1000

        e2e_ms = (time.perf_counter() - t_start) * 1000
        e2e_times.append(e2e_ms)

        name = os.path.basename(img_path)[:21]
        print(f"  {name:<22} {pred['status']:<12} {e2e_ms:>10.1f}  {encode_ms:>12.1f}  {detect_ms:>12.1f}")

    print()
    info(f"Mean E2E latency : {np.mean(e2e_times):.1f} ms")
    info(f"Max  E2E latency : {max(e2e_times):.1f} ms")
    throughput = 1000 / np.mean(e2e_times)
    info(f"Throughput       : {throughput:.1f} images/sec  (CPU-only, encode + detect)")

    if np.mean(e2e_times) < 500:
        ok(f"E2E latency {np.mean(e2e_times):.1f} ms -- suitable for real-time QC.")
    else:
        info(f"E2E latency {np.mean(e2e_times):.1f} ms -- suitable for batch QC.")


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def main():
    print(f"\n{'#'*70}")
    print(f"#  AGENTIC VISION -- INFERENCE BENCHMARK & ACCURACY EVALUATION")
    print(f"#  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*70}")

    header("0. MODULE LOADING")
    try:
        from src.encoder  import VisionEncoder
        from src.detector import AnomalyDetector
        ok("All modules imported successfully.")
    except Exception as e:
        fail(f"Import failed: {e}")
        sys.exit(1)

    info("Initializing VisionEncoder (ResNet-18, CPU) ...")
    t0 = time.perf_counter()
    encoder = VisionEncoder()
    load_ms = (time.perf_counter() - t0) * 1000
    ok(f"VisionEncoder ready in {load_ms:.0f} ms")

    detector = AnomalyDetector()
    ok("AnomalyDetector initialized (untrained).")

    feature_vectors, enc_latencies = section_a_feature_extraction(encoder)
    metrics, test_cases            = section_b_anomaly_detection(encoder, detector)
    section_c_score_analysis(detector, test_cases)
    llm_ms = section_d_llm_agent()
    section_e_e2e_timing(encoder, detector)

    # Final summary
    header("FINAL BENCHMARK SUMMARY")
    print(f"""
  Component Performance
  ------------------------------------------------------------------
  VisionEncoder (ResNet-18 backbone, CPU)
    Model load time    : {load_ms:.0f} ms
    Mean encode latency: {np.mean(enc_latencies):.1f} ms / image
    Output dimension   : 512 (L2-normalized embedding)

  AnomalyDetector (IsolationForest, n_estimators=100)
    Training samples   : 200 normal embeddings x 512-D
    Mean infer latency : {metrics['infer_latency_mean_ms']:.3f} ms / sample

  Accuracy Metrics (labelled test set)
  ------------------------------------------------------------------
    Accuracy           : {metrics['accuracy']:.1f}%
    Precision          : {metrics['precision']:.1f}%
    Recall             : {metrics['recall']:.1f}%
    F1-Score           : {metrics['f1']:.1f}%
    Specificity        : {metrics['specificity']:.1f}%
    TP={metrics['TP']}  TN={metrics['TN']}  FP={metrics['FP']}  FN={metrics['FN']}

  LLM Agent (Groq / LLaMA-3.3-70B)
    Response latency   : {f"{llm_ms:.0f} ms" if llm_ms else "N/A (check GROQ_API_KEY)"}
    Activation mode    : Only triggered on Anomaly detections
  ------------------------------------------------------------------
""")

    if metrics["accuracy"] >= 80 and metrics["recall"] >= 80:
        print("  OVERALL STATUS: PIPELINE PERFORMING WELL")
    elif metrics["recall"] < 60:
        print("  OVERALL STATUS: LOW RECALL -- defects are being MISSED. Improve training data.")
    else:
        print("  OVERALL STATUS: PARTIAL PASS -- review FP/FN above for tuning opportunities.")

    print(f"\n{'#'*70}\n")


if __name__ == "__main__":
    main()
