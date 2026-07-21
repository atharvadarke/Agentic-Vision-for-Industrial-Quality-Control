"""
Full Benchmark with Realistic Test Images
"""
import os, sys, time, textwrap, numpy as np, warnings
from PIL import Image, ImageDraw, ImageFilter
warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if hasattr(sys.stdout, "reconfigure"):
    try: sys.stdout.reconfigure(encoding="utf-8")
    except: pass

RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)
SEP  = "=" * 72
SEP2 = "-" * 72
def hdr(t):  print(f"\n{SEP}\n  {t}\n{SEP}")
def sub(t):  print(f"\n{SEP2}\n  {t}\n{SEP2}")
def ok(m):   print(f"  [PASS] {m}")
def fail(m): print(f"  [FAIL] {m}")
def info(m): print(f"  [INFO] {m}")

def _base_metal(brightness=175, noise_std=4):
    arr = np.full((300,300,3), brightness, dtype=np.float32)
    arr += np.random.normal(0, noise_std, arr.shape)
    return Image.fromarray(np.clip(arr,0,255).astype(np.uint8))

def _brushed(img, axis=0):
    arr = np.array(img).astype(np.float32)
    for i in range(0,300,3):
        v = np.random.uniform(-3,3)
        if axis==0: arr[i,:,:] += v
        else:       arr[:,i,:] += v
    return Image.fromarray(np.clip(arr,0,255).astype(np.uint8))

def gen_normal_uniform(p):   _base_metal(178,3).save(p)
def gen_normal_brushed(p):   _brushed(_base_metal(170,2),0).save(p)
def gen_normal_cast(p):      _base_metal(140,6).filter(ImageFilter.SMOOTH).save(p)
def gen_normal_polished(p):  _brushed(_base_metal(210,2),1).save(p)
def gen_normal_anodized(p):
    arr = np.full((300,300,3),[155,165,190],dtype=np.float32)
    arr += np.random.normal(0,3,arr.shape)
    Image.fromarray(np.clip(arr,0,255).astype(np.uint8)).save(p)

def gen_scratch(p):
    img=_base_metal(175,3); d=ImageDraw.Draw(img)
    for _ in range(6):
        x0,y0=np.random.randint(10,290,2)
        a=np.random.uniform(0,np.pi); l=np.random.randint(40,120)
        d.line([(x0,y0),(int(x0+l*np.cos(a)),int(y0+l*np.sin(a)))],fill=(20,20,20),width=np.random.choice([1,1,2]))
    img.save(p)

def gen_stain(p):
    img=_base_metal(175,3); arr=np.array(img).astype(np.float32)
    cx,cy=np.random.randint(80,220,2)
    for _ in range(800):
        dx,dy=np.random.normal(0,25,2); px,py=int(cx+dx),int(cy+dy)
        if 0<=px<300 and 0<=py<300:
            f=np.random.uniform(0.3,0.6); arr[py,px,0]*=f; arr[py,px,1]*=f*0.7; arr[py,px,2]*=f*0.5
    Image.fromarray(np.clip(arr,0,255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(2)).save(p)

def gen_corrosion(p):
    img=_base_metal(170,5); arr=np.array(img).astype(np.float32)
    for _ in range(12):
        cx,cy=np.random.randint(20,280,2); r=np.random.randint(15,50)
        for _ in range(500):
            dx,dy=np.random.normal(0,r/2.5,2); px,py=int(cx+dx),int(cy+dy)
            if 0<=px<300 and 0<=py<300:
                arr[py,px,0]=np.random.uniform(160,210); arr[py,px,1]=np.random.uniform(60,110); arr[py,px,2]=np.random.uniform(10,50)
    Image.fromarray(np.clip(arr,0,255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1)).save(p)

def gen_missing_comp(p):
    img=_base_metal(175,3); d=ImageDraw.Draw(img)
    x0,y0=np.random.randint(80,150,2); w,h=np.random.randint(40,80),np.random.randint(40,80)
    d.rectangle([x0,y0,x0+w,y0+h],fill=(5,5,5)); img.save(p)

def gen_crack(p):
    img=_base_metal(175,3); d=ImageDraw.Draw(img)
    x,y=np.random.randint(60,240),np.random.randint(60,240)
    for _ in range(4):
        l=np.random.randint(30,80); a=np.random.uniform(0,np.pi)
        x2,y2=int(x+l*np.cos(a)),int(y+l*np.sin(a))
        d.line([(x,y),(x2,y2)],fill=(10,10,10),width=2)
        bx,by=(x+x2)//2,(y+y2)//2; ba=a+np.random.uniform(0.3,1.0)*np.random.choice([-1,1]); bl=np.random.randint(15,40)
        d.line([(bx,by),(int(bx+bl*np.cos(ba)),int(by+bl*np.sin(ba)))],fill=(10,10,10),width=1)
    img.save(p)

def gen_dent(p):
    img=_base_metal(175,3); arr=np.array(img).astype(np.float32)
    cx,cy=np.random.randint(80,220,2); r=np.random.randint(20,50)
    for py2 in range(max(0,cy-r),min(300,cy+r)):
        for px2 in range(max(0,cx-r),min(300,cx+r)):
            dist=np.sqrt((px2-cx)**2+(py2-cy)**2)
            if dist<r: arr[py2,px2]*=1.0-0.4*(1-dist/r)**2
    Image.fromarray(np.clip(arr,0,255).astype(np.uint8)).save(p)

def gen_dust(p):
    img=_base_metal(175,3); d=ImageDraw.Draw(img)
    for _ in range(np.random.randint(30,80)):
        x,y=np.random.randint(10,290,2); r=np.random.randint(1,5); s=np.random.randint(30,90)
        d.ellipse([x-r,y-r,x+r,y+r],fill=(s,s,s))
    img.save(p)

def gen_edge_chip(p):
    img=_base_metal(175,3); d=ImageDraw.Draw(img)
    edge=np.random.choice(["top","bottom","left","right"])
    if edge=="top":    pts=[(np.random.randint(50,150),0),(np.random.randint(150,250),0),(np.random.randint(100,200),np.random.randint(30,80))]
    elif edge=="bottom": pts=[(np.random.randint(50,150),299),(np.random.randint(150,250),299),(np.random.randint(100,200),np.random.randint(220,270))]
    elif edge=="left": pts=[(0,np.random.randint(50,150)),(0,np.random.randint(150,250)),(np.random.randint(30,80),np.random.randint(100,200))]
    else:              pts=[(299,np.random.randint(50,150)),(299,np.random.randint(150,250)),(np.random.randint(220,270),np.random.randint(100,200))]
    d.polygon(pts,fill=(200,200,200)); img.save(p)

TEST_IMAGES = {
    "test_normal_uniform.jpg":  (gen_normal_uniform, "Normal"),
    "test_normal_brushed.jpg":  (gen_normal_brushed, "Normal"),
    "test_normal_cast.jpg":     (gen_normal_cast,    "Normal"),
    "test_normal_polished.jpg": (gen_normal_polished,"Normal"),
    "test_normal_anodized.jpg": (gen_normal_anodized,"Normal"),
    "test_scratch.jpg":         (gen_scratch,        "Anomaly"),
    "test_stain.jpg":           (gen_stain,          "Anomaly"),
    "test_corrosion.jpg":       (gen_corrosion,      "Anomaly"),
    "test_missing_comp.jpg":    (gen_missing_comp,   "Anomaly"),
    "test_crack.jpg":           (gen_crack,          "Anomaly"),
    "test_dent.jpg":            (gen_dent,           "Anomaly"),
    "test_dust.jpg":            (gen_dust,           "Anomaly"),
    "test_edge_chip.jpg":       (gen_edge_chip,      "Anomaly"),
}
TRAIN_IMAGES = {
    "train_normal_uniform.jpg":  gen_normal_uniform,
    "train_normal_brushed.jpg":  gen_normal_brushed,
    "train_normal_cast.jpg":     gen_normal_cast,
    "train_normal_polished.jpg": gen_normal_polished,
    "train_normal_anodized.jpg": gen_normal_anodized,
}

def generate_images():
    np.random.seed(0)
    for fn,gf in TRAIN_IMAGES.items(): gf(os.path.join(RAW_DIR,fn))
    info(f"  {len(TRAIN_IMAGES)} training images ready.")
    for fn,(gf,_) in TEST_IMAGES.items(): gf(os.path.join(RAW_DIR,fn))
    info(f"  {len(TEST_IMAGES)} test images ready.")

def section_a(encoder):
    hdr("A. FEATURE EXTRACTION QUALITY  (VisionEncoder / ResNet-18)")
    lats=[]; normals=[]; anomalies=[]
    print(f"\n  {'Image':<32} {'Label':<10} {'L2-norm':>9}  {'Latency(ms)':>12}")
    print(f"  {'-'*32} {'-'*10} {'-'*9}  {'-'*12}")
    for fn,(_, lbl) in TEST_IMAGES.items():
        path=os.path.join(RAW_DIR,fn); t0=time.perf_counter()
        feat=encoder.extract_features(path); ms=(time.perf_counter()-t0)*1000
        lats.append(ms); norm=np.linalg.norm(feat)
        print(f"  {fn[:31]:<32} {lbl:<10} {norm:>9.5f}  {ms:>12.2f}")
        (normals if lbl=="Normal" else anomalies).append(feat)
    sub("Pairwise Cosine Similarity")
    nn=[float(np.dot(normals[i],normals[j])) for i in range(len(normals)) for j in range(i+1,len(normals))]
    na=[float(np.dot(n,a)) for n in normals for a in anomalies]
    avg_nn=np.mean(nn); avg_na=np.mean(na); gap=avg_nn-avg_na
    info(f"Normal-to-Normal  (avg): {avg_nn:.4f}")
    info(f"Normal-to-Anomaly (avg): {avg_na:.4f}")
    info(f"Separation gap         : {gap:.4f}")
    ok("Encoder IS discriminative.") if gap>0.01 else fail("Encoder separation LOW.")
    sub("Encoding Latency")
    info(f"Min={min(lats):.2f}ms  Mean={np.mean(lats):.2f}ms  P95={np.percentile(lats,95):.2f}ms  Max={max(lats):.2f}ms")
    return lats, normals, anomalies

def section_b(encoder, detector):
    hdr("B. ANOMALY DETECTION ACCURACY  (IsolationForest)")
    sub("Training — 5 normal images x 41 augmentations = 205 samples")
    X=[]; np.random.seed(42)
    for fn,gf in TRAIN_IMAGES.items():
        feat=encoder.extract_features(os.path.join(RAW_DIR,fn)); X.append(feat)
        for _ in range(40):
            j=feat+np.random.normal(0,0.01,512); X.append(j/(np.linalg.norm(j)+1e-9))
    X=np.vstack(X); info(f"Training matrix: {X.shape}")
    t0=time.perf_counter(); detector.train(X); tmx=(time.perf_counter()-t0)*1000
    ok(f"IsolationForest trained in {tmx:.1f} ms")
    sub("Inference on 13 labeled test images")
    print(f"\n  {'Image':<33} {'True':<10} {'Pred':<10} {'Score':>8}  Result")
    print(f"  {'-'*33} {'-'*10} {'-'*10} {'-'*8}  ------")
    results=[]; itimes=[]
    for fn,(_, true_lbl) in TEST_IMAGES.items():
        feat=encoder.extract_features(os.path.join(RAW_DIR,fn))
        t0=time.perf_counter(); pred=detector.predict(feat); ms=(time.perf_counter()-t0)*1000
        itimes.append(ms); predicted=pred["status"]; score=pred["anomaly_score"]
        correct=(predicted==true_lbl); results.append((fn,true_lbl,predicted,score,correct))
        print(f"  {fn[:32]:<33} {true_lbl:<10} {predicted:<10} {score:>8.4f}  {'PASS' if correct else 'FAIL'}")
    TP=sum(1 for _,t,p,_,_ in results if t=="Anomaly" and p=="Anomaly")
    TN=sum(1 for _,t,p,_,_ in results if t=="Normal"  and p=="Normal")
    FP=sum(1 for _,t,p,_,_ in results if t=="Normal"  and p=="Anomaly")
    FN=sum(1 for _,t,p,_,_ in results if t=="Anomaly" and p=="Normal")
    n=len(results)
    acc=( TP+TN)/n*100; prec=TP/(TP+FP)*100 if TP+FP>0 else 0
    rec=TP/(TP+FN)*100 if TP+FN>0 else 0; f1=2*prec*rec/(prec+rec) if prec+rec>0 else 0
    spec=TN/(TN+FP)*100 if TN+FP>0 else 0
    print(f"""
  Confusion Matrix:
  +-------------------+-----------------+-----------------+
  |                   | Pred: Normal    | Pred: Anomaly   |
  +-------------------+-----------------+-----------------+
  | True: Normal      | TN = {TN:<10}  | FP = {FP:<10}  |
  | True: Anomaly     | FN = {FN:<10}  | TP = {TP:<10}  |
  +-------------------+-----------------+-----------------+
  Total={n}  Accuracy={acc:.1f}%  Precision={prec:.1f}%
  Recall(TPR)={rec:.1f}%  F1={f1:.1f}%  Specificity={spec:.1f}%
""")
    if FN>0: fail(f"Missed {FN} defect(s) — FN: {[r[0] for r in results if r[1]=='Anomaly' and r[2]=='Normal']}")
    if FP>0: fail(f"False alarms on {FP} normal image(s) — FP: {[r[0] for r in results if r[1]=='Normal' and r[2]=='Anomaly']}")
    if FN==0 and FP==0: ok("Perfect classification on all 13 images!")
    elif acc>=80: ok(f"Accuracy {acc:.1f}% — good.")
    elif acc>=60: info(f"Accuracy {acc:.1f}% — acceptable.")
    else: fail(f"Accuracy {acc:.1f}% — needs retraining.")
    sub("Inference Latency (IsolationForest only)")
    info(f"Mean={np.mean(itimes):.4f}ms  P95={np.percentile(itimes,95):.4f}ms  Max={max(itimes):.4f}ms")
    return {"accuracy":acc,"precision":prec,"recall":rec,"f1":f1,"specificity":spec,"TP":TP,"TN":TN,"FP":FP,"FN":FN,"total":n,"infer_mean_ms":np.mean(itimes),"train_ms":tmx}, results

def section_c(results):
    hdr("C. ANOMALY SCORE DISTRIBUTION")
    ns=[s for _,t,_,s,_ in results if t=="Normal"]; as_=[s for _,t,_,s,_ in results if t=="Anomaly"]
    if ns:  info(f"Normal  — mean:{np.mean(ns):.4f}  std:{np.std(ns):.4f}  min:{min(ns):.4f}  max:{max(ns):.4f}")
    if as_: info(f"Anomaly — mean:{np.mean(as_):.4f}  std:{np.std(as_):.4f}  min:{min(as_):.4f}  max:{max(as_):.4f}")
    gap=np.mean(ns)-np.mean(as_) if ns and as_ else 0
    info(f"Score gap: {gap:.4f}  (higher=better separation)")
    if gap>0.05: ok("Clear separation — detector well-calibrated.")
    elif gap>0:  info("Modest separation.")
    else:        fail("Negative gap — scores overlap!")
    sub("All images ranked by anomaly score (ascending = most anomalous)")
    for fn,t,p,s,c in sorted(results,key=lambda x:x[3]):
        flag=" <-- MISCLASSIFIED" if not c else ""
        print(f"  {fn[:32]:<33} {t:<10} {s:>8.4f}{flag}")

def section_d():
    hdr("D. LLM AGENT  (Groq / LLaMA-3.3-70B)")
    try:
        from src.agent import run_agent
        path=os.path.join(RAW_DIR,"test_corrosion.jpg"); score=-0.35
        info(f"Sending test_corrosion.jpg to LLM agent (score={score}) ...")
        t0=time.perf_counter(); report=run_agent(path,score); elapsed=(time.perf_counter()-t0)*1000
        print("  "+SEP2)
        for k,v in report.items():
            if not k.startswith("_"): print(f"    {k:<28}: {textwrap.fill(str(v),60,subsequent_indent=' '*32)}")
        print("  "+SEP2)
        info(f"LLM response time: {elapsed:.0f} ms")
        if "_error" in report: fail(f"LLM error: {report['_error']}"); return None,"error"
        ok(f"Valid JSON report — defect_confirmed={report.get('defect_confirmed')}  severity={report.get('severity_score')}")
        return elapsed,"ok"
    except Exception as e:
        fail(f"LLM Agent failed: {e}"); import traceback; traceback.print_exc(); return None,str(e)

def section_e(encoder, detector):
    hdr("E. END-TO-END LATENCY  (image -> verdict, no LLM)")
    print(f"\n  {'Image':<33} {'Verdict':<10} {'E2E':>8}  {'Encode':>8}  {'Detect':>8}")
    print(f"  {'-'*33} {'-'*10} {'-'*8}  {'-'*8}  {'-'*8}")
    e2e=[]
    for fn in TEST_IMAGES:
        path=os.path.join(RAW_DIR,fn); ts=time.perf_counter()
        t0=time.perf_counter(); feat=encoder.extract_features(path); enc_ms=(time.perf_counter()-t0)*1000
        t0=time.perf_counter(); pred=detector.predict(feat); det_ms=(time.perf_counter()-t0)*1000
        total_ms=(time.perf_counter()-ts)*1000; e2e.append(total_ms)
        print(f"  {fn[:32]:<33} {pred['status']:<10} {total_ms:>7.1f}ms  {enc_ms:>7.1f}ms  {det_ms:>7.2f}ms")
    print()
    info(f"Mean={np.mean(e2e):.1f}ms  Median={np.median(e2e):.1f}ms  P95={np.percentile(e2e,95):.1f}ms  Max={max(e2e):.1f}ms")
    info(f"Throughput: {1000/np.mean(e2e):.1f} images/sec (CPU)")
    if np.mean(e2e)<500: ok(f"Mean {np.mean(e2e):.1f}ms < 500ms SLA — REAL-TIME capable.")
    else:                fail(f"Mean {np.mean(e2e):.1f}ms exceeds 500ms SLA.")
    return e2e

def main():
    print(f"\n{'#'*72}\n#  AGENTIC VISION — FULL BENCHMARK  {time.strftime('%Y-%m-%d %H:%M:%S')}\n#  {len(TEST_IMAGES)} test | {len(TRAIN_IMAGES)} train | 5 Normal | 8 Anomaly types\n{'#'*72}")
    hdr("0. MODULE & MODEL LOADING")
    try:
        from src.encoder  import VisionEncoder
        from src.detector import AnomalyDetector
        ok("Modules imported.")
    except Exception as e: fail(f"Import failed: {e}"); sys.exit(1)
    t0=time.perf_counter(); encoder=VisionEncoder(); load_ms=(time.perf_counter()-t0)*1000
    ok(f"VisionEncoder ready in {load_ms:.0f} ms")
    detector=AnomalyDetector(); ok("AnomalyDetector initialized.")
    hdr("GENERATING TEST IMAGES"); generate_images()
    enc_lats,normals,anomalies = section_a(encoder)
    metrics,results            = section_b(encoder,detector)
    section_c(results)
    llm_ms,llm_status          = section_d()
    e2e_times                  = section_e(encoder,detector)
    hdr("FINAL BENCHMARK SUMMARY")
    print(f"""
  Component             | Metric                | Value
  ----------------------+-----------------------+------------------
  VisionEncoder         | Model load            | {load_ms:.0f} ms
                        | Mean encode latency   | {np.mean(enc_lats):.2f} ms
                        | P95 encode latency    | {np.percentile(enc_lats,95):.2f} ms
                        | Output dimension      | 512-D L2-norm
  ----------------------+-----------------------+------------------
  IsolationForest       | Training time         | {metrics['train_ms']:.1f} ms
                        | Mean infer latency    | {metrics['infer_mean_ms']:.4f} ms
                        | contamination param   | 0.05 (5%)
  ----------------------+-----------------------+------------------
  Accuracy (13 images)  | Accuracy              | {metrics['accuracy']:.1f}%
                        | Precision             | {metrics['precision']:.1f}%
                        | Recall (TPR)          | {metrics['recall']:.1f}%
                        | F1-Score              | {metrics['f1']:.1f}%
                        | Specificity (TNR)     | {metrics['specificity']:.1f}%
                        | TP/TN/FP/FN           | {metrics['TP']}/{metrics['TN']}/{metrics['FP']}/{metrics['FN']}
  ----------------------+-----------------------+------------------
  End-to-End (no LLM)   | Mean E2E              | {np.mean(e2e_times):.1f} ms
                        | P95 E2E               | {np.percentile(e2e_times,95):.1f} ms
                        | Max E2E               | {max(e2e_times):.1f} ms
                        | Throughput            | {1000/np.mean(e2e_times):.1f} img/s
  ----------------------+-----------------------+------------------
  LLM Agent             | Response time         | {"N/A" if llm_ms is None else f"{llm_ms:.0f} ms"}
                        | Trigger policy        | Anomaly-only
""")
    issues=[]
    if metrics["recall"]<80:      issues.append(f"LOW RECALL {metrics['recall']:.1f}% — defects missed (FN={metrics['FN']})")
    if metrics["specificity"]<80: issues.append(f"LOW SPECIFICITY {metrics['specificity']:.1f}% — false alarms (FP={metrics['FP']})")
    if metrics["accuracy"]<60:    issues.append(f"LOW ACCURACY {metrics['accuracy']:.1f}%")
    if np.mean(e2e_times)>500:    issues.append(f"LATENCY BREACH — {np.mean(e2e_times):.1f}ms > 500ms SLA")
    if llm_ms is None or llm_status!="ok": issues.append(f"LLM AGENT ISSUE: {llm_status}")
    if not issues: print("  OVERALL: ALL CHECKS PASSED — PIPELINE PRODUCTION-READY")
    else:
        print(f"  OVERALL: {len(issues)} ISSUE(S) FOUND:")
        for i in issues: print(f"    - {i}")
    print(f"\n{'#'*72}\n")

if __name__=="__main__":
    main()
