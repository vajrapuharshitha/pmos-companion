"""PMOS Screening Companion (starter).
Screening and awareness tool, NOT a diagnosis. Edit the TODO parts to connect your trained model."""
import os, pickle, datetime
import streamlit as st

st.set_page_config(page_title="PMOS Screening Companion", page_icon="🧬", layout="wide")
st.title("🧬 PMOS Screening Companion")
st.caption("PMOS = polyendocrine metabolic ovarian syndrome (formerly PCOS). "
           "This tool supports awareness and a better doctor visit. It does not diagnose.")

# ---------------- Inputs ----------------
with st.sidebar:
    st.header("Your information")
    adolescent = st.checkbox("Under 18 (adolescent criteria)")
    cycle_len = st.number_input("Usual cycle length (days)", 10, 120, 30)
    cycles_yr = st.number_input("Periods in the last 12 months", 0, 20, 10)
    hirs = st.slider("Hirsutism score (modified Ferriman-Gallwey, 0-36)", 0, 36, 0)
    acne = st.selectbox("Acne", ["None", "Mild", "Moderate", "Severe"])
    thin = st.checkbox("Scalp hair thinning")
    testo = st.selectbox("Blood androgen test (free testosterone / FAI)", ["Not tested", "Normal", "Elevated"])
    follicles = st.number_input("Follicles per ovary on ultrasound (0 = not done)", 0, 60, 0)
    amh = st.number_input("AMH (ng/mL, 0 = not done)", 0.0, 50.0, 0.0)
    glucose = st.number_input("Fasting glucose (mg/dL, 0 = not done)", 0, 300, 0)
    insulin = st.number_input("Fasting insulin (µU/mL, 0 = not done)", 0.0, 300.0, 0.0)
    bmi = st.number_input("BMI (optional, not used for criteria)", 0.0, 80.0, 0.0)

# ---------------- Rotterdam-style criteria (2 of 3) ----------------
ovul = cycle_len < 21 or cycle_len > 35 or cycles_yr < 8
clin_hyper = hirs >= 6 or (hirs >= 4 and (acne in ("Moderate", "Severe") or thin))
bio_hyper = testo == "Elevated"
hyper = clin_hyper or bio_hyper
pcom = (follicles >= 20) and not adolescent   # ultrasound not used in adolescents
met = {"Irregular / absent ovulation": ovul, "Androgen excess (clinical or lab)": hyper,
       "Polycystic ovarian morphology": pcom}
n_met = sum(met.values())
needed = 2

# ---------------- Multi-system view ----------------
homa = round(glucose * insulin / 405, 2) if glucose and insulin else None
c1, c2, c3 = st.columns(3)
c1.metric("Hormonal signals", "Present" if hyper else ("Not tested" if testo == "Not tested" and hirs < 4 else "Not seen"))
c2.metric("Metabolic (HOMA-IR)", homa if homa is not None else "Not tested",
          help="HOMA-IR = glucose x insulin / 405. Cut-offs vary by lab and population. Lean people can be insulin resistant too.")
c3.metric("Ovarian signals", "Present" if (ovul or pcom) else "Not seen")

# ---------------- Criteria checker ----------------
st.subheader("Diagnostic criteria checker")
for k, v in met.items():
    st.write(("✅ " if v else "⬜ ") + k)
if adolescent:
    st.info("For adolescents, irregular cycles and androgen excess are the main criteria; ultrasound is not used.")
if n_met >= needed:
    st.success(f"{n_met} of 3 features present. This pattern fits the PMOS criteria. Please discuss with a clinician, who must also rule out other causes.")
else:
    st.warning(f"{n_met} of 3 features present on the data entered.")
missing = []
if testo == "Not tested": missing.append("blood androgen test")
if follicles == 0 and amh == 0 and not adolescent: missing.append("ultrasound or AMH")
if homa is None: missing.append("fasting glucose and insulin")
if missing:
    st.write("**Worth asking your doctor about:** " + ", ".join(missing) + ".")

# ---------------- Your ML model (TODO) ----------------
st.subheader("ML risk estimate")
inputs = dict(cycle_len=cycle_len, cycles_yr=cycles_yr, hirs=hirs, bmi=bmi, follicles=follicles,
              amh=amh, glucose=glucose, insulin=insulin, homa=homa or 0)
if os.path.exists("model.pkl"):
    try:
        bundle = pickle.load(open("model.pkl", "rb"))     # expects {"model": clf, "features": [names]}
        X = [[inputs.get(f, 0) for f in bundle["features"]]]
        p = bundle["model"].predict_proba(X)[0][1]
        st.metric("Model probability", f"{p:.0%}")
        st.caption("Rename keys in `inputs` to match your model's feature names. TODO: add SHAP explanation here.")
    except Exception as e:
        st.error(f"Model could not run: {e}")
else:
    st.info("Add your trained `model.pkl` next to this file to enable the ML estimate.")

# ---------------- Doctor summary ----------------
summary = f"""PMOS screening summary ({datetime.date.today()})
Criteria met: {n_met}/3
""" + "\n".join(f"- {k}: {'yes' if v else 'no'}" for k, v in met.items()) + f"""
Cycle length: {cycle_len} days | periods last 12 months: {cycles_yr}
Hirsutism score: {hirs} | acne: {acne} | hair thinning: {thin}
Androgen test: {testo} | follicles/ovary: {follicles or 'n/a'} | AMH: {amh or 'n/a'}
HOMA-IR: {homa if homa is not None else 'n/a'}
Suggested discussion: {', '.join(missing) if missing else 'none missing'}
Screening aid only. Not a diagnosis.
"""
st.download_button("Download summary for your doctor", summary, "pmos_summary.txt")
st.caption("Screening and awareness only. Always confirm with a qualified clinician.")
