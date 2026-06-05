import os
import json
import textwrap
from typing import Optional

import numpy as np
import shap
import xgboost as xgb
import google.generativeai as genai

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

XGBOOST_MODEL_PATH = os.getenv("XGBOOST_MODEL_PATH", "xgboost.json")
GEMINI_API_KEY     = os.getenv("GEMINI_API_KEY", "")          # required
GEMINI_MODEL       = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")

FEATURE_META = {
    "Age": {
        "display": "Usia",
        "unit": "tahun",
        "decode": None,
    },
    "Education": {
        "display": "Tingkat Pendidikan",
        "unit": None,
        "decode": {1: "SD", 2: "SMP", 3: "SMA/SMK", 4: "D3/S1", 5: "S2/S3"},
    },
    "Initial_Capital": {
        "display": "Modal Awal",
        "unit": None,
        "decode": {0: "Tidak memiliki modal awal", 1: "Memiliki modal awal"},
    },
    "Financial_Record_Keeping": {
        "display": "Pencatatan Keuangan",
        "unit": None,
        "decode": {0: "Tidak melakukan pencatatan", 1: "Melakukan pencatatan"},
    },
    "Internet_Usage": {
        "display": "Penggunaan Internet untuk Bisnis",
        "unit": None,
        "decode": {0: "Tidak menggunakan", 1: "Menggunakan internet"},
    },
    "Business_Plan": {
        "display": "Rencana Bisnis",
        "unit": None,
        "decode": {0: "Tidak memiliki rencana bisnis", 1: "Memiliki rencana bisnis"},
    },
    "Marketing_Effort": {
        "display": "Upaya Pemasaran",
        "unit": "skala 1–10",
        "decode": None,
    },
    "Partnership": {
        "display": "Kemitraan",
        "unit": None,
        "decode": {0: "Tidak memiliki mitra", 1: "Memiliki mitra bisnis"},
    },
    "Parent_Business_Experience": {
        "display": "Pengalaman Bisnis Orang Tua",
        "unit": None,
        "decode": {0: "Orang tua tidak berbisnis", 1: "Orang tua memiliki pengalaman bisnis"},
    },
    "Industry_Experience": {
        "display": "Pengalaman di Industri",
        "unit": "tahun",
        "decode": None,
    },
    "Owner_Gender": {
        "display": "Jenis Kelamin Pemilik",
        "unit": None,
        "decode": {0: "Perempuan", 1: "Laki-laki"},
    },
    "Professional_Advice": {
        "display": "Saran Profesional",
        "unit": "skala 1–7",
        "decode": None,
    },
}

FEATURE_ORDER = list(FEATURE_META.keys())

_model: xgb.XGBClassifier = None
_explainer: shap.TreeExplainer = None
_gemini_client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _explainer, _gemini_client

    # XGBoost
    _model = xgb.XGBClassifier()
    _model.load_model(XGBOOST_MODEL_PATH)

    # SHAP TreeExplainer — no background data needed for tree models
    _explainer = shap.TreeExplainer(_model)

    # Gemini
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY environment variable is not set.")
    genai.configure(api_key=GEMINI_API_KEY)
    _gemini_client = genai.GenerativeModel(GEMINI_MODEL)

    print("✓ XGBoost model loaded")
    print("✓ SHAP TreeExplainer ready")
    print(f"✓ Gemini model '{GEMINI_MODEL}' configured")
    yield


app = FastAPI(
    title="UMKM Success Prediction API",
    description="Prediksi keberhasilan UMKM dengan penjelasan SHAP dan analisis AI.",
    version="2.0.0",
    lifespan=lifespan,
)

class UMKMInput(BaseModel):
    Age:                        int = Field(..., ge=15, le=80,  example=28)
    Education:                  int = Field(..., ge=1,  le=5,   example=3,  description="1=SD, 2=SMP, 3=SMA, 4=D3/S1, 5=S2/S3")
    Initial_Capital:            int = Field(..., ge=0,  le=1,   example=1,  description="0=Tidak, 1=Ya")
    Financial_Record_Keeping:   int = Field(..., ge=0,  le=1,   example=1,  description="0=Tidak, 1=Ya")
    Internet_Usage:             int = Field(..., ge=0,  le=1,   example=1,  description="0=Tidak, 1=Ya")
    Business_Plan:              int = Field(..., ge=0,  le=1,   example=1,  description="0=Tidak, 1=Ya")
    Marketing_Effort:           int = Field(..., ge=1,  le=10,  example=7)
    Partnership:                int = Field(..., ge=0,  le=1,   example=0,  description="0=Tidak, 1=Ya")
    Parent_Business_Experience: int = Field(..., ge=0,  le=1,   example=1,  description="0=Tidak, 1=Ya")
    Industry_Experience:        int = Field(..., ge=0,  le=40,  example=3,  description="Tahun pengalaman")
    Owner_Gender:               int = Field(..., ge=0,  le=1,   example=1,  description="0=Perempuan, 1=Laki-laki")
    Professional_Advice:        int = Field(..., ge=1,  le=7,   example=5)


class ShapFactor(BaseModel):
    feature:      str
    display_name: str
    shap_value:   float   # raw SHAP value; positive = pushes toward success
    input_value:  float   # what the user entered
    decoded_value: Optional[str]  # human-readable if categorical
    direction:    str     # "positive" | "negative"
    impact_pct:   float   # abs(shap) / sum(abs(all shap)) * 100


class ShapAnalysis(BaseModel):
    base_value:             float   # model's average prediction (log-odds)
    top_positive_factors:   list[ShapFactor]   # top 3 pushing toward success
    top_negative_factors:   list[ShapFactor]   # top 3 pushing toward failure
    all_factors:            list[ShapFactor]   # all 12, sorted by |shap|


class AIAnalysis(BaseModel):
    summary:         str          # 2–3 sentence plain-language summary
    strengths:       list[str]    # strengths (positive SHAP)
    weaknesses:      list[str]    # weaknessse (negative SHAP)
    recommendations: list[str]    # concrete, actionable next steps (Gemini AI)
    encouragement:   str          # motivational closing for young entrepreneurs


class PredictionResponse(BaseModel):
    prediction:          int    
    label:               str
    probability_success: float
    probability_fail:    float
    confidence:          str    

    shap:                ShapAnalysis
    ai_analysis:         AIAnalysis


def _confidence_label(prob_success: float) -> str:
    if prob_success >= 0.75 or prob_success <= 0.25:
        return "Tinggi"
    if prob_success >= 0.60 or prob_success <= 0.40:
        return "Sedang"
    return "Rendah"


def _decode_value(feature: str, value: float) -> Optional[str]:
    decode = FEATURE_META[feature].get("decode")
    if decode:
        return decode.get(int(value), str(int(value)))
    unit = FEATURE_META[feature].get("unit")
    if unit:
        return f"{int(value)} {unit}"
    return None


def _build_shap_analysis(X: np.ndarray, shap_values: np.ndarray) -> ShapAnalysis:
    """Convert raw SHAP values into structured ShapAnalysis."""
    base_value  = float(_explainer.expected_value
                        if np.isscalar(_explainer.expected_value)
                        else _explainer.expected_value[1])

    sv = shap_values[0]              # shape (12,)
    total_abs   = float(np.abs(sv).sum()) or 1e-9

    factors = []
    for i, feat in enumerate(FEATURE_ORDER):
        raw_val   = float(X[0, i])
        shap_val  = float(sv[i])
        factors.append(ShapFactor(
            feature       = feat,
            display_name  = FEATURE_META[feat]["display"],
            shap_value    = round(shap_val, 5),
            input_value   = raw_val,
            decoded_value = _decode_value(feat, raw_val),
            direction     = "positive" if shap_val >= 0 else "negative",
            impact_pct    = round(abs(shap_val) / total_abs * 100, 2),
        ))

    factors_sorted = sorted(factors, key=lambda f: abs(f.shap_value), reverse=True)
    positives      = [f for f in factors_sorted if f.shap_value > 0][:3]
    negatives      = [f for f in factors_sorted if f.shap_value < 0][:3]

    return ShapAnalysis(
        base_value            = round(base_value, 5),
        top_positive_factors  = positives,
        top_negative_factors  = negatives,
        all_factors           = factors_sorted,
    )


def _build_gemini_prompt(
    data: UMKMInput,
    pred: int,
    prob_success: float,
    shap_analysis: ShapAnalysis,
) -> str:
    
    label = "BERHASIL" if pred == 1 else "TIDAK BERHASIL"
    pct   = round(prob_success * 100, 1)

    positives_text = "\n".join(
        f"  + {f.display_name}: {f.decoded_value or f.input_value} "
        f"(kontribusi positif {f.impact_pct:.1f}%)"
        for f in shap_analysis.top_positive_factors
    ) or "  (tidak ada faktor pendorong signifikan)"

    negatives_text = "\n".join(
        f"  - {f.display_name}: {f.decoded_value or f.input_value} "
        f"(menghambat sebesar {f.impact_pct:.1f}%)"
        for f in shap_analysis.top_negative_factors
    ) or "  (tidak ada faktor penghambat signifikan)"

    # Build full feature context for Gemini
    all_features_text = "\n".join(
        f"  {FEATURE_META[feat]['display']}: "
        f"{_decode_value(feat, getattr(data, feat)) or getattr(data, feat)}"
        for feat in FEATURE_ORDER
    )

    prompt = textwrap.dedent(f"""
        Kamu adalah konsultan bisnis UMKM yang berpengalaman dan bertugas membantu
        anak muda Indonesia yang ingin memulai usaha. Kamu bekerja di platform
        pencatatan keuangan UMKM yang juga menyediakan fitur prediksi keberhasilan bisnis.
        Gunakan bahasa Indonesia yang ramah, jelas, dan memotivasi.

        # DATA CALON PEBISNIS 
        {all_features_text}

        # HASIL PREDIKSI MODEL ML
        Prediksi    : {label}
        Probabilitas keberhasilan: {pct}%

        # FAKTOR PENDORONG KEBERHASILAN (SHAP positif)
        {positives_text}

        # FAKTOR PENGHAMBAT (SHAP negatif)
        {negatives_text}

        # TUGASMU 
        Berikan analisis dalam format JSON PERSIS seperti di bawah ini,
        tanpa teks tambahan, tanpa markdown, tanpa kode block:

        {{
          "summary": "<2–3 kalimat ringkasan prediksi dalam bahasa manusia. Sebutkan probabilitas. Jangan terlalu teknis.>",
          "strengths": [
            "<kekuatan 1 berdasarkan faktor SHAP positif, spesifik pada data pengguna>",
            "<kekuatan 2>",
            "<kekuatan 3 jika ada>"
          ],
          "weaknesses": [
            "<kelemahan 1 berdasarkan faktor SHAP negatif, spesifik pada data pengguna>",
            "<kelemahan 2>",
            "<kelemahan 3 jika ada>"
          ],
          "recommendations": [
            "<saran konkret & actionable 1 — bisa langsung dilakukan>",
            "<saran konkret & actionable 2>",
            "<saran konkret & actionable 3>",
            "<saran konkret & actionable 4>"
          ]
        }}

        Pastikan setiap poin strengths dan weaknesses LANGSUNG mengacu pada
        nilai data yang diberikan (bukan generik). Rekomendasi harus praktis
        dan relevan untuk UMKM skala kecil di Indonesia.
    """).strip()

    return prompt


def _call_gemini(prompt: str) -> AIAnalysis:
    try:
        response = _gemini_client.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.4,
                max_output_tokens=2048,
                response_mime_type="application/json",
            ),
        )
        raw_text = response.text.strip()

        # Remove markdown
        if raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]

        parsed = json.loads(raw_text)

        return AIAnalysis(
            summary         = parsed["summary"],
            strengths       = parsed.get("strengths", []),
            weaknesses      = parsed.get("weaknesses", []),
            recommendations = parsed.get("recommendations", []),
            encouragement   = parsed.get("encouragement", ""),
        )

    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=502,
            detail=f"Gemini returned non JSON response: {e}. Raw: {raw_text[:300]}"
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Gemini API error: {e}")


@app.get("/health", tags=["Utility"])
def health():
    return {"status": "ok", "model": XGBOOST_MODEL_PATH, "gemini": GEMINI_MODEL}


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(data: UMKMInput):
    # features input
    X = np.array([[getattr(data, f) for f in FEATURE_ORDER]], dtype=float)

    # feed to XGBoost
    pred        = int(_model.predict(X)[0])
    proba       = _model.predict_proba(X)[0]
    prob_fail   = round(float(proba[0]), 4)
    prob_success= round(float(proba[1]), 4)

    # Feature improtance using SHAP 
    shap_values   = _explainer.shap_values(X)
    shap_analysis = _build_shap_analysis(X, shap_values)

    # Gemini Feedback
    prompt      = _build_gemini_prompt(data, pred, prob_success, shap_analysis)
    ai_analysis = _call_gemini(prompt)

    return PredictionResponse(
        prediction          = pred,
        label               = "Berhasil" if pred == 1 else "Tidak Berhasil",
        probability_success = prob_success,
        probability_fail    = prob_fail,
        confidence          = _confidence_label(prob_success),
        shap                = shap_analysis,
        ai_analysis         = ai_analysis,
    )