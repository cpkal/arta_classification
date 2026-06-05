# Arta

API untuk memprediksi keberhasilan UMKM menggunakan XGBoost + SHAP + Gemini AI.

---

## Requirements

```
fastapi
uvicorn
numpy
shap
xgboost
google-generativeai
pydantic
```

Install:
```bash
pip install fastapi uvicorn numpy shap xgboost google-generativeai pydantic
```

---

## Setup

### 1. Siapkan environment variables

```bash
export XGBOOST_MODEL_PATH=xgboost.json   # path ke file model
export GEMINI_API_KEY=your_api_key_here  # wajib diisi
export GEMINI_MODEL=gemini-3.5-flash     # opsional, default: gemini-3.5-flash
```

### 2. Jalankan server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

---

## Endpoints

### `GET /health`
Cek status server.

**Response:**
```json
{
  "status": "ok",
  "model": "xgboost.json",
  "gemini": "gemini-3.5-flash"
}
```

---

### `POST /predict`
Kirim data UMKM, dapatkan prediksi + penjelasan SHAP + analisis AI.

**Request Body:**

| Field | Tipe | Range | Keterangan |
|---|---|---|---|
| `Age` | int | 15–80 | Usia pemilik (tahun) |
| `Education` | int | 1–5 | 1=SD, 2=SMP, 3=SMA, 4=D3/S1, 5=S2/S3 |
| `Initial_Capital` | int | 0–1 | 0=Tidak punya modal, 1=Punya modal |
| `Financial_Record_Keeping` | int | 0–1 | 0=Tidak mencatat, 1=Mencatat |
| `Internet_Usage` | int | 0–1 | 0=Tidak pakai internet, 1=Pakai |
| `Business_Plan` | int | 0–1 | 0=Tidak punya rencana, 1=Punya |
| `Marketing_Effort` | int | 1–10 | Skala upaya pemasaran |
| `Partnership` | int | 0–1 | 0=Tidak punya mitra, 1=Punya |
| `Parent_Business_Experience` | int | 0–1 | 0=Orang tua tidak berbisnis, 1=Berbisnis |
| `Industry_Experience` | int | 0–40 | Pengalaman di industri (tahun) |
| `Owner_Gender` | int | 0–1 | 0=Perempuan, 1=Laki-laki |
| `Professional_Advice` | int | 1–7 | Skala akses saran profesional |

**Contoh Request:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Age": 28,
    "Education": 3,
    "Initial_Capital": 1,
    "Financial_Record_Keeping": 1,
    "Internet_Usage": 1,
    "Business_Plan": 1,
    "Marketing_Effort": 7,
    "Partnership": 0,
    "Parent_Business_Experience": 1,
    "Industry_Experience": 3,
    "Owner_Gender": 1,
    "Professional_Advice": 5
  }'
```

**Response mencakup:**
- `prediction` — 0 (gagal) atau 1 (berhasil)
- `label` — "Berhasil" / "Tidak Berhasil"
- `probability_success` / `probability_fail` — skor probabilitas (0–1)
- `confidence` — "Tinggi" / "Sedang" / "Rendah"
- `shap` — breakdown kontribusi tiap fitur
- `ai_analysis` — ringkasan, kekuatan, kelemahan, dan rekomendasi dari Gemini AI

---

## Docs (Swagger UI)

Setelah server berjalan, buka:
```
http://localhost:8000/docs
```