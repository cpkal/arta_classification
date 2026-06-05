# UMKM Success Prediction API

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

| # | Field | Deskripsi | Nilai |
|---|---|---|---|
| 1 | `Age` | Usia pemilik usaha | Numerik (tahun) |
| 2 | `Education` | Tingkat pendidikan pemilik | 1=SD, 2=SMP, 3=SMA, 4=Sarjana, dst. |
| 3 | `Initial_Capital` | Kecukupan modal awal | 1=Memadai, 0=Tidak Memadai |
| 4 | `Financial_Record_Keeping` | Kualitas pencatatan keuangan | 1=Baik, 0=Buruk |
| 5 | `Internet_Usage` | Penggunaan internet | 1=Ya, 0=Tidak |
| 6 | `Business_Plan` | Ada tidaknya perencanaan bisnis | 1=Ada, 0=Tidak Ada |
| 7 | `Marketing_Effort` | Tingkat usaha pemasaran | Skala Likert 1–7 |
| 8 | `Partnership` | Kemitraan | 1=Ya, 0=Tidak |
| 9 | `Parent_Business_Experience` | Pengalaman bisnis orang tua | 1=Pernah, 0=Tidak Pernah |
| 10 | `Industry_Experience` | Pengalaman pemilik di industri | Numerik (tahun) |
| 11 | `Owner_Gender` | Jenis kelamin pemilik | 0=Perempuan, 1=Laki-laki |
| 12 | `Professional_Advice` | Frekuensi konsultasi dengan profesional | Skala Likert 1–7 |

**Output Label (`status_keberhasilan`):** `1` = Berhasil, `0` = Tidak Berhasil

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