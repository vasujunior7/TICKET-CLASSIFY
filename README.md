# Support Ticket Classifier

Categorize support tickets into: **Billing**, **Technical**, or **Other** using DistilBERT fine-tuned with Hugging Face Transformers. Exposes a FastAPI REST API for predictions.

## Project Structure
- `datasets/support_tickets.csv` — Labeled support ticket data
- `train.py` — Fine-tune DistilBERT and save model to `./model/`
- `app.py` — FastAPI app exposing `/predict` endpoint
- `model/` — Saved model and tokenizer

## Setup Instructions

1. **Install dependencies**
   ```bash
   pip install torch transformers datasets fastapi uvicorn scikit-learn
   ```

2. **Train the model**
   ```bash
   python train.py
   ```
   This will save the model and tokenizer to `./model/`.

3. **Run the API server**
   ```bash
   uvicorn app:app --reload
   ```

## API Usage

### Endpoint
- `POST /predict`
- Request body (JSON):
  ```json
  { "input": "I need help updating my billing address" }
  ```

### Example curl request
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"input": "I need help updating my billing address"}'
```

### Example response
```json
{
  "input": "I need help updating my billing address",
  "prediction": "Billing",
  "confidence": {
    "Billing": 0.92,
    "Technical": 0.05,
    "Other": 0.03
  }
}
```

## Notes
- The model expects the same label set as in the training data: `Billing`, `Technical`, `Other`.
- For best results, ensure your `support_tickets.csv` is formatted as:
  | text | label |
  |------|-------|
  | ...  | ...   |
