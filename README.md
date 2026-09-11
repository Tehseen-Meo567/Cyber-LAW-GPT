# ⚖️ Cyber Law GPT

A free-to-run Retrieval-Augmented-Generation (RAG) chatbot that answers
questions **strictly grounded in a cyber-law PDF document**. Built with
Streamlit, sentence-transformers, FAISS, and the free-tier Groq LLM API.

> ⚠️ **Disclaimer:** This tool provides general information drawn only from
> the supplied document. It is not legal advice. Consult a qualified lawyer
> for guidance on a specific situation.

---

## How it works

1. On startup the app looks for the cyber-law PDF in this order:
   1. **A `.pdf` file committed in the same folder as `app.py`** (most
      reliable — recommended for deployment, no network dependency).
   2. A Google Drive link (used only if no PDF is bundled locally).
   3. A manual upload via the sidebar (always available as a fallback).
2. The PDF text is extracted page-by-page and split into overlapping chunks.
3. Each chunk is embedded locally with `sentence-transformers`
   (`all-MiniLM-L6-v2` — free, no API key, runs on CPU) and indexed in FAISS.
   This embedding step happens once per document and is cached.
4. When you ask a question, the app retrieves the most relevant chunks and
   sends them, along with your question, to a free Groq LLM
   (`llama-3.3-70b-versatile`) to generate a grounded answer.
5. If you don't provide a Groq API key, the app still works — it shows you
   the most relevant passages from the document directly (no generated text).

## Features

- **Technical level** — Beginner / Intermediate / Expert phrasing of answers.
- **Response size** — Short / Medium / Detailed.
- **Answer language** — English / Urdu.
- Toggle to show the exact source passages/pages used for each answer.
- 100% free to run: no paid API, no paid database, no paid hosting required.

---

## Files

| File               | Purpose                                  |
|--------------------|-------------------------------------------|
| `app.py`           | The full Streamlit application            |
| `requirements.txt` | Python dependencies                       |
| `README.md`        | This file                                 |

---

## Get a free Groq API key (optional but recommended)

1. Go to https://console.groq.com/keys
2. Sign up (free) and create an API key.
3. Paste it into the "Groq API key" field in the app's sidebar.

Groq's free tier is generous and requires no credit card.

---

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`).

---

## Run on Google Colab

```python
!pip install -r requirements.txt -q

# Streamlit needs a tunnel to be viewable from Colab.
!npm install -g localtunnel -q

!streamlit run app.py &>/content/logs.txt &
import time; time.sleep(5)
!npx localtunnel --port 8501
```

Open the printed `localtunnel` URL. The password it may ask for is your
public IP, printed by running `!curl https://loca.lt/mytunnelpassword` in a
separate cell.

---

## Deploy on Streamlit Community Cloud (free)

1. Push these 3 files to a public (or private) GitHub repository.
2. Go to https://share.streamlit.io and click "New app".
3. Select your repo, branch, and `app.py` as the entry point.
4. (Optional) Add your Groq key as a secret instead of typing it every time:
   In the app's **Settings → Secrets**, add:
   ```toml
   GROQ_API_KEY = "your-key-here"
   ```
   The app automatically reads `GROQ_API_KEY` from the environment as the
   default value for the sidebar field.
5. Click **Deploy**. First load will take a minute while dependencies install
   and the document is embedded; after that it's cached.

---

## Deploy on any generic cloud VM

```bash
git clone <your-repo>
cd <your-repo>
pip install -r requirements.txt
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

Open port 8501 in your cloud provider's firewall/security group settings.

---

## Using your own document

- **Recommended:** commit a `.pdf` file into the same repo folder as
  `app.py`. The app auto-detects any PDF sitting next to it and uses it
  first — no download step, no permission issues, works identically on
  Streamlit Cloud, Colab, or any server.
- Otherwise, put a **public** Google Drive share link (Anyone with the
  link → Viewer) in the sidebar as a fallback.
- Or use the "upload the PDF manually" option in the sidebar — this always
  works regardless of the other two.

## Notes on the source PDF

Bundling the PDF directly in the repo (option 1 above) is the most reliable
approach: Google Drive automated downloads can be blocked by file-size virus
scans, rate limits, or permission changes, none of which affect a file that
ships with your code.
