# VeriLens

## AI-Powered Claim Verification & Evidence Analysis

VeriLens is an AI-powered claim verification system developed as a **hackathon project** to analyze claims, retrieve relevant evidence, and provide evidence-based verification results.

The project combines **Artificial Intelligence, Retrieval-Augmented Generation (RAG), evidence retrieval, and source analysis** into a single verification pipeline.

---

## 🌐 Live Demo

Try VeriLens online:

### [🚀 Launch VeriLens](https://verilens-ai-nzmn.onrender.com/)

> The application is deployed on Render and may take a few moments to start if the service is idle.

---

## ✨ Features

* 🤖 AI-powered claim verification
* 🔎 Evidence retrieval and analysis
* 🧠 Retrieval-Augmented Generation (RAG)
* 📊 Evidence-based reasoning
* 🔗 Source-aware verification
* 📈 Confidence-based assessment
* 🎨 Clean and responsive web interface
* 🌐 Flask backend
* 💻 HTML/CSS/JavaScript frontend
* ☁️ Cloud deployment support

---

## 🔍 How VeriLens Works

```text
                    User Claim
                        │
                        ▼
                Claim Processing
                        │
                        ▼
                 Query Generation
                        │
                        ▼
                Evidence Retrieval
                        │
                        ▼
                 Evidence Analysis
                        │
                        ▼
                   AI Reasoning
                        │
                        ▼
                Verification Result
                        │
                        ▼
               Evidence & Sources
```

VeriLens does not simply generate an answer. It attempts to evaluate a claim using relevant evidence and sources.

---

## 🧪 Example

### Input

```text
The Great Wall of China is visible from the Moon with the naked eye.
```

### Process

VeriLens analyzes the claim, retrieves relevant evidence, evaluates the available information, and produces a verification result with supporting evidence.

---

## 🛠️ Tech Stack

### Backend

* Python
* Flask
* Gunicorn

### AI & Machine Learning

* Generative AI
* Retrieval-Augmented Generation (RAG)
* Natural Language Processing
* Evidence-based reasoning

### Frontend

* HTML5
* CSS3
* JavaScript

### Data Processing

* NumPy
* Pandas

### Deployment

* Render
* Gunicorn

---

## 📁 Project Structure

```text
VeriLens/
│
├── app.py
├── integration.py
├── requirements.txt
├── README.md
│
├── src/
│   └── ai/
│       ├── claim_extractor.py
│       ├── evidence_retriever.py
│       ├── evidence_ranker.py
│       ├── verifier.py
│       ├── scorer.py
│       ├── llm.py
│       └── pipeline.py
│
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── script.js
│
├── templates/
│   └── index.html
│
└── tests/
    ├── test_claims.py
    ├── test_evidence.py
    ├── test_pipeline.py
    ├── test_ranker.py
    ├── test_scorer.py
    └── test_verifier.py
```

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/vaishnavi-rathiii/verilens-ai.git
cd verilens-ai
```

### 2. Create a Virtual Environment

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔐 Environment Variables

If VeriLens requires API credentials, store them as environment variables.

For local development, create a `.env` file:

```env
YOUR_API_KEY=your_api_key_here
```

> **Never commit API keys, passwords, or other secrets to GitHub.**

For cloud deployment, configure secrets through the hosting provider's environment-variable settings.

---

## 💻 Running Locally

Start the Flask application:

```bash
python app.py
```

Or run it using Gunicorn:

```bash
gunicorn app:app
```

The application should be available at:

```text
http://127.0.0.1:5000
```

---

## ☁️ Deployment

VeriLens is deployed as a Flask web service using Render.

### Render Configuration

| Setting       | Value                             |
| ------------- | --------------------------------- |
| Runtime       | Python 3                          |
| Branch        | `main`                            |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `gunicorn app:app`                |
| Instance      | Free                              |

### Live Application

[**https://verilens-ai-nzmn.onrender.com/**](https://verilens-ai-nzmn.onrender.com/)

Environment variables and API keys should be configured through Render instead of being committed to the repository.

---

## 🧠 Verification Philosophy

VeriLens is built around a simple principle:

> **A claim should be evaluated using evidence, not simply accepted because an AI model generated an answer.**

The system focuses on:

* Finding relevant information
* Retrieving supporting evidence
* Comparing available evidence
* Evaluating sources
* Producing a transparent verification result

---

## 📊 Verification Pipeline

```text
Claim
  │
  ▼
Understand
  │
  ▼
Search
  │
  ▼
Retrieve Evidence
  │
  ▼
Analyze Evidence
  │
  ▼
Compare Sources
  │
  ▼
AI Reasoning
  │
  ▼
Final Verification
```

---

## 🏆 Hackathon Project

VeriLens was developed as a **team-based hackathon project** under the team:

### Algorithm Avengers

The project focuses on combining AI, RAG, evidence retrieval, and web technologies to address the challenge of evaluating the credibility of factual claims.

---

## 👥 Team & Collaborators

### Algorithm Avengers

#### 👑 Team Lead

* [Vaishnavi Rathi](https://github.com/vaishnavi-rathiii)

#### 🤝 Collaborators

* [KrishDalvi](https://github.com/KrishDalvi)
* [deoreshree](https://github.com/deoreshree)

---

## 🚧 Current Status

**Active Development**

VeriLens is currently under active development.

Current development areas include:

* Verification pipeline
* Evidence retrieval
* RAG integration
* AI integration
* User interface
* Testing
* Cloud deployment

---

## ❤️ Acknowledgement

> Built with ❤️ by **Algorithm Avengers**

---

## ⚠️ Disclaimer

VeriLens is an experimental AI-powered verification system.

AI-generated verification should not be treated as an absolute guarantee of truth. Users should review the provided evidence and sources, especially when dealing with important or high-stakes claims.

---

## 👨‍💻 Project Vision

Building VeriLens as an exploration of:

```text
AI × RAG × Evidence × Verification
```

---

## 📄 License

This project is licensed under the **MIT License**.

### MIT License

Copyright (c) 2026 **Algorithm Avengers**

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

**© 2026 Algorithm Avengers. All rights reserved where applicable.**
