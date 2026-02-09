# A.R.I.A. - Automatic Reporting and Intelligent Analysis

> Offline-first healthcare AI for real-time medical transcription, entity extraction, ICD-10 coding, and FHIR-compliant documentation.

![A.R.I.A. Banner](https://img.shields.io/badge/Healthcare-AI-blue?style=for-the-badge&logo=health&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10+-green?style=flat-square&logo=python)
![Next.js](https://img.shields.io/badge/Next.js-14-black?style=flat-square&logo=next.js)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

## 🚀 Features

- **🎙️ Real-time Transcription**: Faster-Whisper ASR with int8 quantization
- **🧠 Multi-Agent AI Pipeline**: LangGraph workflow with Scribe, Coder, and Auditor agents
- **🔍 ICD-10 Coding**: ChromaDB-powered RAG for automatic diagnosis coding
- **📋 SOAP Note Generation**: Structured medical documentation
- **✅ FHIR/ABDM Compliance**: Ayushman Bharat Digital Mission compatible output
- **💻 Offline-First**: Runs entirely on local hardware (GTX 1650 optimized)

## 📁 Project Structure

```
├── backend/
│   ├── data/
│   │   ├── slang_dictionary.json    # Medical slang normalization
│   │   └── icd10_sample.json        # ICD-10 codes for RAG
│   ├── services/
│   │   └── transcriber.py           # Whisper ASR module
│   ├── agent_graph.py               # LangGraph multi-agent workflow
│   ├── main.py                      # FastAPI WebSocket server
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   │   ├── app/                     # Next.js App Router
│   │   ├── components/              # UI Components
│   │   └── hooks/                   # Custom React hooks
│   └── package.json
```

## 🛠️ Prerequisites

- **Python 3.10+**
- **Node.js 18+**
- **NVIDIA GPU** with CUDA (GTX 1650 4GB or better)
- **Ollama** with `phi3:mini` model

## 📦 Installation

### Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
.\venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Pull Phi-3 model via Ollama
ollama pull phi3:mini
```

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Create .env.local (optional)
echo "NEXT_PUBLIC_WS_URL=ws://localhost:8000/ws/listen" > .env.local
```

## 🚀 Running the Application

### Start Backend

```bash
cd backend
python main.py
# Server runs on http://localhost:8000
```

### Start Frontend

```bash
cd frontend
npm run dev
# App runs on http://localhost:3000
```

## 🧪 Testing

### Demo Mode
Click "Demo Mode" in the UI to test with sample medical text without microphone.

### API Testing
```bash
# Health check
curl http://localhost:8000/api/health

# Process text directly
curl -X POST http://localhost:8000/api/process \
  -H "Content-Type: application/json" \
  -d '{"text": "Patient has high sugars and BP issues"}'
```

## ⚡ Performance Optimization

| Component | Optimization | VRAM Usage |
|-----------|--------------|------------|
| Whisper | int8 quantization | ~1GB |
| Phi-3-Mini | 4-bit via Ollama | ~2GB |
| ChromaDB | CPU-based | 0GB |
| **Total** | Sequential loading | **<3.5GB** |

## 📄 FHIR Output Example

```json
{
  "resourceType": "Composition",
  "type": {"text": "OPConsultRecord"},
  "section": [
    {"title": "Subjective", "text": "..."},
    {"title": "Objective", "text": "..."},
    {"title": "Assessment", "text": "...", "codes": [{"code": "E11.9", "description": "Type 2 diabetes"}]},
    {"title": "Plan", "text": "..."}
  ]
}
```

## 🙏 Acknowledgments

- [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) for efficient ASR
- [LangGraph](https://github.com/langchain-ai/langgraph) for agent orchestration
- [Ollama](https://ollama.ai/) for local LLM inference
- [ChromaDB](https://www.trychroma.com/) for vector storage

## 📜 License

MIT License - See [LICENSE](LICENSE) for details.