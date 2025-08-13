# 🌊 AI-Powered Flood Risk Assessment System

![Project Banner](https://ai-powered-flood-risk-assessment-to.vercel.app/og-image.png)

A **cutting-edge MVP** for flood risk assessment that leverages AI to analyze geographic coordinates and terrain images, providing actionable insights for disaster preparedness.

🔗 **[Live Frontend Demo](https://ai-powered-flood-risk-assessment-to.vercel.app/)**  
🔗 **[Live Backend API](https://ai-powered-flood-risk-assessment-tool.onrender.com)**

---

## ✨ Key Features

### **Dual Analysis Modes**
- **Coordinate Assessment** — Enter precise latitude/longitude for location-based risk evaluation.
- **Image Analysis** — Upload terrain photos for AI-powered visual flood risk assessment.

### **Intelligent Risk Evaluation**
- Powered by **Google's Gemini AI** for comprehensive analysis.
- Considers **elevation**, **proximity to water bodies**, and **historical data**.

### **Interactive Visualization**
- Dynamic map display with **risk overlays**.
- Clear visual indicators of threat levels.

### **Actionable Insights**
- Risk level classification (`Low` / `Medium` / `High`).
- Prevention recommendations.
- Historical context for the area.

---

## 🚀 Quick Deployment

### **Prerequisites**
- Node.js v18+
- Python 3.9+
- Gemini API Key

### **Recommended Setup (Single Command)**
```bash
# Configure environment
cd backend
echo "GEMINI_API_KEY=your_actual_key_here" > .env

# Launch full stack
./start-dev.sh
```

Access:
- **Frontend** → http://localhost:3000  
- **Backend API** → http://localhost:8000  
- **API Docs** → http://localhost:8000/docs  

---

### **Alternative Setup**

#### Backend Service
```bash
cd backend
python3 -m pip install -r requirements.txt
python3 start.py
```

#### Frontend Service
```bash
npm install
npm run dev
```

---

## 🖥️ User Guide

1. **Select Analysis Mode**  
   - **Coordinates** → For precise geographic evaluation.  
   - **Image Upload** → For visual terrain assessment.  

2. **Submit Data**  
   - Enter valid coordinates (*Decimal Degrees format*).  
   - Or upload clear terrain/waterway images.  

3. **Review Results**  
   - Risk level indicator.  
   - Key factors influencing assessment.  
   - Recommended precautions.  
   - Interactive map visualization.  

---

## 🔌 API Reference

### **Core Endpoints**

| Method | Endpoint | Payload | Description |
|--------|----------|---------|-------------|
| **POST** | `/api/analyze/coordinates` | `{ "latitude": float, "longitude": float }` | Analyze flood risk by coordinates |
| **POST** | `/api/analyze/image` | Form-data with image file | Analyze flood risk from uploaded image |
| **GET**  | `/health` | None | Service health check |

📄 **Full API Documentation** → [https://your-backend-service-on-render.com/docs](https://your-backend-service-on-render.com/docs)

---

## 🛠 Technical Architecture

### **Frontend**
- Framework: **Next.js 15** (App Router)
- Language: **TypeScript**
- Styling: **Tailwind CSS** + [shadcn/ui](https://ui.shadcn.com/)
- Mapping: **Google Maps JavaScript API**
- State Management: **React Context API**

### **Backend**
- Framework: **FastAPI**
- AI Engine: **Google Gemini Pro**
- Validation: **Pydantic models**
- Runtime: **Python 3.9+**

---

## 📂 Project Structure
```
flood-risk-assessment/
├── app/                    # Next.js application
│   ├── components/         # React components
│   ├── lib/                # Utilities and hooks
│   └── page.tsx            # Main interface
├── backend/                # API service
│   ├── main.py             # FastAPI application
│   ├── schemas.py          # Data models
│   └── services/           # Business logic
├── public/                 # Static assets
└── README.md               # Project documentation
```

---

## 🌐 Hosting & Deployment

- **Frontend** → Hosted on [Vercel](https://vercel.com/)  
- **Backend** → Hosted on [Render](https://render.com/)  

**Production Environment Variables:**
```env
GEMINI_API_KEY=your_production_key
NEXT_PUBLIC_MAPS_API_KEY=optional_google_maps_key
```

---

## 📈 Roadmap

- [ ] Historical flood data integration  
- [ ] Multi-image analysis support  
- [ ] Mobile application port  
- [ ] Community reporting features  

---

## 🤝 Contributing

We welcome contributions!  

1. Fork the repository.  
2. Create a new branch (`feature/new-feature`).  
3. Commit your changes.  
4. Push to the branch.  
5. Open a Pull Request.  

For major changes, open an issue first to discuss the proposed modifications.

---

## 📜 License
MIT License — see [LICENSE](LICENSE) for details.

---

## 👤 Author

**Thanka Bharathi** — Data Scientist & AI/ML Developer  
📧 Email: [thankabharathi0@gmail.com]  
🔗 LinkedIn: [https://www.linkedin.com/in/thankabharathi/]  
🐙 GitHub: [https://github.com/ThankaBharathi]  
