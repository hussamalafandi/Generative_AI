# 🚀 **Abschlussprojekt: „Multimodales Marktanalyse-KI-System“**

## 📌 **Projektzusammenfassung:**

Die Studierenden entwickeln ein multimodales KI-System, das in der Lage ist, marktspezifische Anfragen zu beantworten, Investitionseinblicke zu liefern, historische Marktleistungen zu analysieren, Prognosen zu erstellen und Finanzdaten zu visualisieren. Das System besteht aus spezialisierten, kooperierenden Agenten, die von einem zentralen Koordinator-Agenten gesteuert werden und reale Finanzdaten ausschließlich aus Investor-Relations-(IR-)Veröffentlichungen (PDF-Berichte, Präsentationsfolien, Transkripte von Ergebnispräsentationen usw.) von **Apple, Microsoft, Google, NVIDIA und Meta** aus den **vergangenen 5 Jahren** nutzen.

---

## 🖥️ **Systemübersicht & Rollen der Agenten**

Das Multi-Agenten-Framework beinhaltet klar definierte, spezialisierte Agenten:

### 🌟 **1. Multimodaler Agentic RAG-Spezialist**

**Hauptverantwortung:**

* Bearbeitung multimodaler Finanzanfragen (Textfragen, Finanztabellen, Bilder/Diagramme, PDFs).
* Abruf relevanter Finanzdaten speziell aus IR-Dokumenten von Apple, Microsoft, Google, NVIDIA und Meta (letzte 5 Jahre).
* Bereitstellung präziser und zitierter Antworten, ausschließlich basierend auf diesen Quellen.

**Fähigkeiten & Aufgaben:**

* Erzeugung multimodaler Embeddings (CLIP, SentenceTransformers).
* Dokumentenindexierung und -abruf (Chroma).
* Antwortsynthese mit expliziten **Quellenangaben**.

**Beispielhafte Nutzeranfrage:**

> „Fassen Sie die jüngste Finanzleistung von NVIDIA basierend auf dieser Ergebnispräsentation zusammen.“

**Beispielausgabe:**

> „NVIDIAs Umsatz im Q4 GJ24 stieg um 18 %, getrieben durch starke GPU-Verkäufe (Quelle: NVIDIA Q4 FY24 Earnings Slides, Seite 5).“

---

### 🌟 **2. Datenwissenschafts- & Analyse-Agent**

**Hauptverantwortung:**

* Durchführung fortgeschrittener Marktanalysen, Trendanalysen und prädiktiver Modellierung.
* Erstellung von Prognosen, erklärenden Erkenntnissen und Visualisierungen.

**Fähigkeiten & Aufgaben:**

* Extraktion strukturierter Daten aus IR-Dokumenten (Finanztabellen, Ergebnisdaten).
* Prognosen und prädiktive Modellierung (z. B. Aktienkursprognosen mit Prophet/ARIMA).
* Erstellung von Visualisierungen (Matplotlib, Plotly).
* Generierung erklärender Texte zu Analyseergebnissen.

**Beispielhafte Nutzeranfrage:**

> „Analysieren Sie Microsofts Aktienentwicklung im letzten Jahr und prognostizieren Sie die Performance im nächsten Quartal.“

**Beispielausgabe:**

* Interaktive Aktienkurs-Visualisierung.
* Prognose für das nächste Quartal mit klar dargestellten Konfidenzintervallen und erläuterndem Text.

---

### 🌟 **3. Websuche- & Echtzeit-Markt-Agent**

**Hauptverantwortung:**

* Echtzeitbeschaffung von Marktnachrichten, Finanzereignissen und aktueller Stimmung.
* Extraktion aktueller Informationen aus seriösen Online-Quellen.

**Fähigkeiten & Aufgaben:**

* Web-Scraping und Datenabruf in Echtzeit (Yahoo Finance, Alpha Vantage, NewsAPI).
* Klare Zusammenfassung der aktuellen Marktlage und Updates mit Quellenangabe.

**Beispielhafte Nutzeranfrage:**

> „Was sind die neuesten Nachrichten, die heute den Aktienkurs von Google beeinflussen?“

**Beispielausgabe:**

> "Googles Aktie stieg heute um 3 %, ausgelöst durch positive Reaktionen auf neue KI-Produktankündigungen (Quelle: CNBC, Mai 2025)."

---

### 🌟 **4. Koordinator-Agent**

**Hauptverantwortung:**

* Orchestrierung komplexer Anfragen, Aufgabenzerlegung und Koordination der Agenten.
* Aggregation der Ergebnisse in kohärente, zitierte Zusammenfassungen.

**Fähigkeiten & Aufgaben:**

* Aufgabenzerlegung und Delegierung (LangChain, LangGraph).
* Workflow-Koordination und Integration der Antworten.

**Beispielhafter Workflow:**

* Zerlegung einer multimodalen Anfrage:

  * Datenabruf und Synthese (RAG-Agent).
  * Prognose und Visualisierung (Datenanalyse-Agent).
  * Echtzeitnachrichten und Stimmung (Websuche-Agent).
* Zusammenführung zu einer einheitlichen, zitierten Analyse.

---

### 🌟 **(Optional) 5. Qualitätssicherungs- & Ethik-KI-Prüfer**

**Hauptverantwortung:**

* Sicherstellung der Genauigkeit, Zuverlässigkeit und ethischen Integrität der Ausgaben.
* Validierung der Fakten und Quellenangaben.

**Fähigkeiten & Aufgaben:**

* Automatisierte Moderation, Bias-Prüfungen, Faktenüberprüfung.
* Sicherstellung von Transparenz, Fairness und ethischer Konformität.

---

## 🎨 **System-Workflow (Beispielszenario):**

1. **Benutzeranfrage (multimodaler Input):**

   > "Basierend auf diesen aktuellen Diagrammen und Nachrichten: Fassen Sie Metas Aktienleistung zusammen und geben Sie eine Prognose für das nächste Quartal ab."

2. **Koordinator-Agent:**

   * Analysiert die Anfrage.
   * Leitet Aufgaben an die passenden Agenten weiter.

3. **Einzelne Agenten antworten:**

   * **RAG-Agent:** Fasst bereitgestellte IR-Dokumente zusammen.
   * **Websuche-Agent:** Ruft aktuelle Marktnachrichten und Stimmung ab.
   * **Datenanalyse-Agent:** Erstellt Kursprognosen und Visualisierungen.

4. **Koordinator-Agent aggregiert:**

   * Generiert eine integrierte, multimodale Finanzanalyse mit Quellenangaben.

5. **(Optional) QA-Agent:** Prüft Antwortqualität, Quellen und ethische Einhaltung.

6. **Endergebnis:** Präsentation über eine **Gradio-Benutzeroberfläche auf Hugging Face Spaces**.

---

## 🛠️ **Empfohlener Technologiestack**

| **Agent**                   | **Tools/Modelle**                                                                  |
| --------------------------- | ---------------------------------------------------------------------------------- |
| **Agentic RAG-Spezialist**  | CLIP, SentenceTransformers, LangChain, Chroma, Gemini (QLoRA Fine-Tuning optional) |
| **Datenanalyse-Agent**      | Pandas, Matplotlib, Plotly, Prophet, scikit-learn, Gemini                          |
| **Websuche-Agent**          | SerpAPI/NewsAPI, Tavily, BeautifulSoup, newspaper3k, Gemini/HF API                 |
| **Koordinator-Agent**       | LangChain-Agentenframework, Gemini (API-basiert)                                   |
| **QA & Ethik-Prüfer-Agent** | BERT-Klassifikatoren, GPT-Moderation-API, Hugging Face Evaluationstools            |

---

## 🎯 **Datensatz (explizit definiert):**

* **Investor-Relations-Dokumente (2020–2024)** von:

  * Apple, Microsoft, Google, NVIDIA, Meta
* Dokumenttypen:

  * Jahresberichte (10-K), Quartalsberichte (10-Q)
  * Transkripte und Folien von Ergebnispräsentationen
  * Investorenpräsentationen, Diagramme, Grafiken

---

## 🧑‍💻 **Studentischer Workflow (Agil):**

* **Woche 1:**

  * Beschaffung und Aufbereitung der IR-Daten.
  * Verarbeitung multimodaler Dokumente und Erstellung von Embeddings.
  * Erste Implementierung der Agenten.
  * Vollständige RAG-Implementierung inkl. Datenabruf und Quellenangabe.
  * Analyse-Agent: Prognosen und Visualisierungen.

* **Woche 2:**

  * Integration von Websuche und Echtzeitdaten.
  * Umsetzung des Koordinator-Agenten.
  * Integration aller Agenten.
  * QA-Agent und ethische Validierung (optional).
  * Fine-Tuning der RAG- und QA-Agenten auf bereitgestelltem Datensatz (optional).
  * UI-Entwicklung mit Gradio; finale Bereitstellung.

---

## 📦 **Endabgaben:**

* 🚀 **Gradio-App auf Hugging Face Spaces**
* 📁 **Dokumentiertes GitHub-Repository**
* 📊 **Jira-Projektboard (agile Dokumentation)**
* 🎬 **Präsentation & Demo**
* 📑 **Technischer Bericht (Architektur, Entscheidungen, Reflexionen)**

---

## ✅ **Warum dieses System?**

Die Studierenden erwerben praktische Erfahrungen, die direkt mit Berufsrollen in der Finanzanalyse und generativen KI verbunden sind:

* Fortschrittlicher multimodaler Datenabruf
* Finanzdatenanalyse und -visualisierung
* Prädiktive Analytik und Prognoseerstellung
* Web-Scraping und Integration von Echtzeitdaten
* Agiles Teamwork, Projektmanagement mit Jira
* Fähigkeiten zur produktionsreifen Bereitstellung

Dieses Projekt spiegelt exakt die Art von multimodalen KI-Analysetools wider, die aktuell in der Industrie eingesetzt werden – ein klarer Pluspunkt für die spätere Beschäftigungsfähigkeit.

---

## **Ressourcen**

* [Workflows und Agenten](https://langchain-ai.github.io/langgraph/tutorials/workflows/)
* [Multi-Agent-Supervisor](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/agent_supervisor/)
* [Multi-Vector Retriever für RAG mit Tabellen, Text & Bildern](https://blog.langchain.dev/semi-structured-multi-modal-rag/)
* [Multimodale Daten an Modelle übergeben](https://python.langchain.com/docs/how_to/multimodal_inputs/)
* [Multi-Agent-Systeme](https://langchain-ai.github.io/langgraph/concepts/multi_agent/)
* [Quellenangaben in RAG-Anwendungen](https://python.langchain.com/docs/how_to/qa_sources/)
* [RAG-Antworten mit Zitaten anreichern](https://python.langchain.com/docs/how_to/qa_citations/)
