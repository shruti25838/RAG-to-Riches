import React, { useState, useEffect } from "react";
import "./App.css";

function App() {
  const [files, setFiles] = useState([]); // For multiple files upload
  const [uploadMessage, setUploadMessage] = useState("");
  const [jdText, setJdText] = useState("");
  const [summary, setSummary] = useState(null);
  const [chunks, setChunks] = useState([]);
  const [darkMode, setDarkMode] = useState(false);
  const [chatHistory, setChatHistory] = useState(() => {
    const saved = localStorage.getItem("rag_chat_history");
    return saved ? JSON.parse(saved) : [];
  });
  const [currentHistoryIndex, setCurrentHistoryIndex] = useState(null);
  const [uploadLoading, setUploadLoading] = useState(false);
  const [analyzeLoading, setAnalyzeLoading] = useState(false);
  const [conversation, setConversation] = useState([]);
  const [isFollowUp, setIsFollowUp] = useState(false);

  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [activeTab, setActiveTab] = useState("analyze"); // analyze, assess
  const [assessmentResult, setAssessmentResult] = useState(null);
  const [assessLoading, setAssessLoading] = useState(false);

  useEffect(() => {
    if (darkMode) {
      document.body.classList.add("dark-mode");
    } else {
      document.body.classList.remove("dark-mode");
    }
  }, [darkMode]);

  useEffect(() => {
    localStorage.setItem("rag_chat_history", JSON.stringify(chatHistory));
  }, [chatHistory]);

  // Fetch uploaded files list
  const fetchUploadedFiles = async () => {
    try {
      const response = await fetch("http://127.0.0.1:5000/list-uploaded-files");
      const data = await response.json();
      if (response.ok) {
        setUploadedFiles(data.uploaded_files || []);
      }
    } catch (error) {
      console.error("Failed to fetch uploaded files:", error);
    }
  };

  useEffect(() => {
    fetchUploadedFiles();
  }, []);

  const handleFileChange = (e) => {
    setFiles(Array.from(e.target.files));
    setUploadMessage("");
  };

  const handleUpload = async () => {
    if (files.length === 0) {
      setUploadMessage("Please select at least one PDF file.");
      return;
    }

    setUploadLoading(true);
    const formData = new FormData();
    files.forEach((file) => formData.append("files", file));

    try {
      const response = await fetch("http://127.0.0.1:5000/upload-pdf", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();

      if (response.ok) {
        setUploadMessage("Upload complete.");
        await fetchUploadedFiles();
      } else {
        setUploadMessage(data.error || "Upload failed.");
      }
    } catch (error) {
      console.error(error);
      setUploadMessage("Upload failed.");
    }
    setUploadLoading(false);
  };

  const handleAnalyze = async () => {
    if (!jdText.trim()) {
      setSummary(null);
      setChunks([]);
      setCurrentHistoryIndex(null);
      setConversation([]);
      setIsFollowUp(false);
      return;
    }

    setAnalyzeLoading(true);

    const multiTurnQuery = isFollowUp
      ? conversation.map((c) => `Q: ${c.query}\nA: ${c.response}`).join("\n") + `\nQ: ${jdText}`
      : jdText;

    try {
      const response = await fetch("http://127.0.0.1:5000/qa", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: multiTurnQuery,
          top_k: 5,
          chat_history: conversation,
          uploaded_files: uploadedFiles,
        }),
      });
      const data = await response.json();

      setSummary(data.summary);
      setChunks(data.chunks_used || []);
    } catch (error) {
      console.error(error);
      setSummary(null);
      setChunks([]);
    }
    setAnalyzeLoading(false);

    if (isFollowUp) {
      setConversation((prev) => [...prev, { query: jdText, response: summary?.Summary || "" }]);
    } else {
      setConversation([{ query: jdText, response: summary?.Summary || "" }]);
    }

    setChatHistory((prev) => {
      const newEntry = {
        query: jdText,
        summary: summary,
        chunks_used: chunks || [],
        timestamp: new Date().toISOString(),
      };
      return [...prev, newEntry];
    });
    setCurrentHistoryIndex(chatHistory.length);
  };

  const handleAssessResume = async () => {
    if (!jdText.trim()) {
      alert("Please enter a job description first.");
      return;
    }
    if (uploadedFiles.length === 0) {
      alert("Please upload at least one resume first.");
      return;
    }

    const resumeToAssess = uploadedFiles[uploadedFiles.length - 1]; // last uploaded file

    setAssessLoading(true);

    try {
      const response = await fetch("http://127.0.0.1:5000/assess-resume", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          job_description: jdText,
          resume_file: resumeToAssess,
          top_k: 15,
          chat_history: conversation,
        }),
      });
      const data = await response.json();

      if (response.ok) {
        setAssessmentResult(data);
      } else {
        alert(data.error || "Assessment failed.");
        setAssessmentResult(null);
      }
    } catch (error) {
      console.error(error);
      alert("Assessment failed.");
      setAssessmentResult(null);
    }
    setAssessLoading(false);
  };

  const toggleFollowUp = () => {
    setIsFollowUp(!isFollowUp);
    if (!isFollowUp) {
      alert("Follow-up mode enabled. Queries will use conversation history.");
    } else {
      alert("Follow-up mode disabled. New queries start fresh.");
      setConversation([]);
    }
  };

  const loadHistoryItem = (index) => {
    const item = chatHistory[index];
    if (!item) return;
    setSummary(item.summary);
    setChunks(item.chunks_used);
    setJdText(item.query);
    setCurrentHistoryIndex(index);
  };

  const clearHistory = () => {
    setChatHistory([]);
    setCurrentHistoryIndex(null);
  };

  const handleExport = () => {
    let exportData = {
      analysis_type: activeTab,
      job_description: jdText,
    };

    if (activeTab === "analyze" && summary) {
      exportData.summary = summary;
      exportData.chunks_used = chunks;
    } else if (activeTab === "assess" && assessmentResult) {
      exportData.assessment = assessmentResult.assessment;
      exportData.chunks_used = assessmentResult.chunks_used;
    } else {
      alert("No analysis available to export.");
      return;
    }

    fetch("http://127.0.0.1:5000/export", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(exportData),
    })
      .then((res) => res.blob())
      .then((blob) => {
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `resume_${activeTab}_${Date.now()}.json`;
        document.body.appendChild(a);
        a.click();
        a.remove();
      });
  };

  const clearCurrentResults = () => {
    setSummary(null);
    setChunks([]);
    setAssessmentResult(null);
  };

  return (
    <div className="App">
      <header className="app-header">
        <h1>RAG to Riches</h1>
        <p className="app-subtitle">Resume–JD Analyzer</p>
        <p className="app-tagline">
          Accelerate your career with AI-powered resume analysis
        </p>
      </header>

      <div className="dark-mode-toggle">
        <label className="switch">
          <input
            type="checkbox"
            checked={darkMode}
            onChange={() => setDarkMode(!darkMode)}
          />
          <span className="slider round"></span>
        </label>
        <span>{darkMode ? "Dark Mode" : "Light Mode"}</span>
      </div>

      <div style={{ marginBottom: "15px" }}>
        <label>
          <input type="checkbox" checked={isFollowUp} onChange={toggleFollowUp} />
          &nbsp; Enable Multi-turn Follow-up
        </label>
      </div>

      <section className="upload-section">
        <h3>Upload Resume (PDF)</h3>
        <div className="upload-controls">
          <input
            type="file"
            accept="application/pdf"
            multiple
            onChange={handleFileChange}
          />
          <button onClick={handleUpload} disabled={uploadLoading}>
            {uploadLoading ? "Uploading..." : "Upload"}
          </button>
        </div>
        {uploadMessage && <p className="upload-message">{uploadMessage}</p>}
      </section>

      <section className="jd-input-section">
        <textarea
          placeholder={
            isFollowUp
              ? "Enter your follow-up question here..."
              : "Paste job description here..."
          }
          rows={6}
          value={jdText}
          onChange={(e) => setJdText(e.target.value)}
        />
      </section>

      {/* Tab Navigation */}
      <section className="analysis-tabs">
        <div className="tab-buttons">
          <button
            className={activeTab === "analyze" ? "tab-active" : "tab-inactive"}
            onClick={() => {
              setActiveTab("analyze");
              clearCurrentResults();
            }}
          >
            📊 Analyze Fit
          </button>
          <button
            className={activeTab === "assess" ? "tab-active" : "tab-inactive"}
            onClick={() => {
              setActiveTab("assess");
              clearCurrentResults();
            }}
          >
            🔍 Assess Resume
          </button>
        </div>

        {/* Analyze Tab Content */}
        {activeTab === "analyze" && (
          <div className="tab-content">
            <div className="action-buttons">
              <button
                onClick={handleAnalyze}
                disabled={analyzeLoading}
                className="primary-btn"
              >
                {analyzeLoading ? "Analyzing..." : "Analyze Fit"}
              </button>
              <button onClick={handleExport} disabled={!summary} className="export-btn">
                Export Analysis
              </button>
            </div>
          </div>
        )}

        {/* Assess Tab Content */}
        {activeTab === "assess" && (
          <div className="tab-content">
            {uploadedFiles.length === 0 ? (
              <p className="warning-message">
                Please upload at least 1 resume to use assessment feature.
              </p>
            ) : (
              <>
                <p>
                  <strong>Resume to Assess:</strong> {uploadedFiles[uploadedFiles.length - 1]}
                </p>
                <div className="action-buttons">
                  <button
                    onClick={handleAssessResume}
                    disabled={assessLoading}
                    className="primary-btn"
                  >
                    {assessLoading ? "Assessing..." : "Assess Resume"}
                  </button>
                  <button
                    onClick={handleExport}
                    disabled={!assessmentResult}
                    className="export-btn"
                  >
                    Export Assessment
                  </button>
                </div>
              </>
            )}
          </div>
        )}
      </section>

      {/* Results Display */}
      {activeTab === "analyze" && summary && (
        <section className="fit-summary">
          <h3>Fit Summary:</h3>
          <div className="summary-card">
            <div className="summary-row">
              <strong>Verdict:</strong>
              <span
                className={`verdict-badge ${
                  summary?.Verdict?.includes("Fit") ? "good" : "bad"
                }`}
              >
                {summary?.Verdict || "N/A"}
              </span>
            </div>
            <div className="summary-row">
              <strong>Score:</strong>
              <span className="score-badge">{summary?.Score || "-"} / 10</span>
            </div>
            <div className="summary-row">
              <strong>Relevant Skills:</strong>
              <div className="skills-list">
                {(summary["Relevant Skills"] || []).map((skill, idx) => (
                  <span key={idx} className="skill-badge">
                    {skill}
                  </span>
                ))}
              </div>
            </div>
            <div className="summary-row details-row">
              <strong>Details:</strong>
              <p>{summary?.Summary || "No detailed summary available."}</p>
            </div>
          </div>
        </section>
      )}

      {activeTab === "assess" && assessmentResult && (
        <section className="assessment-results">
          <h3>Resume Assessment Results:</h3>
          <div className="assessment-card">
            <div className="overall-assessment">
              <h4>📋 Overall Assessment</h4>
              <div className="score-display">
                Overall Score:{" "}
                <span className="score-badge">
                  {assessmentResult.assessment["Overall Score"]}/10
                </span>
              </div>
              <p>{assessmentResult.assessment["Overall Assessment"]}</p>
            </div>

            <div className="assessment-details">
              <div className="assessment-section">
                <h4>✅ Strengths</h4>
                <ul>
                  {assessmentResult.assessment.Strengths.map((strength, idx) => (
                    <li key={idx}>{strength}</li>
                  ))}
                </ul>
              </div>

              <div className="assessment-section">
                <h4>❌ Weaknesses</h4>
                <ul>
                  {assessmentResult.assessment.Weaknesses.map((weakness, idx) => (
                    <li key={idx}>{weakness}</li>
                  ))}
                </ul>
              </div>

              <div className="assessment-section">
                <h4>🔧 Areas for Improvement</h4>
                <ul>
                  {assessmentResult.assessment["Areas for Improvement"].map(
                    (area, idx) => (
                      <li key={idx}>{area}</li>
                    )
                  )}
                </ul>
              </div>

              <div className="assessment-section">
                <h4>🎯 Missing Skills</h4>
                <ul>
                  {assessmentResult.assessment["Missing Skills"].map(
                    (skill, idx) => (
                      <li key={idx}>{skill}</li>
                    )
                  )}
                </ul>
              </div>

              <div className="assessment-section">
                <h4>💡 Recommendations</h4>
                <ul>
                  {assessmentResult.assessment.Recommendations.map((rec, idx) => (
                    <li key={idx}>{rec}</li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        </section>
      )}

      {activeTab === "analyze" && chunks.length > 0 && (
        <section className="matched-snippets">
          <h3>Matched Resume Snippets:</h3>
          {chunks.map((chunk, idx) => (
            <details key={idx} className="snippet-card">
              <summary>
                {chunk.source} — Score: {chunk.score.toFixed(3)}
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    navigator.clipboard.writeText(chunk.preview);
                    alert("Snippet copied to clipboard!");
                  }}
                  style={{
                    marginLeft: "10px",
                    fontSize: "12px",
                    cursor: "pointer",
                    color: "#0d6efd",
                    background: "none",
                    border: "none",
                  }}
                >
                  Copy
                </button>
              </summary>
              <p>{chunk.preview}</p>
            </details>
          ))}
        </section>
      )}

      {chatHistory.length > 0 && (
        <section className="chat-history">
          <h3>Previous Queries</h3>
          <button onClick={clearHistory} style={{ marginBottom: "10px" }}>
            Clear History
          </button>
          <ul
            style={{
              listStyle: "none",
              paddingLeft: 0,
              maxHeight: "180px",
              overflowY: "auto",
            }}
          >
            {chatHistory.map((item, idx) => (
              <li
                key={idx}
                onClick={() => loadHistoryItem(idx)}
                style={{
                  cursor: "pointer",
                  padding: "8px",
                  borderBottom: "1px solid #ccc",
                  backgroundColor: currentHistoryIndex === idx ? "#dbeafe" : "#f9f9f9",
                  marginBottom: "4px",
                  borderRadius: "4px",
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                }}
                title={new Date(item.timestamp).toLocaleString()}
              >
                <div style={{ flex: 1 }}>
                  <strong>Query:</strong>{" "}
                  {item.query.length > 80 ? item.query.slice(0, 80) + "..." : item.query}
                  <br />
                  <small style={{ fontSize: "0.85rem", color: "#555" }}>
                    Verdict: {item.summary?.Verdict || "N/A"} | PDFs:{" "}
                    {item.chunks_used
                      ?.map((c) => c.source)
                      .filter((v, i, a) => a.indexOf(v) === i)
                      .join(", ")}
                    <br />
                    {new Date(item.timestamp).toLocaleString()}
                  </small>
                </div>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    navigator.clipboard.writeText(item.query);
                    alert("Query copied to clipboard!");
                  }}
                  style={{
                    marginLeft: "10px",
                    fontSize: "12px",
                    cursor: "pointer",
                    color: "#0d6efd",
                    background: "none",
                    border: "none",
                  }}
                >
                  Copy Query
                </button>
              </li>
            ))}
          </ul>
        </section>
      )}
    </div>
  );
}

export default App;
