"""
Agentic Fallback Layer — LangChain Agent with SQLite Historical Defect DB.

This module implements the "expensive" reasoning layer that activates ONLY
when the IsolationForest flags an anomaly. It:
    1. Queries a local SQLite database for historical defect analogies.
    2. Uses an LLM (ChatOpenAI) as a Quality Assurance Specialist to generate
       a structured rejection report with severity scoring and mitigation advice.

Design Rationale:
    - SQLite is zero-overhead: no server process, no network, single .db file.
    - The agent is never instantiated for "Normal" frames, saving compute.
    - Output is strictly validated JSON for downstream automated systems.
"""

import os
import json
import sqlite3

from src.config import settings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage



def _initialize_database() -> None:
    """
    Create the SQLite database and seed it with mock historical defect records.
    This runs once on first use. The database file is placed under data/.
    """
    db_path = settings.db_path
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create the defect history table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS defect_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            defect_signature TEXT NOT NULL,
            description TEXT NOT NULL,
            quarter TEXT NOT NULL,
            severity REAL NOT NULL,
            resolution TEXT NOT NULL
        )
    """)

    # Seed with mock data only if table is empty
    cursor.execute("SELECT COUNT(*) FROM defect_history")
    if cursor.fetchone()[0] == 0:
        mock_records = [
            (
                "surface_scratch",
                "Conveyor belt friction mismatch causing micro-abrasions on part surface.",
                "Q3 2024",
                0.82,
                "Replaced conveyor belt pads and recalibrated belt tension."
            ),
            (
                "dimensional_drift",
                "Thermal expansion in stamping press causing 0.3mm dimensional overshoot.",
                "Q1 2024",
                0.91,
                "Installed thermal compensation sensor on press hydraulics."
            ),
            (
                "color_deviation",
                "Paint viscosity fluctuation due to ambient temperature drop below 18°C.",
                "Q4 2023",
                0.67,
                "Added paint reservoir heater with PID temperature control."
            ),
            (
                "weld_porosity",
                "Gas shielding inconsistency in MIG welding station causing micro-porosity.",
                "Q2 2024",
                0.95,
                "Replaced gas flow regulator and added real-time flow monitoring."
            ),
            (
                "alignment_offset",
                "Robotic arm calibration drift after 50K cycle maintenance interval.",
                "Q3 2024",
                0.78,
                "Reduced calibration interval to 30K cycles; added encoder verification."
            ),
            (
                "edge_burr",
                "Cutting tool wear causing excessive burr formation on stamped edges.",
                "Q1 2025",
                0.73,
                "Implemented tool wear prediction model and preemptive replacement schedule."
            ),
        ]
        cursor.executemany(
            "INSERT INTO defect_history (defect_signature, description, quarter, severity, resolution) VALUES (?, ?, ?, ?, ?)",
            mock_records
        )
        conn.commit()

    conn.close()


def query_historical_defects(defect_signature: str) -> str:
    """
    Query the local SQLite database for historical defect analogies.

    This function performs a fuzzy LIKE search against the defect_history table,
    returning a human-readable summary of past similar defects.

    Args:
        defect_signature: A keyword or phrase describing the detected anomaly
                          (e.g., "surface_scratch", "alignment").

    Returns:
        str: A formatted string describing matching historical defects,
             or a fallback message if no matches are found.
    """
    # Ensure DB exists and is seeded
    _initialize_database()

    db_path = settings.db_path
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Fuzzy search across signature and description fields
    search_term = f"%{defect_signature}%"
    cursor.execute("""
        SELECT defect_signature, description, quarter, severity, resolution
        FROM defect_history
        WHERE defect_signature LIKE ? OR description LIKE ?
        ORDER BY severity DESC
        LIMIT 3
    """, (search_term, search_term))

    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return (
            f"No exact historical match for '{defect_signature}'. "
            "This may be a novel defect type requiring manual inspection and new baseline."
        )

    # Format results as readable text for the LLM
    results = []
    for sig, desc, quarter, severity, resolution in rows:
        results.append(
            f"- [{quarter}] {sig} (severity: {severity:.2f}): {desc} "
            f"Resolution: {resolution}"
        )

    return "Historical defect matches found:\n" + "\n".join(results)


def run_agent(image_path: str, anomaly_score: float) -> dict:
    """
    Activate the LLM agent to generate a structured rejection report.

    This function should ONLY be called when the AnomalyDetector has flagged
    a frame as anomalous. It:
        1. Queries the SQLite history for analogous defects.
        2. Sends context to ChatOpenAI with a strict QA Specialist prompt.
        3. Parses the LLM response into validated JSON.

    Args:
        image_path: Path to the anomalous image (for context in the report).
        anomaly_score: The raw anomaly score from IsolationForest.

    Returns:
        dict: Structured report with keys:
              - defect_confirmed (bool)
              - severity_score (float)
              - historical_analogy (str)
              - recommended_action (str)
    """
    # Step 1: Query historical defects using the anomaly context
    historical_context = query_historical_defects("anomaly")

    # Step 2: Initialize the LLM using settings (no raw os.getenv)
    llm = ChatGroq(
        model=settings.groq_model,
        temperature=settings.llm_temperature,
        groq_api_key=settings.groq_api_key,  # type: ignore[arg-type]
    )

    # Step 3: Construct the strict system prompt
    system_prompt = """You are an expert Quality Assurance Specialist for an industrial manufacturing assembly line.
You are analyzing a part that has been flagged as anomalous by an automated vision system.

Based on the anomaly score and historical defect data provided, you MUST respond with ONLY a valid JSON object
(no markdown, no explanation, no code blocks) with exactly these four keys:

{
    "defect_confirmed": true or false,
    "severity_score": a float between 0.0 and 1.0,
    "historical_analogy": "A brief description of the most similar past defect",
    "recommended_action": "A specific, actionable mitigation step"
}

Rules:
- If the anomaly score magnitude exceeds 0.3, confirm the defect.
- Severity score should reflect both the anomaly score magnitude and historical precedent.
- The historical analogy must reference the provided database records if available.
- The recommended action must be specific and immediately actionable on the assembly line.
- Respond with ONLY the JSON object. No other text."""

    # Step 4: Build the human message with full context
    human_message = f"""Anomaly Detection Report:
- Image path: {image_path}
- Anomaly score: {anomaly_score:.4f} (more negative = more anomalous)
- Detection method: IsolationForest with contamination=0.05

Historical Defect Database Query Results:
{historical_context}

Generate the structured rejection report as JSON."""

    # Step 5: Invoke the LLM
    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_message),
        ])

        # Parse the LLM response as JSON (str cast avoids Pylance union type error for list/dict content)
        report = json.loads(str(response.content).strip())

        # Validate required keys exist
        required_keys = {"defect_confirmed", "severity_score", "historical_analogy", "recommended_action"}
        if not required_keys.issubset(report.keys()):
            raise ValueError(f"Missing keys in LLM response: {required_keys - report.keys()}")

        return report

    except (json.JSONDecodeError, ValueError) as e:
        # Fallback: return a conservative default report if LLM output is malformed
        return {
            "defect_confirmed": True,
            "severity_score": min(abs(anomaly_score), 1.0),
            "historical_analogy": historical_context.split('\n')[0] if historical_context else "No historical data available.",
            "recommended_action": "Manual inspection required — LLM parsing failed. Halt line for safety.",
            "_error": str(e)
        }

    except Exception as e:
        # Handle API/network errors gracefully
        return {
            "defect_confirmed": True,
            "severity_score": min(abs(anomaly_score), 1.0),
            "historical_analogy": "Unable to query LLM — using conservative defaults.",
            "recommended_action": "Halt assembly line and escalate to shift supervisor for manual review.",
            "_error": f"LLM API error: {str(e)}"
        }
