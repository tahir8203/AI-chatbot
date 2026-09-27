from fastapi.testclient import TestClient

import app as chatbot

client = TestClient(chatbot.app)


def test_health_runs_in_demo_mode():
    data = client.get("/api/health").json()
    assert data["status"] == "ok"
    assert data["mode"] == "demo"
    assert data["chunks"] > 5


def test_attendance_question_finds_attendance_rule():
    data = client.post("/api/ask", json={"question": "What is the attendance requirement?"}).json()
    assert data["sources"][0]["section"] == "Attendance rule"
    assert "75%" in data["answer"]


def test_installments_question_finds_payment_section():
    data = client.post("/api/ask", json={"question": "Can I pay the fee in installments?"}).json()
    assert data["sources"][0]["section"] == "Payment and installments"


def test_merit_scholarship_question():
    data = client.post("/api/ask", json={"question": "How do merit scholarships work?"}).json()
    assert data["sources"][0]["section"] == "Merit scholarships"


def test_unrelated_question_has_no_sources():
    data = client.post("/api/ask", json={"question": "xyzzy quantum banana"}).json()
    assert data["sources"] == []


def test_rejects_empty_question():
    assert client.post("/api/ask", json={"question": ""}).status_code == 422
