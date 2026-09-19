#!/usr/bin/env python3
"""
Test script for Multilingual Audio Call Recording & Transcription Verification
"""
import sys
from app import app, SUPPORTED_AUDIO_LANGUAGES, DEMO_CALL_RECORDINGS

def run_tests():
    client = app.test_client()
    
    print("1. Testing GET /api/audio-languages...")
    res = client.get("/api/audio-languages")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    data = res.get_json()
    assert data["status"] == "success"
    assert len(data["languages"]) >= 10, f"Expected >= 10 languages, got {len(data['languages'])}"
    print(f"   ✓ Returned {len(data['languages'])} languages: {[l['code'] for l in data['languages']]}")

    print("\n2. Testing GET /api/demo-calls...")
    res = client.get("/api/demo-calls")
    assert res.status_code == 200
    demo_calls = res.get_json()["demo_calls"]
    demo_ids = [d["id"] for d in demo_calls]
    assert "threat" in demo_ids
    assert "threat_hindi" in demo_ids
    assert "threat_tamil" in demo_ids
    assert "advance_fee" in demo_ids
    assert "legitimate" in demo_ids
    print(f"   ✓ All 5 demo calls present: {demo_ids}")

    print("\n3. Testing POST /api/transcribe-audio (Hindi demo preset)...")
    res = client.post("/api/transcribe-audio", json={"demo_id": "threat_hindi", "language": "hi-IN"})
    assert res.status_code == 200
    t_hindi = res.get_json()
    assert t_hindi["status"] == "success"
    assert "RupyaFast" in t_hindi["transcript"]
    assert len(t_hindi["dialogue"]) >= 3
    print("   ✓ Hindi transcription returned correctly with dialogue turns.")

    print("\n4. Testing POST /api/transcribe-audio (Tamil demo preset)...")
    res = client.post("/api/transcribe-audio", json={"demo_id": "threat_tamil", "language": "ta-IN"})
    assert res.status_code == 200
    t_tamil = res.get_json()
    assert t_tamil["status"] == "success"
    assert "InstantPanam" in t_tamil["transcript"]
    print("   ✓ Tamil transcription returned correctly with dialogue turns.")

    print("\n5. Testing POST /api/analyze-audio (Hindi call analysis)...")
    res = client.post("/api/analyze-audio", json={"demo_id": "threat_hindi", "call_time": "22:15", "language": "hi-IN"})
    assert res.status_code == 200
    analysis_hi = res.get_json()
    assert analysis_hi["risk_level"] == "HIGH"
    assert analysis_hi["risk_score"] >= 80
    assert analysis_hi["violation_count"] >= 3, f"Expected >=3 violations, got {analysis_hi['violation_count']}"
    v_ids_hi = [v["id"] for v in analysis_hi["rbi_recovery_violations"]]
    print(f"   ✓ Hindi Risk Level: {analysis_hi['risk_level']} ({analysis_hi['risk_score']}/100)")
    print(f"   ✓ Violations Detected ({len(v_ids_hi)}): {v_ids_hi}")

    print("\n6. Testing POST /api/analyze-audio (Tamil call analysis)...")
    res = client.post("/api/analyze-audio", json={"demo_id": "threat_tamil", "call_time": "21:30", "language": "ta-IN"})
    assert res.status_code == 200
    analysis_ta = res.get_json()
    assert analysis_ta["risk_level"] == "HIGH"
    assert analysis_ta["risk_score"] >= 80
    assert analysis_ta["violation_count"] >= 3, f"Expected >=3 violations, got {analysis_ta['violation_count']}"
    v_ids_ta = [v["id"] for v in analysis_ta["rbi_recovery_violations"]]
    print(f"   ✓ Tamil Risk Level: {analysis_ta['risk_level']} ({analysis_ta['risk_score']}/100)")
    print(f"   ✓ Violations Detected ({len(v_ids_ta)}): {v_ids_ta}")

    print("\n7. Testing POST /api/analyze-audio (Telugu custom call)...")
    telugu_transcript = (
        "Agent: Mee intiki gundalani pampistha, intiki vachi gola chestha. "
        "Mee contacts andariki morphed photo viral chestham! "
        "Ee UPI id ki ventane 5000 Google Pay cheyandi, lekapothe police station nundi arrest warrant vasthundi!"
    )
    res = client.post("/api/analyze-audio", json={"transcript": telugu_transcript, "call_time": "22:00", "language": "te-IN"})
    assert res.status_code == 200
    analysis_te = res.get_json()
    assert analysis_te["risk_level"] == "HIGH"
    assert analysis_te["violation_count"] >= 3
    print(f"   ✓ Telugu Risk Level: {analysis_te['risk_level']} ({analysis_te['risk_score']}/100)")
    print(f"   ✓ Violations Detected ({analysis_te['violation_count']}): {[v['id'] for v in analysis_te['rbi_recovery_violations']]}")

    print("\n8. Testing POST /api/analyze-audio (Marathi custom call)...")
    marathi_transcript = (
        "Agent: Ghari gunde pathvin, aai vadilana call karen! "
        "Photo morph karun badnaam karen. Lagach hya UPI id var Google Pay kara!"
    )
    res = client.post("/api/analyze-audio", json={"transcript": marathi_transcript, "call_time": "21:15", "language": "mr-IN"})
    assert res.status_code == 200
    analysis_mr = res.get_json()
    assert analysis_mr["risk_level"] == "HIGH"
    assert analysis_mr["violation_count"] >= 2
    print(f"   ✓ Marathi Risk Level: {analysis_mr['risk_level']} ({analysis_mr['risk_score']}/100)")
    print(f"   ✓ Violations Detected ({analysis_mr['violation_count']}): {[v['id'] for v in analysis_mr['rbi_recovery_violations']]}")

    print("\n✅ ALL MULTILINGUAL AUDIO TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    run_tests()

