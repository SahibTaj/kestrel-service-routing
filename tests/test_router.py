from router import predict_record


def test_predict_returns_canonical_team():
    result = predict_record({"request_id":"test-1", "channel":"chat", "product_family":"Mixer Grinder",
        "warranty_status":"in_warranty", "request_text":"Need installation for my new mixer grinder", "source":"crm"})
    assert result["team"] in {"Installations", "Repairs", "Consumables", "Billing", "Returns & Replacement", "Warranty Claims", "Product Advice"}
    assert isinstance(result["reasons"], list) and result["reasons"]


def test_empty_text_fails_clearly():
    result = predict_record({"request_text":"   "})
    assert result["code"] == "INVALID_INPUT"


def test_fault_with_consumable_cue_flags_review():
    result = predict_record({"request_id":"test-3", "channel":"chat", "product_family":"Water Purifier",
        "warranty_status":"in_warranty", "request_text":"Water purifier is leaking from the bottom", "source":"crm"})
    assert result["model_team"] == "Consumables"
    assert result["team"] == "Repairs"
    assert result["policy_override"] is True
    assert any("product fault" in reason.lower() for reason in result["reasons"])
