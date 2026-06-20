from datetime import datetime

def evidence_id(prefix):
    return f"{prefix}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

def risk_level(count):
    if count == 0:
        return "NORMAL"
    elif count <= 5:
        return "WARNING"
    elif count <= 20:
        return "HIGH"
    else:
        return "CRITICAL"

def confidence_score(count):
    if count == 0:
        return 0.00
    elif count <= 5:
        return 0.72
    elif count <= 20:
        return 0.86
    else:
        return 0.94

def recommendation(violation_type):
    actions = {
        "NO_HELMET": "Issue warning/e-challan and increase helmet checking.",
        "TRIPLE_RIDING": "Deploy enforcement team and start road safety warning.",
        "WRONG_SIDE": "Add barricade/signage and deploy officer at hotspot.",
        "WATERLOGGING": "Reroute traffic and alert municipal drainage team."
    }
    return actions.get(violation_type, "Continue monitoring.")