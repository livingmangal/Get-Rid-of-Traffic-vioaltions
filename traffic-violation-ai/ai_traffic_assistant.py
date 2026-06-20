import os
import pandas as pd
from datetime import datetime

os.makedirs("data", exist_ok=True)

reports = {
    "Helmet": "data/helmet_report.csv",
    "Triple Riding": "data/triple_riding_report.csv",
    "Wrong Side": "data/wrong_side_report.csv",
    "Pedestrian": "data/pedestrian_report.csv",
    "Waterlogging": "data/waterlogging_report.csv",
    "License Plate": "data/license_plate_report.csv",
}

recommendations = []

for name, path in reports.items():
    if os.path.exists(path):
        df = pd.read_csv(path)
        text = df.to_string()

        if "Helmet" in name and "0" not in text:
            recommendations.append("Helmet violations detected: issue warning/e-challan and increase helmet checking.")

        if "Triple Riding" in name and "0" not in text:
            recommendations.append("Triple riding detected: deploy enforcement team and start safety alert.")

        if "Wrong Side" in name and "0" not in text:
            recommendations.append("Wrong-side driving detected: add barricade/signage and deploy officer at hotspot.")

        if "Pedestrian" in name:
            recommendations.append("Pedestrian activity detected: reduce vehicle speed and improve crossing safety.")

        if "Waterlogging" in name:
            recommendations.append("Waterlogging risk detected: reroute vehicles and alert municipal drainage team.")

        if "License Plate" in name:
            recommendations.append("License plates extracted: attach plate data to violation evidence records.")

if not recommendations:
    recommendations.append("No major risk detected. Continue monitoring traffic feed.")

print("\n==============================")
print("AI TRAFFIC POLICE ASSISTANT")
print("==============================")
print("Time:", datetime.now())
print("\nRecommendations:\n")

for i, rec in enumerate(recommendations, start=1):
    print(f"{i}. {rec}")

with open("data/ai_recommendations.txt", "w") as file:
    file.write("AI TRAFFIC POLICE ASSISTANT\n")
    file.write("===========================\n")
    file.write(f"Time: {datetime.now()}\n\n")
    for i, rec in enumerate(recommendations, start=1):
        file.write(f"{i}. {rec}\n")

print("\nSaved to data/ai_recommendations.txt")