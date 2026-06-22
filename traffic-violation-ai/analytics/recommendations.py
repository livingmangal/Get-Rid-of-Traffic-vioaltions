import pandas as pd
from datetime import datetime
from config.settings import DATA_DIR

class RecommendationEngine:
    def __init__(self):
        self.reports = {
            "Helmet": DATA_DIR / "helmet_report.csv",
            "Triple Riding": DATA_DIR / "triple_riding_report.csv",
            "Wrong Side": DATA_DIR / "wrong_side_report.csv",
            "Pedestrian": DATA_DIR / "pedestrian_report.csv",
            "Waterlogging": DATA_DIR / "waterlogging_report.csv",
            "License Plate": DATA_DIR / "license_plate_ocr_improved_report.csv"
        }
        self.output_file = DATA_DIR / "ai_recommendations.txt"

    def generate(self):
        recommendations = []
        for name, path in self.reports.items():
            if path.exists():
                try:
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
                except Exception:
                    pass

        if not recommendations:
            recommendations.append("No major risk detected. Continue monitoring traffic feed.")

        print("\n==============================")
        print("AI TRAFFIC POLICE ASSISTANT")
        print("==============================")
        print("Time:", datetime.now())
        print("\nRecommendations:\n")
        
        for i, rec in enumerate(recommendations, start=1):
            print(f"{i}. {rec}")

        with open(self.output_file, "w") as file:
            file.write("AI TRAFFIC POLICE ASSISTANT\n")
            file.write("===========================\n")
            file.write(f"Time: {datetime.now()}\n\n")
            for i, rec in enumerate(recommendations, start=1):
                file.write(f"{i}. {rec}\n")
                
        print(f"\nSaved to {self.output_file.name}")

if __name__ == "__main__":
    engine = RecommendationEngine()
    engine.generate()
