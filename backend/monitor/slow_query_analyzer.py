import torch
import torch.nn as nn
import re

# -------------------------------
# 1️⃣ Define the same model structure
# -------------------------------
class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=1):
        super(SimpleNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.network(x)

# -------------------------------
# 2️⃣ Load trained weights
# -------------------------------
# MODEL_PATH = "monitor/trained_model.pth"
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "trained_model.pth")
INPUT_DIM = 13  # same as your training feature size
model = SimpleNN(INPUT_DIM)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# -------------------------------
# 3️⃣ Feature extraction for queries
# -------------------------------
def extract_features(query: str):
    """
    Convert SQL query into numeric features for the model.
    Example features (length 13):
      - query length
      - number of joins
      - number of subqueries
      - count of WHERE/AND/OR
      - number of SELECT columns
      - number of tables
      - presence of *, DISTINCT, GROUP BY, ORDER BY
      - etc.
    """
    q = query.lower()
    features = [
        len(q),  # total length
        q.count("join"),
        q.count("select"),
        q.count("where"),
        q.count("and"),
        q.count("or"),
        q.count("from"),
        q.count("*"),
        int("distinct" in q),
        int("group by" in q),
        int("order by" in q),
        q.count("("),  # subqueries
        q.count(")")   # subqueries
    ]
    return torch.tensor(features).float().unsqueeze(0)  # shape [1,13]

# -------------------------------
# 4️⃣ Root cause ranking
# -------------------------------
def analyze_root_causes(query: str):
    """
    Returns a ranked list of root causes contributing to query slowness.
    Combines ML model prediction and rule-based heuristics.
    """
    # Extract features and predict
    features = extract_features(query)
    with torch.no_grad():
        score = torch.sigmoid(model(features))[0][0].item()  # probability between 0-1

    # Rule-based hints
    causes = []
    if re.search(r"join", query, re.I):
        causes.append({"cause": "large_joins", "score": 0.8})
    if re.search(r"select\s+\*", query, re.I):
        causes.append({"cause": "full_table_scan", "score": 0.9})
    if re.search(r"where", query, re.I) is None:
        causes.append({"cause": "missing_where", "score": 0.85})
    if re.search(r"\(", query, re.I):
        causes.append({"cause": "subqueries", "score": 0.7})

    # ML-predicted cause
    causes.append({"cause": "ml_predicted_slow", "score": score})

    # Sort descending by score
    ranked = sorted(causes, key=lambda x: x["score"], reverse=True)
    return ranked
