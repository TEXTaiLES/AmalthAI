import os
import re
from utils.user_paths import user_root

def _parse_classification_output(output_text):
    """Extract the classifier's selected class and probability table."""
    predicted_class = ""
    confidence = ""
    probabilities = []

    for line in (output_text or "").splitlines():
        line = line.strip()
        if not line:
            continue

        predicted_match = re.match(
            r"Predicted Class:\s*(.*?)\s*(?:\|\s*Confidence:\s*(.*))?$",
            line,
            re.IGNORECASE,
        )
        if predicted_match:
            predicted_class = predicted_match.group(1).strip()
            confidence = (predicted_match.group(2) or "").strip()
            continue

        probability_match = re.match(
            r"([^:]+):\s*.*?\(([0-9]+(?:\.[0-9]+)?)%\)", line
        )
        if probability_match:
            probabilities.append({
                "name": probability_match.group(1).strip(),
                "value": f"{probability_match.group(2)}%",
            })

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": probabilities,
    }

def _inference_handoff_path(user_slug, relative_path):
    """Resolve a session-provided inference file without allowing path escape."""
    inference_dir = os.path.abspath(os.path.join(user_root(user_slug), "inference"))
    path = os.path.abspath(os.path.join(inference_dir, relative_path))
    if not path.startswith(inference_dir + os.sep):
        return None
    return path