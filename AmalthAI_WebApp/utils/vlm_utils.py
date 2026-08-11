SYSTEM_PROMPT_TEMPLATE = """
You are a visual analysis assistant helping cultural heritage experts understand {blank1}.
The images show {blank2}.
 
You will receive TWO images:
1. Image 1 (raw): {blank3}.
2. Image 2 (Grad-CAM): a heatmap overlay showing where the classifier focused.
 
"Rules:\\n"
    "- visible_description: describe ONLY what you see in Image 1. "
    "Do NOT mention the heatmap or Grad-CAM here.\\n"
    "- attention_description: describe WHERE the heat in Image 2 is spatially "
    "(center / edge / background / specific feature). "
    "State whether it overlaps {blank4}. Do NOT mention colors.\\n"
    "- Do NOT name any {blank5}\\n"
    "- Do NOT say the classifier was right or wrong.\\n"
    "- Do NOT describe {blank6}.\\n"
    "- Do NOT use information {blank7}.\\n"
    "Return ONLY valid JSON, 1-2 sentences per field:\\n"
    '{{
  "visible_description": "",
  "attention_description": "",
  "task_properties": {{
    "task": "",
    "ground_truth": "",
    "predicted_class": "",
    "confidence": "",
    "class_probabilities": {{}}
  }}
}}'
"""
 
SYSTEM_PROMPT_BLANKS = [
    ("blank1", "microscope images from an experimental archaeology study"),
    ("blank2", "clay imprints made by pressing textile samples (fibres and threads) into clay under controlled conditions"),
    ("blank3", "the actual microscope photograph of the clay imprint"),
    ("blank4", "the textile imprint"),
    ("blank5", "class, technique, or material type (no Drilling, Spinning, Nettle, Wool, etc.)"),
    ("blank6", "the material or the background as concrete, stone, or fossil"),
    ("blank7", "about classes, techniques, or materials to inform your descriptions"),
]
 
USER_PROMPT_TEMPLATE = """Task: {task}
The attached images are:
1. The original Cultural Heritage image.
2. The corresponding Grad-CAM visualization.
Known classification results (do NOT predict or re-derive them):
Ground-truth class: {ground_truth}
Predicted class: {predicted_class}
Prediction confidence: {confidence}
Class probabilities:
{class_probabilities}
Using the original image and the Grad-CAM, explain why the classifier may have predicted the predicted class instead of the ground-truth class.
Base your explanation only on visible evidence in the images.
Use the class probabilities only as supporting information.
Return only the JSON specified in the system prompt."""
 
USER_PROMPT_FIELDS = [
    ("task", "technique classification"),
    ("ground_truth", "Drilling"),
    ("predicted_class", "Splicing"),
    ("confidence", "0.3488"),
]
 
USER_PROMPT_CLASS_EXAMPLES = [
    ("Drilling", "31.39%"),
    ("Spinning", "33.73%"),
    ("Splicing", "34.88%"),
]
 

def build_user_prompt(form):
    fields = {key: form.get(key, '') for key, _ in USER_PROMPT_FIELDS}
    class_names = form.getlist('class_name')
    class_values = form.getlist('class_value')
    class_lines = [
        f"{name}: {value}"
        for name, value in zip(class_names, class_values)
        if name.strip() or value.strip()
    ]
    fields['class_probabilities'] = "\n".join(class_lines)
    return USER_PROMPT_TEMPLATE.format(**fields)