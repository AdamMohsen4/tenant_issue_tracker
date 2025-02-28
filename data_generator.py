# data_generator.py
import csv
import random
from datetime import datetime, timedelta

# Categories and their associated terms
categories = {
    "plumbing": [
        "toilet", "sink", "faucet", "drain", "leak", "water", "pipe", "clog", "shower", 
        "bathtub", "dripping", "overflow", "running", "pressure", "hot water", "cold water"
    ],
    "electrical": [
        "light", "switch", "outlet", "power", "circuit", "breaker", "bulb", "fixture", 
        "fan", "electricity", "socket", "plug", "wiring", "lamp", "flickering"
    ],
    "hvac": [
        "heat", "heating", "ac", "air conditioning", "furnace", "thermostat", "temperature", 
        "cold", "hot", "ventilation", "filter", "duct", "radiator", "heater"
    ],
    "structural": [
        "wall", "ceiling", "floor", "door", "window", "roof", "crack", "damage", "hole", 
        "tile", "wood", "cabinet", "counter", "foundation", "paint", "broken"
    ],
    "appliance": [
        "refrigerator", "stove", "oven", "dishwasher", "microwave", "washer", "dryer", 
        "disposal", "range", "freezer", "exhaust", "hood", "garbage disposal"
    ],
    "pest": [
        "bug", "insect", "rodent", "mouse", "rat", "cockroach", "ant", "spider", 
        "termite", "bed bug", "pest", "infestation", "droppings"
    ]
}

# Urgency terms
urgent_terms = [
    "leak", "flood", "broken", "emergency", "dangerous", "hazard", "unsafe", "sparking",
    "fire", "smoke", "gas", "smell", "urgent", "immediately", "asap", "serious", "not working",
    "cannot", "unable", "health", "safety", "risk", "major", "severe", "extreme"
]

# Issue templates
templates = [
    "The {item} is {problem}",
    "My {item} {problem}",
    "{problem} with the {item}",
    "{item} has been {problem} for {time}",
    "There's a {problem} {item}",
    "Can someone fix the {problem} {item}?",
    "Need help with {problem} {item}",
    "{item} {problem} and needs repair",
    "Issue with {item}: {problem}",
    "The {item} in my {location} is {problem}"
]

# Problem descriptions
problems = [
    "leaking", "broken", "not working", "making strange noises", "damaged",
    "clogged", "overflowing", "too hot", "too cold", "flickering",
    "stopped working", "won't turn on", "won't turn off", "dripping", "cracked",
    "loose", "stuck", "malfunctioning", "smells bad", "dirty"
]

# Locations
locations = ["bathroom", "kitchen", "bedroom", "living room", "hallway", "closet", "basement"]

# Time descriptions
times = ["a day", "a few days", "a week", "several days", "2 days", "3 days", "yesterday"]

def generate_mock_issue():
    # Pick a random category and associated item
    category = random.choice(list(categories.keys()))
    item = random.choice(categories[category])
    
    # Determine if this should be urgent (30% chance)
    is_urgent = random.random() < 0.3
    
    # If urgent, possibly include an urgent term
    problem = random.choice(problems)
    if is_urgent and random.random() < 0.7:
        urgent_term = random.choice(urgent_terms)
        problem = f"{problem} {urgent_term}"
    
    # Generate the issue text
    template = random.choice(templates)
    text = template.format(
        item=item,
        problem=problem,
        location=random.choice(locations) if "{location}" in template else "",
        time=random.choice(times) if "{time}" in template else ""
    )
    
    # Determine urgency
    urgency = "urgent" if is_urgent else "non-urgent"
    
    # Generate a random timestamp within the last month
    days_ago = random.randint(0, 30)
    timestamp = (datetime.now() - timedelta(days=days_ago)).strftime('%Y-%m-%d %H:%M:%S')
    
    return {
        "text": text,
        "category": category,
        "urgency": urgency,
        "timestamp": timestamp,
        "has_photo": random.random() < 0.5  # 50% chance of having a photo
    }

# Generate the dataset
dataset_size = 100
with open('mock_issues.csv', 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=["text", "category", "urgency", "timestamp", "has_photo"])
    writer.writeheader()
    for _ in range(dataset_size):
        writer.writerow(generate_mock_issue())

print(f"Generated {dataset_size} mock issues and saved to mock_issues.csv")