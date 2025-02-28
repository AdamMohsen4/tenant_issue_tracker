import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import re

def generate_tenant_issue_dataset(num_samples=700, random_seed=42):
    """
    Generate synthetic tenant issue tracker dataset with specified distributions.
    
    Parameters:
    num_samples (int): Number of samples to generate (default: 700)
    random_seed (int): Random seed for reproducibility
    
    Returns:
    pandas.DataFrame: Generated dataset
    """
    # Set random seed for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Define categories and sample counts
    categories = ['structural', 'plumbing', 'hvac', 'pest', 'electrical', 'appliance', 'miscellaneous']
    samples_per_category = num_samples // len(categories)
    
    # Define urgency levels and their distribution within each category
    urgency_levels = ['urgent', 'semi-urgent', 'non-urgent']
    
    # Define date range for timestamps (February 2025)
    start_date = datetime(2025, 2, 1)
    end_date = datetime(2025, 2, 28, 23, 59, 59)
    
    # Define photo probability by urgency
    photo_prob = {
        'urgent': 0.6,       # 60% of urgent issues have photos
        'semi-urgent': 0.5,  # 50% of semi-urgent issues have photos
        'non-urgent': 0.2    # 20% of non-urgent issues have photos
    }
    
    # Sample text templates by category and urgency
    text_templates = {
        'structural': {
            'urgent': [
                "Ceiling {verb} in the {location}",
                "Large crack in {location} wall, {additional}",
                "{location} floor {verb}, {additional}",
                "Part of the {location} ceiling collapsed",
                "Balcony railing is loose and unstable",
                "Window shattered in {location}, {additional}",
                "Door to {location} won't lock, {additional}",
                "Stairs collapsed in {location}",
                "{location} floor has collapsed partially",
                "Major crack in foundation, {additional}"
            ],
            'semi-urgent': [
                "Small crack in {location} wall",
                "Loose tiles in {location} floor",
                "{location} window {verb}",
                "Door to {location} {verb}",
                "Minor ceiling damage in {location}",
                "Some {location} floor boards are warped",
                "Grouting coming out between tiles in {location}",
                "Window in {location} won't close properly",
                "Small hole in {location} wall",
                "Cabinet doors in {location} won't stay closed"
            ],
            'non-urgent': [
                "Paint peeling in {location}",
                "Small chip in {location} countertop",
                "Minor scratch on {location} floor",
                "{location} door squeaks when opening",
                "Slight discoloration on {location} ceiling",
                "Loose doorknob in {location}",
                "Weatherstripping coming off {location} window",
                "Minor dent in {location} wall",
                "Baseboard separating from wall in {location}",
                "Curtain rod loose in {location}"
            ]
        },
        'plumbing': {
            'urgent': [
                "Water {verb} from {location} pipe, {additional}",
                "{location} toilet overflowing onto floor",
                "Main water pipe burst in {location}, {additional}",
                "Sewage backing up in {location}, {additional}",
                "Hot water tank leaking heavily in {location}",
                "Major water leak coming through {location} ceiling",
                "Bathroom pipe burst, water everywhere",
                "Kitchen sink is overflowing onto the floor",
                "Water gushing from under {location} sink",
                "Severe leak from ceiling in {location}, {additional}"
            ],
            'semi-urgent': [
                "{location} sink {verb}",
                "Clogged {location} toilet, {additional}",
                "Slow drain in {location} sink",
                "{location} faucet {verb}",
                "Low water pressure in {location}",
                "Hot water not working in {location}",
                "Minor leak under {location} sink",
                "Toilet in {location} running continuously",
                "Dripping faucet in {location}",
                "{location} shower drain partially clogged"
            ],
            'non-urgent': [
                "Slight drip from {location} faucet",
                "Minor stain under {location} sink pipe",
                "{location} faucet handle loose",
                "Toilet in {location} occasionally runs",
                "Shower head in {location} sprays unevenly",
                "Sink stopper in {location} not working properly",
                "Water pressure fluctuates in {location}",
                "Sink in {location} drains slowly",
                "Toilet seat loose in {location}",
                "Shower curtain rod rusting in {location}"
            ]
        },
        'hvac': {
            'urgent': [
                "No heat in apartment, {additional}",
                "Strong gas smell from {location} heater",
                "AC unit {verb} with electrical smell",
                "Furnace making loud banging noise and not heating",
                "Carbon monoxide detector alarming near {location} heater",
                "Smoke coming from {location} vent",
                "Heat not working and temperature below freezing inside",
                "HVAC system completely failed, {additional}",
                "Strange burning smell from heating system in {location}",
                "AC leaking heavily into electrical outlet in {location}"
            ],
            'semi-urgent': [
                "Heat working intermittently in {location}",
                "AC not cooling {location} properly",
                "Thermostat in {location} {verb}",
                "Heating vent cover fell off in {location}",
                "Air conditioner making strange noise in {location}",
                "Heat blowing cold air in {location}",
                "Weak airflow from vents in {location}",
                "Furnace cycles on and off repeatedly",
                "Unusual smell when heat turns on in {location}",
                "Radiator leaking in {location}"
            ],
            'non-urgent': [
                "Air filter needs replacement in {location}",
                "Slight noise from {location} AC unit",
                "One vent not working in {location}",
                "Minor temperature fluctuation in {location}",
                "Thermostat display flickering in {location}",
                "Heating vent dusty in {location}",
                "AC unit vibrating slightly in {location}",
                "Uneven heating in different rooms",
                "Slight delay when adjusting thermostat",
                "Vent cover loose in {location}"
            ]
        },
        'pest': {
            'urgent': [
                "Large rat infestation in {location}, {additional}",
                "Bed bugs found throughout apartment, {additional}",
                "Wasps nest inside {location} window",
                "Cockroach infestation in {location}, {additional}",
                "Found a snake in {location}",
                "Multiple scorpions spotted in {location}",
                "Swarm of bees entered apartment through {location}",
                "Flea infestation causing severe allergic reactions",
                "Mice chewed through electrical wiring in {location}",
                "Black widow spiders found in {location} with children in home"
            ],
            'semi-urgent': [
                "Several {pest} spotted in {location}",
                "Evidence of {pest} in {location} cabinets",
                "Found {pest} droppings in {location}",
                "Ants in {location}, {additional}",
                "Small wasp nest forming outside {location} window",
                "Recurring spiders in {location}",
                "Flies coming from {location} drain",
                "Silverfish in {location} bathroom",
                "Occasional mouse sightings in {location}",
                "Moth infestation in {location} closet"
            ],
            'non-urgent': [
                "Occasional ant in {location}",
                "Spider webs in {location} corner",
                "Single {pest} spotted in {location}",
                "Few gnats around {location} plants",
                "Small anthill near {location} entrance",
                "Fruit flies in {location}",
                "Occasional cricket heard in {location}",
                "Minor cobwebs in {location} windows",
                "Single cockroach spotted in {location}",
                "Few houseflies in {location}"
            ]
        },
        'electrical': {
            'urgent': [
                "Electrical outlet {verb} in {location}, {additional}",
                "Burning smell from {location} electrical panel",
                "Multiple circuit breakers tripping, {additional}",
                "Exposed wires in {location}, {additional}",
                "Ceiling light fixture sparking in {location}",
                "Power outage in entire unit, {additional}",
                "Shock received from {location} switch",
                "Main electrical panel smoking",
                "Lightning struck building, power surging in outlets",
                "Water leaking onto electrical box in {location}"
            ],
            'semi-urgent': [
                "Light fixture in {location} {verb}",
                "Half of outlets not working in {location}",
                "Circuit breaker for {location} {verb}",
                "Flickering lights in {location}",
                "Buzzing sound from {location} outlet",
                "Dimming lights when appliances run in {location}",
                "Bathroom fan not working, causing humidity issues",
                "GFI outlet in {location} won't reset",
                "Doorbell not functioning",
                "Ceiling fan making grinding noise in {location}"
            ],
            'non-urgent': [
                "Single light bulb out in {location}",
                "One outlet not working in {location}",
                "Light switch plate cracked in {location}",
                "Ceiling fan wobbling slightly in {location}",
                "Dimmer switch not dimming smoothly in {location}",
                "Motion sensor light staying on in {location}",
                "Outlet cover missing in {location}",
                "Light fixture shade cracked in {location}",
                "Bathroom fan making slight noise",
                "Doorbell sound distorted"
            ]
        },
        'appliance': {
            'urgent': [
                "Oven {verb} and smells like gas",
                "Refrigerator completely stopped working, {additional}",
                "Dishwasher leaking all over {location} floor",
                "Washing machine overflowing, {additional}",
                "Dryer smoking and smells like burning",
                "Gas stove burner won't turn off",
                "Refrigerator making loud grinding noise and not cooling",
                "Microwave sparked and caught fire briefly",
                "Carbon monoxide alarm going off near {location} appliance",
                "Washing machine violently shaking and moving across floor"
            ],
            'semi-urgent': [
                "Refrigerator {verb} but still somewhat cool",
                "Oven not heating to correct temperature",
                "Dishwasher not draining properly",
                "Washing machine {verb} mid-cycle",
                "Dryer not heating clothes",
                "Garbage disposal jammed in {location}",
                "Stove burner not lighting",
                "Microwave stopped heating food",
                "Ice maker leaking water onto floor",
                "Range hood fan not working"
            ],
            'non-urgent': [
                "Dishwasher rack missing wheel",
                "Refrigerator ice maker working intermittently",
                "Dryer lint trap damaged",
                "Microwave light bulb burnt out",
                "Oven temperature slightly off",
                "Washing machine making slight noise during spin cycle",
                "Freezer door not sealing perfectly",
                "Stove burner heating unevenly",
                "Refrigerator vegetable drawer cracked",
                "Oven clock reset needed"
            ]
        },
        'miscellaneous': {
            'urgent': [
                "Suspicious gas smell in {location}, {additional}",
                "Front door lock broken, cannot secure apartment",
                "Carbon monoxide detector alarming in {location}",
                "Break-in attempt, {location} window damaged",
                "Neighbor's sewage leaking into my apartment",
                "Threatening graffiti on {location} door",
                "Fire damage in {location}, {additional}",
                "Severe mold in {location}, causing breathing difficulties",
                "Fallen tree branch crashed through {location} window",
                "Entire ceiling water damaged and sagging dangerously"
            ],
            'semi-urgent': [
                "Smoke detector {verb} in {location}",
                "Mold growing in {location}, {additional}",
                "Mailbox broken into",
                "Strange odor in {location}, {additional}",
                "Security camera not working at {location} entrance",
                "Garbage chute blocked on our floor",
                "Excessive moisture in {location}",
                "Key fob not working for building access",
                "Garage door opener malfunctioning",
                "Intercom system not functioning properly"
            ],
            'non-urgent': [
                "Hallway light timer too short",
                "Recycle bin missing lid",
                "Package delivery shelf damaged",
                "Building directory needs updating",
                "Minor water stain on {location} ceiling",
                "Mailbox number faded",
                "Squeaky floorboard in {location}",
                "Fitness room equipment needs maintenance",
                "Community room blinds damaged",
                "Small patch of carpet worn in {location} hallway"
            ]
        }
    }
    
    # Define location options
    locations = [
        "kitchen", "bathroom", "bedroom", "living room", "hallway", "closet", 
        "laundry room", "dining room", "entryway", "office", "balcony", "basement"
    ]
    
    # Define pest options for pest category
    pests = ["mouse", "cockroach", "spider", "ant", "rat", "beetle", "centipede"]
    
    # Define verbs by urgency
    verbs = {
        'urgent': ["leaking severely", "completely broken", "collapsed", "sparking", "smoking", 
                   "burst", "overflowing", "shattered", "failed completely", "making loud banging noise"],
        'semi-urgent': ["leaking steadily", "not working properly", "loose", "cracked", 
                         "malfunctioning", "running constantly", "stopped working", "making strange noise", 
                         "not closing properly", "tripping frequently"],
        'non-urgent': ["dripping slightly", "squeaking", "slightly damaged", "minor wear", 
                        "needs adjustment", "cosmetic damage", "making occasional noise", 
                        "slightly loose", "minorly discolored", "operating inconsistently"]
    }
    
    # Define additional details by urgency
    additional_details = {
        'urgent': ["water damage spreading rapidly", "need immediate attention", "safety hazard", 
                   "caused injury", "getting worse quickly", "extremely unsafe", 
                   "affecting multiple units", "children in home", "elderly resident needs assistance", 
                   "health risk", "unable to stay in apartment", "emergency situation"],
        'semi-urgent': ["been ongoing for days", "getting gradually worse", "affecting daily activities", 
                        "inconvenient but manageable", "need to resolve soon", "moderate concern", 
                        "causing some discomfort", "recurring problem", "need to fix this week"],
        'non-urgent': ["minor issue", "no rush", "whenever convenient", "not urgent", 
                       "just wanted to report", "low priority", "purely cosmetic", 
                       "slight inconvenience", "whenever next available"]
    }
    
    # Initialize dataset
    data = []
    
    # Generate data for each category
    for category in categories:
        for _ in range(samples_per_category):
            # Distribute urgency levels evenly within each category
            urgency = random.choice(urgency_levels)
            
            # Determine if photo is attached (based on urgency probability)
            has_photo = random.random() < photo_prob[urgency]
            
            # Generate random timestamp within February 2025
            time_delta = random.random() * (end_date - start_date).total_seconds()
            timestamp = start_date + timedelta(seconds=time_delta)
            
            # Generate text based on templates for the category and urgency
            template = random.choice(text_templates[category][urgency])
            
            # Replace placeholders
            text = template
            if "{location}" in text:
                text = text.replace("{location}", random.choice(locations))
            if "{verb}" in text:
                text = text.replace("{verb}", random.choice(verbs[urgency]))
            if "{additional}" in text:
                text = text.replace("{additional}", random.choice(additional_details[urgency]))
            if "{pest}" in text:
                text = text.replace("{pest}", random.choice(pests))
            
            # Capitalize first letter
            text = text[0].upper() + text[1:]
            
            # Add period if not present
            if not text.endswith('.'):
                text += '.'
                
            # Add entry to dataset
            data.append({
                'text': text,
                'category': category,
                'urgency': urgency,
                'timestamp': timestamp,
                'has_photo': has_photo
            })
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Shuffle data
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    return df

def save_dataset(df, filepath='tenant_issue_dataset.csv'):
    """Save the generated dataset to a CSV file"""
    df.to_csv(filepath, index=False)
    print(f"Dataset saved to {filepath}")
    
def display_dataset_stats(df):
    """Display statistics about the generated dataset"""
    print(f"Dataset size: {len(df)} entries")
    
    print("\nCategory distribution:")
    print(df['category'].value_counts())
    
    print("\nUrgency distribution:")
    print(df['urgency'].value_counts())
    
    print("\nHas photo distribution by urgency:")
    for urgency in df['urgency'].unique():
        subset = df[df['urgency'] == urgency]
        photo_percent = subset['has_photo'].mean() * 100
        print(f"{urgency}: {photo_percent:.1f}% have photos ({subset['has_photo'].sum()} out of {len(subset)})")
    
    print("\nSample entries:")
    for urgency in df['urgency'].unique():
        print(f"\n--- {urgency.upper()} example ---")
        sample = df[df['urgency'] == urgency].sample(1).iloc[0]
        print(f"Text: {sample['text']}")
        print(f"Category: {sample['category']}")
        print(f"Timestamp: {sample['timestamp']}")
        print(f"Has photo: {sample['has_photo']}")

# Generate and display the dataset
if __name__ == "__main__":
    # Generate dataset
    tenant_dataset = generate_tenant_issue_dataset()
    
    # Display statistics
    display_dataset_stats(tenant_dataset)
    
    # Save dataset to CSV
    save_dataset(tenant_dataset)
    
    print("\nDataset generation complete!")