import pandas as pd
import numpy as np

# Create categories and ads
categories = {
    "Technology": ["Smartphone", "Laptop", "Tablet", "Smartwatch", "Headphones"],
    "Fashion": ["Shoes", "Shirts", "Jeans", "Dresses", "Accessories"],
    "Food": ["Pizza", "Burger", "Sushi", "Salad", "Coffee"],
    "Travel": ["Hotels", "Flights", "Car Rentals", "Vacation Packages", "Cruises"],
    "Entertainment": ["Movies", "Music", "Games", "Streaming Services", "Books"]
}

# Create a list of ads with their properties
ads = []
ad_id = 0

for category, products in categories.items():
    for product in products:
        for i in range(1, 4):  # Create 3 variations of each product ad
            ads.append({
                'ad_id': ad_id,
                'category': category,
                'product': product,
                'title': f"Amazing {product} {i}",
                'description': f"Check out our {product} deals! Version {i}",
                'click_rate_base': np.random.uniform(0.01, 0.05)  # Base click rate for simulation
            })
            ad_id += 1

# Convert to DataFrame
ads_df = pd.DataFrame(ads)

print(f"Created {len(ads_df)} ads across {len(categories)} categories")
print("\nSample ads:")
print(ads_df.sample(5))

# Function to get ads by category
def get_ads_by_category(category=None):
    if category:
        return ads_df[ads_df['category'] == category]
    return ads_df

# Function to get ad by ID
def get_ad_by_id(ad_id):
    return ads_df[ads_df['ad_id'] == ad_id].iloc[0] if ad_id in ads_df['ad_id'].values else None