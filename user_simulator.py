import numpy as np
import pandas as pd
from ad_database import get_ad_by_id, categories

class UserSimulator:
    def __init__(self, num_users=10):
        """
        Simulate user behavior for testing the recommendation system
        
        Parameters:
        - num_users: Number of simulated users
        """
        self.num_users = num_users
        self.user_preferences = {}
        self.user_history = {}
        
        # Initialize user preferences
        self._initialize_users()
        
    def _initialize_users(self):
        """Initialize user preferences for each category with zeros"""
        category_names = list(categories.keys())
        
        for user_id in range(self.num_users):
            preferences = {}
            for category in category_names:
                preferences[category] = 0.0
            self.user_preferences[user_id] = preferences
            self.user_history[user_id] = []
            
    def simulate_interaction(self, user_id, ad_id):
        """
        Simulate user interaction with an ad
        
        Returns:
        - clicked: Boolean indicating if user clicked
        - reward: Reward value (1 for click, 0 for no click)
        """
        ad = get_ad_by_id(ad_id)
        if ad is None:
            return False, 0
            
        category = ad['category']
        preference = self.user_preferences[user_id].get(category, 0.0)
        
        # Base click probability
        click_probability = 0.1 + preference * 0.8
        click_probability = min(click_probability, 0.95)
        
        # Determine if user clicks
        clicked = np.random.random() < click_probability
        
        # Record interaction in user history
        self.user_history[user_id].append({
            'ad_id': ad_id,
            'category': category,  # Đảm bảo luôn có key 'category'
            'product': ad['product'],
            'clicked': clicked
        })
        
        # Update user preferences based on interaction
        self.update_preferences(user_id, category, clicked)
        
        return clicked, 1 if clicked else 0
    
    def update_preferences(self, user_id, category, clicked):
        """Update user preferences based on interaction"""
        current_pref = self.user_preferences[user_id][category]
        update_rate = 0.05

        if clicked:
            new_pref = current_pref + update_rate * (1.0 - current_pref)
        else:
            new_pref = current_pref - update_rate * current_pref

        new_pref = max(0.0, min(1.0, new_pref))
        self.user_preferences[user_id][category] = new_pref

        # Apply "forgetting" mechanism
        all_max_prefs = all(pref >= 0.99 for pref in self.user_preferences[user_id].values())
        decay_rate = 0.02 if all_max_prefs else 0.01  # Tăng decay_rate khi tất cả sở thích đạt 1.0
        for cat in self.user_preferences[user_id]:
            if cat != category:
                current_cat_pref = self.user_preferences[user_id][cat]
                new_cat_pref = current_cat_pref - decay_rate * current_cat_pref
                self.user_preferences[user_id][cat] = max(0.0, new_cat_pref)
            
    def get_user_history(self, user_id):
        """Get user interaction history"""
        return self.user_history.get(user_id, [])
        
    def print_user_preferences(self, user_id):
        """Print user preferences for debugging"""
        if user_id in self.user_preferences:
            print(f"User {user_id} preferences:")
            for category, pref in self.user_preferences[user_id].items():
                print(f"  {category}: {pref:.2f}")
        else:
            print(f"User {user_id} not found")