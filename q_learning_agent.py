import numpy as np
import pandas as pd
from ad_database import get_ads_by_category, ads_df

class AdRecommendationAgent:
    def __init__(self, num_ads, num_categories, alpha=0.1, gamma=0.85, epsilon=0.1):
        """
        Initialize the Q-learning agent without persistent storage
        
        Parameters:
        - num_ads: Total number of ads
        - num_categories: Total number of categories
        - alpha: Learning rate
        - gamma: Discount factor
        - epsilon: Exploration rate
        """
        self.num_ads = num_ads
        self.num_categories = num_categories
        self.base_alpha = alpha
        self.alpha = alpha
        self.gamma = gamma
        self.base_epsilon = epsilon
        self.epsilon = epsilon
        self.user_q_tables = {} 
        self.user_states = {}   
        # Định nghĩa danh mục với thứ tự cố định
        self.categories = ["Technology", "Fashion", "Food", "Travel", "Entertainment"]
        self.q_update_history = {}  
        
    def get_state_index(self, user_history):
        """Convert user history to a state index"""
        if not user_history:
            return 0
        recent_history = user_history[-5:] if len(user_history) > 5 else user_history
        categories = [item['category'] for item in recent_history if 'category' in item]
        if not categories:
            return 0
        from collections import Counter
        most_common = Counter(categories).most_common(1)
        if not most_common or most_common[0][0] not in self.categories:
            return 0
        return self.categories.index(most_common[0][0]) + 1
    
    def initialize_user(self, user_id):
        """Initialize Q-table for a new user in memory"""
        num_states = 1 + self.num_categories
        num_actions = self.num_ads
        q_table = np.random.uniform(0, 0.01, (num_states, num_actions))
        self.user_q_tables[user_id] = q_table
        self.user_states[user_id] = 0
        self.q_update_history[user_id] = []
    
    def select_ad(self, user_id, user_history=None):
        """Select an ad for the user using epsilon-greedy policy"""
        if user_id not in self.user_q_tables:
            self.initialize_user(user_id)
        if user_history:
            self.user_states[user_id] = self.get_state_index(user_history)

        # Adjust epsilon dynamically
        interaction_count = len(self.q_update_history.get(user_id, []))
        self.epsilon = max(0.05, 0.5 * (1 - interaction_count / 1000))

        # Tăng epsilon nếu một danh mục đạt pref = 1.0
        user_prefs = st.session_state.simulator.user_preferences[user_id]
        max_pref = max(user_prefs.values())
        if max_pref >= 0.99:  # Nếu có một danh mục đạt 1.0
            self.epsilon = min(0.3, self.epsilon * 1.5)  # Tăng epsilon để khuyến khích khám phá

        state = self.user_states[user_id]
        if np.random.random() < self.epsilon:
            ad_id = np.random.randint(0, self.num_ads)
        else:
            ad_id = np.argmax(self.user_q_tables[user_id][state])
        return ad_id
    
    def update_q_table(self, user_id, ad_id, reward, new_history=None):
        """Update Q-table based on reward in memory"""
        if user_id not in self.user_q_tables:
            self.initialize_user(user_id)
            
        old_state = self.user_states[user_id]
        if new_history:
            new_state = self.get_state_index(new_history)
            self.user_states[user_id] = new_state
        else:
            new_state = old_state
            
        # Adjust alpha dynamically
        interaction_count = len(self.q_update_history.get(user_id, []))
        self.alpha = max(0.05, self.base_alpha * (1 - interaction_count / 1000))
        
        old_value = self.user_q_tables[user_id][old_state, ad_id]
        next_max = np.max(self.user_q_tables[user_id][new_state])
        new_value = old_value + self.alpha * (reward + self.gamma * next_max - old_value)
        self.user_q_tables[user_id][old_state, ad_id] = new_value
        
        # Update history in memory
        self.q_update_history[user_id].append({
            'old_state': old_state,
            'new_state': new_state,
            'ad_id': ad_id,
            'reward': reward,
            'old_value': old_value,
            'next_max': next_max,
            'new_value': new_value,
            'timestamp': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    
    def get_top_ads_for_user(self, user_id, n=10):
        """Get top n ads for a user based on Q-values"""
        if user_id not in self.user_q_tables:
            self.initialize_user(user_id)
        state = self.user_states[user_id]
        q_values = self.user_q_tables[user_id][state]
        top_ad_indices = q_values.argsort()[-n:][::-1]
        return top_ad_indices.tolist()
    
    def get_preference_based_ads(self, user_id, user_preferences, n=10):
        """Get ads based on user preferences"""
        if user_id not in self.user_q_tables:
            self.initialize_user(user_id)
        sorted_categories = sorted(user_preferences.items(), key=lambda x: x[1], reverse=True)
        recommended_ads = []
        ads_per_category = {}
        state = self.user_states[user_id]
        q_values = self.user_q_tables[user_id][state]
        
        for category, preference in sorted_categories:
            category_ads = ads_df[ads_df['category'] == category]['ad_id'].values
            if len(category_ads) > 0:
                category_q_values = q_values[category_ads]
                sorted_indices = category_q_values.argsort()[::-1]
                top_category_ads = category_ads[sorted_indices]
                ads_per_category[category] = top_category_ads.tolist()
        
        for category, preference in sorted_categories:
            if preference > 0.1 and category in ads_per_category and len(ads_per_category[category]) > 0:
                recommended_ads.append(ads_per_category[category][0])
                ads_per_category[category] = ads_per_category[category][1:]
                if len(recommended_ads) >= n:
                    break
        
        remaining_slots = n - len(recommended_ads)
        if remaining_slots > 0:
            for category, preference in sorted_categories:
                if category in ads_per_category:
                    category_ads_to_add = ads_per_category[category][:remaining_slots]
                    recommended_ads.extend(category_ads_to_add)
                    remaining_slots -= len(category_ads_to_add)
                    if remaining_slots <= 0:
                        break
        
        if len(recommended_ads) < n:
            top_q_ads = q_values.argsort()[::-1]
            for ad_id in top_q_ads:
                if ad_id not in recommended_ads:
                    recommended_ads.append(ad_id)
                    if len(recommended_ads) >= n:
                        break
        
        return recommended_ads[:n]
    
    def get_q_table(self, user_id):
        """Return the Q-table for a specific user"""
        if user_id not in self.user_q_tables:
            self.initialize_user(user_id)
        return self.user_q_tables[user_id]
    
    def get_q_update_history(self, user_id):
        """Return the history of Q-value updates for a specific user"""
        if user_id not in self.q_update_history:
            self.initialize_user(user_id)
        return self.q_update_history[user_id]