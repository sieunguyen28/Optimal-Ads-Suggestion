import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import uuid
import networkx as nx
from ad_database import ads_df, get_ad_by_id
from q_learning_agent import AdRecommendationAgent
from user_simulator import UserSimulator

# Cache ads_df to improve performance
@st.cache_data(max_entries=100, ttl=3600)
def load_ads_data():
    if ads_df.empty:
        raise ValueError("ads_df is empty")
    required_columns = ['ad_id', 'category', 'product', 'title']
    missing_columns = [col for col in required_columns if col not in ads_df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in ads_df: {missing_columns}")
    return ads_df

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.agent = AdRecommendationAgent(
        num_ads=len(ads_df),
        num_categories=len(ads_df['category'].unique()),
        alpha=0.1,
        gamma=0.85,
        epsilon=0.1
    )
    st.session_state.simulator = UserSimulator(num_users=10)
    st.session_state.current_user = 0
    st.session_state.clicks = []
    st.session_state.rewards = []
    st.session_state.categories = []
    st.session_state.current_page = "home"
    st.session_state.current_ad = None
    st.session_state.interaction_count = 0
    st.session_state.last_category_update = {}
    st.session_state.category_avg_rewards = {}
    st.session_state.initialized = True
    st.session_state.needs_rerun = False
    st.session_state.purchase_message = False

# App title
st.title("Personalized Ad System with Q-Learning")

# Sidebar for user selection and filters
st.sidebar.header("User Settings")
user_id = st.sidebar.selectbox(
    "Select User",
    options=list(range(10)),
    index=st.session_state.current_user
)

if user_id != st.session_state.current_user:
    st.session_state.current_user = user_id
    st.session_state.clicks = []
    st.session_state.rewards = []
    st.session_state.categories = []
    st.session_state.needs_rerun = True
    st.session_state.purchase_message = False

# Display user preferences
st.sidebar.subheader("User Preferences (Learned)")
user_prefs = st.session_state.simulator.user_preferences[user_id]
for category, pref in user_prefs.items():
    st.sidebar.progress(pref)
    st.sidebar.text(f"{category}: {pref:.2f}")

# Add category filter
selected_category = st.sidebar.selectbox("Filter by Category", ["All"] + list(ads_df['category'].unique()))

# Add reset button
if st.sidebar.button("Reset All"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.session_state.agent = AdRecommendationAgent(
        num_ads=len(ads_df),
        num_categories=len(ads_df['category'].unique()),
        alpha=0.1,
        gamma=0.85,
        epsilon=0.1
    )
    st.session_state.simulator = UserSimulator(num_users=10)
    st.session_state.current_user = 0
    st.session_state.clicks = []
    st.session_state.rewards = []
    st.session_state.categories = []
    st.session_state.current_page = "home"
    st.session_state.current_ad = None
    st.session_state.interaction_count = 0
    st.session_state.last_category_update = {}
    st.session_state.category_avg_rewards = {}
    st.session_state.initialized = True
    st.session_state.needs_rerun = True
    st.session_state.purchase_message = False
    # Xóa sạch user_history, Q-table và trạng thái
    st.session_state.simulator.user_history = {i: [] for i in range(10)}
    st.session_state.agent.user_q_tables = {}
    st.session_state.agent.user_states = {}
    st.session_state.agent.q_update_history = {}

# Function to create a unique key for each button
def get_unique_key(prefix, ad_id):
    return f"{prefix}_{ad_id}_{uuid.uuid4()}"

# Function to calculate reward based on probabilities
def calculate_reward(user_id, ad_id, action_type):
    ad = get_ad_by_id(ad_id)
    category = ad['category']
    user_preference = st.session_state.simulator.user_preferences[user_id].get(category, 0.0)
    base_click_rate = ad.get('click_rate_base', 0.03)
    history = st.session_state.simulator.get_user_history(user_id)

    # Yếu tố tương tác gần đây (5 tương tác cuối)
    recent_interactions = [item for item in history[-5:] if item['category'] == category]
    interaction_factor = 1 + 0.1 * len(recent_interactions)

    # Yếu tố tương tác dài hạn (toàn bộ lịch sử)
    total_interactions = len([item for item in history if item['category'] == category])
    diversity_penalty = 1 / (1 + total_interactions * 0.05)  # Giảm phần thưởng nếu danh mục được gợi ý quá nhiều

    # Tính xác suất nhấp/mua
    click_probability = base_click_rate * (1 + 2 * user_preference) * interaction_factor * diversity_penalty
    click_probability = max(0.1, min(0.5, click_probability))
    purchase_probability = click_probability * 0.2

    if action_type == 'view':
        return click_probability * 10
    elif action_type == 'purchase':
        return (0.5 + purchase_probability) * 10
    else:
        return 0

# Function to update category average rewards
def update_category_avg_reward(user_id, category, reward):
    if user_id not in st.session_state.category_avg_rewards:
        st.session_state.category_avg_rewards[user_id] = {}
    if category not in st.session_state.category_avg_rewards[user_id]:
        st.session_state.category_avg_rewards[user_id][category] = {'total': 0, 'count': 0}
    st.session_state.category_avg_rewards[user_id][category]['total'] += reward
    st.session_state.category_avg_rewards[user_id][category]['count'] += 1

# Function to get category average reward
def get_category_avg_reward(user_id, category):
    if user_id not in st.session_state.category_avg_rewards or category not in st.session_state.category_avg_rewards[user_id]:
        return 0
    cat_data = st.session_state.category_avg_rewards[user_id][category]
    return cat_data['total'] / cat_data['count'] if cat_data['count'] > 0 else 0

# Handle ad click
def handle_ad_click(ad_id):
    st.session_state.current_ad = ad_id
    st.session_state.current_page = "ad_detail"
    user_id = st.session_state.current_user
    reward = calculate_reward(user_id, ad_id, 'view')
    ad = get_ad_by_id(ad_id)
    category = ad['category']
    current_pref = st.session_state.simulator.user_preferences[user_id].get(category, 0.0)
    update_rate = 0.05
    new_pref = current_pref + update_rate * (1.0 - current_pref)
    new_pref = max(0.0, min(1.0, new_pref))
    st.session_state.simulator.user_preferences[user_id][category] = new_pref
    if user_id not in st.session_state.simulator.user_history:
        st.session_state.simulator.user_history[user_id] = []
    st.session_state.simulator.user_history[user_id].append({
        'ad_id': ad_id,
        'category': category,
        'product': ad['product'],
        'action': 'view',
        'reward': reward,
        'timestamp': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    update_category_avg_reward(user_id, category, reward)
    st.session_state.agent.update_q_table(
        user_id,
        ad_id,
        reward,
        st.session_state.simulator.get_user_history(user_id)
    )
    st.session_state.clicks.append(True)
    st.session_state.rewards.append(reward)
    st.session_state.categories.append(category)
    st.session_state.last_category_update[category] = time.time()
    st.session_state.interaction_count += 1
    st.session_state.needs_rerun = True
    st.session_state.purchase_message = False

# Handle purchase
def handle_purchase(ad_id):
    user_id = st.session_state.current_user
    purchase_reward = calculate_reward(user_id, ad_id, 'purchase')
    ad = get_ad_by_id(ad_id)
    category = ad['category']
    current_pref = st.session_state.simulator.user_preferences[user_id].get(category, 0.0)
    update_rate = 0.075
    new_pref = current_pref + update_rate * (1.0 - current_pref)
    new_pref = max(0.0, min(1.0, new_pref))
    st.session_state.simulator.user_preferences[user_id][category] = new_pref
    st.session_state.simulator.user_history[user_id].append({
        'ad_id': ad_id,
        'category': category,
        'product': ad['product'],
        'action': 'purchase',
        'reward': purchase_reward,
        'timestamp': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    update_category_avg_reward(user_id, category, purchase_reward)
    st.session_state.agent.update_q_table(
        user_id,
        ad_id,
        purchase_reward,
        st.session_state.simulator.get_user_history(user_id)
    )
    if len(st.session_state.rewards) > 0:
        st.session_state.rewards[-1] = purchase_reward
    st.session_state.purchase_message = True
    st.session_state.needs_rerun = True

# Go back to home
def go_back_to_home():
    st.session_state.current_page = "home"
    st.session_state.current_ad = None
    st.session_state.needs_rerun = True
    st.session_state.purchase_message = False

# Get recommendations
def get_recommendations(user_id, user_prefs, n=30):
    all_categories = ads_df['category'].unique()
    if st.session_state.interaction_count < 5:
        recommended_ads = []
        for category in all_categories:
            category_ads = ads_df[ads_df['category'] == category]
            if len(category_ads) > 0:
                num_ads = min(3, len(category_ads))
                category_sample = category_ads['ad_id'].sample(num_ads).tolist()
                recommended_ads.extend(category_sample)
        remaining_slots = n - len(recommended_ads)
        if remaining_slots > 0:
            remaining_ads = ads_df[~ads_df['ad_id'].isin(recommended_ads)]
            if len(remaining_ads) > 0:
                random_fill = remaining_ads['ad_id'].sample(min(remaining_slots, len(remaining_ads))).tolist()
                recommended_ads.extend(random_fill)
        random.shuffle(recommended_ads)
        return recommended_ads[:n]

    all_max_prefs = all(pref >= 0.99 for pref in user_prefs.values())
    recommended_ads = []
    categories_with_data = []
    history = st.session_state.simulator.get_user_history(user_id)
    category_interaction_counts = {cat: 0 for cat in all_categories}
    for item in history:
        if 'category' in item:
            category_interaction_counts[item['category']] += 1

    for category in all_categories:
        pref = user_prefs.get(category, 0.0)
        avg_reward = get_category_avg_reward(user_id, category)
        last_update = st.session_state.last_category_update.get(category, 0)
        interaction_count = category_interaction_counts.get(category, 0)
        diversity_bonus = 1 / (1 + interaction_count)

        if all_max_prefs:
            # Sử dụng lịch sử tương tác dài hạn và Q-value
            total_interactions = len([item for item in history if item['category'] == category])
            recent_interactions = len([item for item in history[-5:] if item['category'] == category])
            # Lấy giá trị Q trung bình của danh mục
            category_ads = ads_df[ads_df['category'] == category]['ad_id'].tolist()
            q_values = st.session_state.agent.get_q_table(user_id)[st.session_state.agent.user_states[user_id]]
            avg_q_value = np.mean([q_values[ad_id] for ad_id in category_ads]) if category_ads else 0
            adjusted_pref = (1 + avg_reward) * (1 + 0.1 * recent_interactions) * (1 + 0.05 * total_interactions) * diversity_bonus * (1 + avg_q_value)
        else:
            adjusted_pref = pref * (1 + avg_reward) * diversity_bonus

        categories_with_data.append((category, adjusted_pref, avg_reward, last_update))

    sorted_categories = sorted(categories_with_data, key=lambda x: (x[1], x[3]), reverse=True)
    min_ads_per_category = 3
    diversity_ads = []
    max_ads_per_category = 8
    category_counts = {category: 0 for category, _, _, _ in sorted_categories}

    for category, _, _, _ in sorted_categories:
        category_ads = ads_df[ads_df['category'] == category]
        if len(category_ads) == 0:
            continue
        category_ad_ids = category_ads['ad_id'].tolist()
        top_q_ads = [ad_id for ad_id in st.session_state.agent.get_top_ads_for_user(user_id, n=len(category_ad_ids)) if ad_id in category_ad_ids]
        num_ads = min(min_ads_per_category, len(category_ads))
        if len(top_q_ads) >= num_ads:
            diversity_ads.extend(top_q_ads[:num_ads])
            category_counts[category] += num_ads
        else:
            diversity_ads.extend(top_q_ads)
            category_counts[category] += len(top_q_ads)
            remaining = num_ads - len(top_q_ads)
            remaining_ads = [ad for ad in category_ad_ids if ad not in top_q_ads]
            if remaining_ads:
                additional_ads = random.sample(remaining_ads, min(remaining, len(remaining_ads)))
                diversity_ads.extend(additional_ads)
                category_counts[category] += len(additional_ads)

    remaining_slots = n - len(diversity_ads)
    if remaining_slots > 0:
        total_adjusted_pref = sum(pref for _, pref, _, _ in sorted_categories) or 1
        max_additional_per_category = min(max_ads_per_category, remaining_slots // 2)
        category_allocation = {}
        remaining_after_allocation = remaining_slots

        for category, adjusted_pref, _, _ in sorted_categories:
            if adjusted_pref > 0:
                raw_allocation = int((adjusted_pref / total_adjusted_pref) * remaining_slots)
                current_count = category_counts.get(category, 0)
                allocation = min(raw_allocation, max_additional_per_category, max_ads_per_category - current_count)
                category_allocation[category] = allocation
                remaining_after_allocation -= allocation

        if remaining_after_allocation > 0:
            low_pref_categories = [(cat, pref) for cat, pref, _, _ in sorted_categories if pref < 0.3]
            if low_pref_categories:
                low_pref_categories.sort(key=lambda x: x[1])
                slots_per_category = remaining_after_allocation // len(low_pref_categories) or 1
                for category, _ in low_pref_categories:
                    additional = min(slots_per_category, remaining_after_allocation)
                    category_allocation[category] = category_allocation.get(category, 0) + additional
                    remaining_after_allocation -= additional
                    if remaining_after_allocation <= 0:
                        break
            if remaining_after_allocation > 0 and sorted_categories:
                top_category = sorted_categories[0][0]
                category_allocation[top_category] = category_allocation.get(top_category, 0) + remaining_after_allocation

        for category, _, _, _ in sorted_categories:
            if category not in category_allocation or category_allocation[category] <= 0:
                continue
            category_ads = ads_df[ads_df['category'] == category]
            category_ads = category_ads[~category_ads['ad_id'].isin(diversity_ads)]
            if len(category_ads) == 0:
                continue
            all_ads_for_category = category_ads['ad_id'].tolist()
            top_q_ads_for_category = [ad_id for ad_id in st.session_state.agent.get_top_ads_for_user(user_id, n=len(all_ads_for_category)) if ad_id in all_ads_for_category and ad_id not in recommended_ads]
            if len(top_q_ads_for_category) < category_allocation[category]:
                remaining_ads = [ad_id for ad_id in all_ads_for_category if ad_id not in top_q_ads_for_category and ad_id not in recommended_ads]
                random.shuffle(remaining_ads)
                top_q_ads_for_category.extend(remaining_ads)
            selected_ads = top_q_ads_for_category[:category_allocation[category]]
            recommended_ads.extend(selected_ads)

    final_recommendations = diversity_ads + recommended_ads
    if len(final_recommendations) < n:
        remaining = n - len(final_recommendations)
        available_ads = ads_df[~ads_df['ad_id'].isin(final_recommendations)]
        if len(available_ads) > 0:
            random_fill = available_ads['ad_id'].sample(min(remaining, len(available_ads))).tolist()
            final_recommendations.extend(random_fill)

    return final_recommendations[:n]

st.markdown("""
    <style>
    .ad-item {
        font-size: 14px;  
        margin-bottom: 10px;  
        white-space: nowrap;  
    }
    .ad-item h4 {
        font-size: 18px; 
        margin: 0;
    }
    .ad-item p {
        margin: 2px 0;  
    }
    .ad-item button {
        font-size: 16px;  
        padding: 5px;
    }
    .st-emotion-cache-4u6e0b {
        width: 1000px;  
        position -webkit-box-sizing: border-box;
    }
    .st-emotion-cache-mtjnbi{
        max-width: 90%;
    }
    .st-emotion-cache-4u6e0b {
        position: relative;
        display: flex;
        flex: 1 1 auto;  /* Co giãn linh hoạt */
        flex-direction: column;
        gap: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Check which page to display
if st.session_state.current_page == "home":
    col1, col2 = st.columns([3, 2])  

    # Cột bên trái: Ad Recommendations
    with col1:
        st.markdown('<div class="scrollable-column">', unsafe_allow_html=True)
        st.header("Ad Recommendations")
        user_history = st.session_state.simulator.get_user_history(user_id)
        user_prefs = st.session_state.simulator.user_preferences[user_id]
        all_ad_ids = get_recommendations(user_id, user_prefs, n=30)
        
        # Apply category filter
        if selected_category != "All":
            all_ad_ids = [ad_id for ad_id in all_ad_ids if get_ad_by_id(ad_id)['category'] == selected_category]

        st.subheader("Top Recommendations")
        for row in range(6):
            cols = st.columns(5)
            for i, ad_id in enumerate(all_ad_ids[row*5:(row+1)*5]):
                ad = get_ad_by_id(ad_id)
                with cols[i]:
                    # Sử dụng div để kiểm soát hiển thị quảng cáo
                    st.markdown(f"""
                        <div class="ad-item">
                            <h4>{ad['title']}</h4>
                            <p>Category: {ad['category']}</p>
                            <p>Product: {ad['product']}</p>
                        </div>
                    """, unsafe_allow_html=True)
                    button_key = get_unique_key("ad", ad_id)
                    if st.button(f"View {row*5+i+1}", key=button_key, on_click=handle_ad_click, args=(ad_id,)):
                        pass
        st.markdown('</div>', unsafe_allow_html=True)

    # Cột bên phải: Learning Progress, User Interaction History, v.v.
    with col2:
        st.markdown('<div class="scrollable-column">', unsafe_allow_html=True)
        
        # Learning Progress
        st.header("Learning Progress")
        if len(st.session_state.clicks) > 0:
            total_interactions = len(st.session_state.clicks)
            avg_reward = sum(st.session_state.rewards) / len(st.session_state.rewards)
            col1_inner, col2_inner = st.columns(2)
            col1_inner.metric("Total Interactions", f"{total_interactions}")
            col2_inner.metric("Average Reward", f"{avg_reward:.4f}")
            
            if len(st.session_state.categories) > 0:
                st.subheader("Ad Categories Viewed")
                category_counts = pd.Series(st.session_state.categories).value_counts()
                st.bar_chart(category_counts)
                
            if len(st.session_state.categories) > 5:
                st.subheader("Preference Evolution")
                fig, ax = plt.subplots()
                prefs = st.session_state.simulator.user_preferences[user_id]
                categories = list(prefs.keys())
                values = list(prefs.values())
                ax.bar(categories, values)
                ax.set_xlabel("Category")
                ax.set_ylabel("Preference Level")
                ax.set_ylim(0, 1)
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
                
            if user_id in st.session_state.category_avg_rewards and len(st.session_state.category_avg_rewards[user_id]) > 0:
                st.subheader("Average Reward by Category")
                fig, ax = plt.subplots()
                cat_rewards = {}
                for cat, data in st.session_state.category_avg_rewards[user_id].items():
                    if data['count'] > 0:
                        cat_rewards[cat] = data['total'] / data['count']
                if cat_rewards:
                    categories = list(cat_rewards.keys())
                    values = list(cat_rewards.values())
                    ax.bar(categories, values)
                    ax.set_xlabel("Category")
                    ax.set_ylabel("Average Reward")
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)
                
            if len(st.session_state.rewards) > 5:
                st.subheader("Learning Curve")
                fig, ax = plt.subplots()
                window_size = min(10, len(st.session_state.rewards))
                rewards_series = pd.Series(st.session_state.rewards)
                moving_avg = rewards_series.rolling(window=window_size).mean()
                ax.plot(moving_avg)
                ax.set_xlabel("Interaction")
                ax.set_ylabel("Reward (Moving Average)")
                st.pyplot(fig)
        else:
            st.info("Click on ads to see learning progress.")

        # Display user history with Q-table and VIPER model
        st.header("User Interaction History")
        if user_history:
            history_df = pd.DataFrame(user_history[-10:])
            columns_to_display = ['timestamp', 'category', 'product', 'action', 'reward', 'ad_id']
            display_columns = [col for col in columns_to_display if col in history_df.columns]
            if display_columns:
                st.dataframe(history_df[display_columns])
            else:
                st.dataframe(history_df)

            # Hiển thị Q-table
            st.subheader("Current Q-Table for User")
            def highlight_high_q(val):
                color = 'background-color: lightgreen' if float(val) > 0.1 else ''
                return color
            q_table = st.session_state.agent.get_q_table(user_id)
            q_table_df = pd.DataFrame(
                q_table,
                index=[f"State {i}" for i in range(q_table.shape[0])],
                columns=[f"Ad {i}" for i in range(q_table.shape[1])]
            )
            st.dataframe(q_table_df.style.format("{:.4f}").applymap(highlight_high_q))
            
            # Tạo ánh xạ state_to_category động dựa trên self.categories
            state_to_category = {0: "No History"}
            for idx, category in enumerate(st.session_state.agent.categories):
                state_to_category[idx + 1] = category
            st.markdown("**State to Category Mapping:**")
            for state, category in state_to_category.items():
                st.markdown(f"- **State {state}**: {category}")

            # VIPER Model: Q-Value Transition Diagram
            st.subheader("Q-Value Transition Diagram (VIPER Model)")
            q_update_history = st.session_state.agent.get_q_update_history(user_id)
            if q_update_history:
                # Tạo biểu đồ
                G = nx.DiGraph()
                states = set()
                ad_nodes = set()
                for update in q_update_history:
                    old_state = f"State {update['old_state']}"
                    new_state = f"State {update['new_state']}"
                    states.add(old_state)
                    states.add(new_state)
                G.add_nodes_from(states)
                G.add_node("Start")

                # Thêm cạnh từ "Start" đến tất cả các trạng thái
                for state in states:
                    G.add_edge("Start", state)

                # Thêm các nút quảng cáo và cạnh từ trạng thái đến quảng cáo
                edge_labels = {}
                edge_colors = []
                ad_ids_seen = set()
                for update in q_update_history:
                    old_state = f"State {update['old_state']}"
                    ad_id = update['ad_id']
                    if ad_id not in ad_ids_seen:
                        ad = get_ad_by_id(ad_id)
                        if ad is not None:
                            ad_node = f"Ad {ad_id}"
                            ad_nodes.add(ad_node)
                            G.add_node(ad_node)
                            G.add_edge(old_state, ad_node)
                            # Thêm thông tin Q-value và phần thưởng vào nhãn
                            q_value = update['new_value']
                            reward = update['reward']
                            edge_labels[(old_state, ad_node)] = f"Ad {ad_id}: {ad['product']} ({ad['category']})\nQ: {q_value:.2f}, R: {reward:.2f}"
                            # Tô màu cạnh dựa trên Q-value
                            edge_colors.append('red' if q_value > 0.5 else 'blue' if q_value > 0.1 else 'gray')
                            ad_ids_seen.add(ad_id)

                # Tạo bố cục tùy chỉnh
                pos = {}
                state_nodes = sorted([node for node in G.nodes() if node.startswith("State")], key=lambda x: int(x.split()[-1]))
                for i, state in enumerate(state_nodes):
                    pos[state] = (0, -i)  # Xếp dọc ở x=0
                pos["Start"] = (1, -len(state_nodes) / 2)
                ad_nodes_list = sorted([node for node in G.nodes() if node.startswith("Ad")], key=lambda x: int(x.split()[1]))
                for i, ad_node in enumerate(ad_nodes_list):
                    pos[ad_node] = (-1, -i)  # Xếp dọc ở x=-1

                # Vẽ biểu đồ
                fig, ax = plt.subplots(figsize=(12, 8))
                nx.draw_networkx_nodes(G, pos, nodelist=[node for node in G.nodes() if node.startswith("State")], node_color='lightblue', node_size=1000, ax=ax)
                nx.draw_networkx_nodes(G, pos, nodelist=["Start"], node_color='lightblue', node_size=1000, ax=ax)
                nx.draw_networkx_nodes(G, pos, nodelist=[node for node in G.nodes() if node.startswith("Ad")], node_color='lightgreen', node_size=1000, ax=ax)
                nx.draw_networkx_edges(G, pos, edge_color=edge_colors, arrows=True, arrowsize=20, ax=ax)
                nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)
                nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, label_pos=0.5, ax=ax)
                plt.title("Q-Value Transition Diagram", fontsize=14)
                plt.axis('off')
                st.pyplot(fig)

                # Hiển thị các hành động có Q-value cao nhất
                st.subheader("Top Q-Value Actions")
                q_table = st.session_state.agent.get_q_table(user_id)
                for state in range(q_table.shape[0]):
                    q_values = q_table[state]
                    top_ad_id = np.argmax(q_values)
                    top_q_value = q_values[top_ad_id]
                    ad = get_ad_by_id(top_ad_id)
                    if ad is not None and top_q_value > 0.1:  # Sửa lỗi ở đây
                        st.markdown(f"- **State {state} ({state_to_category[state]})**: Ad {top_ad_id} ({ad['product']}, {ad['category']}) - Q-Value: {top_q_value:.4f}")

            # Interactive: Select a state to see top actions
            st.subheader("Explore Q-Table by State")
            num_states = q_table.shape[0]
            selected_state = st.selectbox("Select a State to Explore", [f"State {i}" for i in range(num_states)])
            state_idx = int(selected_state.split()[-1])
            q_values = q_table[state_idx]
            top_action_indices = q_values.argsort()[-5:][::-1]  # Top 5 actions
            st.markdown(f"**Top Actions for {selected_state} ({state_to_category[state_idx]})**")
            for ad_id in top_action_indices:
                q_value = q_values[ad_id]
                if q_value > 0.1:  # Chỉ hiển thị các hành động có Q-value > 0.1
                    ad = get_ad_by_id(ad_id)
                    if ad is not None:
                        st.markdown(f"- Ad {ad_id}: {ad['product']} ({ad['category']}) - Q-Value: {q_value:.4f}")

        else:
            st.info("No interaction history yet.")
        
        st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.current_page == "ad_detail":
    ad_id = st.session_state.current_ad
    ad = get_ad_by_id(ad_id)
    if ad is None:
        st.error(f"Ad not found! ID: {ad_id}")
        if st.button("Return to Home"):
            go_back_to_home()
    else:
        if st.button("← Back to Recommendations"):
            go_back_to_home()
        st.header(ad['title'])
        # Display purchase message if set
        if st.session_state.purchase_message:
            st.success("Purchase successful! (This is a simulation)")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(f"https://via.placeholder.com/300x200?text={ad['product']}", caption=f"{ad['product']} Image")
        with col2:
            st.subheader("Product Details")
            st.markdown(f"**Category:** {ad['category']}")
            st.markdown(f"**Product:** {ad['product']}")
            st.markdown(f"**Description:** {ad.get('description', 'No description available')}")
            st.markdown("### Features")
            st.markdown("- High quality product")
            st.markdown("- Best in class performance")
            st.markdown("- Excellent customer reviews")
            st.markdown("### Price")
            random.seed(ad_id)
            price = random.randint(50, 500)
            st.markdown(f"**${price}.99**")
            if st.button("Buy Now"):
                handle_purchase(ad_id)
        
        st.header("Related Products")
        same_category_ads = ads_df[ads_df['category'] == ad['category']]
        same_category_ads = same_category_ads[same_category_ads['ad_id'] != ad_id]
        if len(same_category_ads) > 0:
            sample_size = min(3, len(same_category_ads))
            same_category_ads = same_category_ads.sample(sample_size)
            cols = st.columns(sample_size)
            for i, (_, related_ad) in enumerate(same_category_ads.iterrows()):
                related_ad_id = related_ad['ad_id']
                with cols[i]:
                    st.subheader(related_ad['title'])
                    st.text(f"Product: {related_ad['product']}")
                    button_key = get_unique_key("related", related_ad_id)
                    if st.button(f"View Product", key=button_key, on_click=handle_ad_click, args=(related_ad_id,)):
                        pass
        else:
            st.info("No related products found.")

# Instructions
st.sidebar.markdown("""
## How to use
1. Select a user from the dropdown
2. Filter ads by category if desired
3. Click on ads that interest you
4. The system will learn your preferences
5. Watch how recommendations improve over time
6. Use the "Reset All" button to start over
""")

# Handle rerun only when needed
if st.session_state.needs_rerun:
    st.session_state.needs_rerun = False
    st.rerun()