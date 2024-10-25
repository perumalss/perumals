import numpy as np
import random

# Ad campaign rewards (conversion rate simulation)
ads_conversion_rates = [0.05, 0.08, 0.02, 0.1]  # Different ads have different conversion rates
n_ads = len(ads_conversion_rates)

# Parameters
epsilon = 0.1  # Exploration rate
n_rounds = 1000  # Number of bidding rounds
ad_rewards = np.zeros(n_ads)  # Cumulative rewards for each ad
ad_selections = np.zeros(n_ads)  # Number of times each ad was selected

# Multi-Armed Bandit (Epsilon-Greedy)
for round in range(n_rounds):
    if random.uniform(0, 1) < epsilon:
        selected_ad = random.choice(range(n_ads))  # Explore
    else:
        selected_ad = np.argmax(ad_rewards / (ad_selections + 1e-5))  # Exploit
    
    # Simulate conversion (reward) based on the ad's conversion rate
    reward = 1 if random.uniform(0, 1) < ads_conversion_rates[selected_ad] else 0

    # Update reward and selection count
    ad_rewards[selected_ad] += reward
    ad_selections[selected_ad] += 1

# Results
print("Ad selections:", ad_selections)
print("Ad rewards:", ad_rewards)
print("Average reward per ad:", ad_rewards / (ad_selections + 1e-5))
