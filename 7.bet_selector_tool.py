import pandas as pd

# 🏦 Step 0: Enter today's bankroll
daily_bankroll = 10000
bankroll_perc = 0.1
stake_pool = daily_bankroll * bankroll_perc

# 🎯 Strategy filters
min_ev_threshold = 0.00
min_kelly_fraction = 0.00
max_odds_threshold = 100.0

# ✅ NEW: Winrate filter settings
winrate_filter_type = 'none'  # options: 'none', 'fixed', 'dynamic'
fixed_winrate_threshold = 0.03

# ✅ NEW: Field size constraints
min_runners = 1
max_runners = 11

# ✅ NEW: Optional rank filter
rank_filter_enabled = True  # Set to False to disable
allowed_predicted_ranks = [1,2,3]  # Only allow top 1 or 2 picks per race

# 📥 Step 1: Load predictions
predictions = pd.read_csv('Daily_Bets/predicted_win_probabilities.14.07.2025.csv')
predictions = predictions.copy()

# 🆔 Step 2: Create Race ID
predictions['Race_ID'] = predictions['Date of Race'].astype(str) + "_" + predictions['Time'].astype(str)

# 💰 Step 3: Use correct odds source
predictions['Odds_To_Use'] = predictions['Industry SP']

# 🧠 Step 4: Normalize win probabilities per race
predictions['Predicted_Win_Probability'] = predictions.groupby('Race_ID')['Predicted_Win_Probability'].transform(
    lambda x: x / x.sum()
)

# ✅ Step 4.5: Assign predicted rank and apply optional rank filter
predictions['Predicted_Rank'] = predictions.groupby('Race_ID')['Predicted_Win_Probability'] \
                                           .rank(method='first', ascending=False)

if rank_filter_enabled:
    predictions = predictions[predictions['Predicted_Rank'].isin(allowed_predicted_ranks)]

# 📈 Step 5: Calculate EV and Kelly
p = predictions['Predicted_Win_Probability']
b = predictions['Odds_To_Use'] - 1
q = 1 - p
predictions['Expected_Value'] = (p * b) - q
predictions['Kelly_Fraction'] = (b * p - q) / b

# ✅ Step 6: Field size and winrate filters
field_sizes = predictions.groupby('Race_ID')['Horse'].count().rename('Field_Size')
predictions = predictions.merge(field_sizes, left_on='Race_ID', right_index=True)

predictions['Reject_Reason'] = ''
predictions.loc[predictions['Expected_Value'] <= min_ev_threshold, 'Reject_Reason'] += 'ev_low|'
predictions.loc[predictions['Kelly_Fraction'] <= min_kelly_fraction, 'Reject_Reason'] += 'kelly_low|'
predictions.loc[predictions['Odds_To_Use'] > max_odds_threshold, 'Reject_Reason'] += 'odds_high|'

# Apply field size filter
predictions.loc[
    (predictions['Field_Size'] < min_runners) | (predictions['Field_Size'] > max_runners),
    'Reject_Reason'
] += 'field_size|'

# Apply winrate filter
if winrate_filter_type == 'fixed':
    predictions['Winrate_Threshold'] = fixed_winrate_threshold
elif winrate_filter_type == 'dynamic':
    predictions['Winrate_Threshold'] = 1 / predictions['Field_Size']
else:
    predictions['Winrate_Threshold'] = 0

predictions.loc[predictions['Predicted_Win_Probability'] <= predictions['Winrate_Threshold'], 'Reject_Reason'] += 'winrate_low|'

# ✅ Step 7: Bet flag
predictions['Bet_Recommended'] = (
    (predictions['Expected_Value'] > min_ev_threshold) &
    (predictions['Kelly_Fraction'] > min_kelly_fraction) &
    (predictions['Odds_To_Use'] <= max_odds_threshold) &
    (predictions['Field_Size'] >= min_runners) &
    (predictions['Field_Size'] <= max_runners) &
    (predictions['Predicted_Win_Probability'] > predictions['Winrate_Threshold'])
)

# 💸 Step 8: Stake = Kelly × stake pool per race
predictions['Recommended_Stake'] = 0

# 💰 Step 9: Assign stakes race-by-race
bets_to_place = []
for race_id, race_df in predictions.groupby('Race_ID'):
    temp = race_df.copy()
    temp['Stake_Unscaled'] = temp['Kelly_Fraction'] * stake_pool
    total = temp.loc[temp['Bet_Recommended'], 'Stake_Unscaled'].sum()
    if total > stake_pool and total > 0:
        scale = stake_pool / total
        temp.loc[temp['Bet_Recommended'], 'Recommended_Stake'] = temp.loc[temp['Bet_Recommended'], 'Stake_Unscaled'] * scale
    else:
        temp.loc[temp['Bet_Recommended'], 'Recommended_Stake'] = temp.loc[temp['Bet_Recommended'], 'Stake_Unscaled']
    bets_to_place.append(temp)

predictions = pd.concat(bets_to_place).reset_index(drop=True)

# 🏅 Step 10: Top 3 model picks (for display info)
predictions = predictions.sort_values(by=['Race_ID', 'Predicted_Win_Probability'], ascending=[True, False])
predictions['Top_3_Pick_Rank'] = predictions.groupby('Race_ID').cumcount() + 1
predictions['Top_3_Pick_Rank'] = predictions['Top_3_Pick_Rank'].where(predictions['Top_3_Pick_Rank'] <= 3)

# 📋 Step 11: Filter final list
final_bets = predictions[predictions['Bet_Recommended'] == True].copy()
final_bets = final_bets[['Date of Race', 'Time', 'Horse', 'Industry SP', 'Predicted_Win_Probability',
                         'Expected_Value', 'Kelly_Fraction', 'Top_3_Pick_Rank', 'Recommended_Stake']]

# 📤 Step 12: Save and display
final_bets = final_bets.sort_values(by=['Date of Race', 'Time']).reset_index(drop=True)
final_bets.to_csv('Daily_Bets/Selection_12.07.2025 Copy.csv', index=False)

# 🔍 Optional: Save rejected list for audit
rejected_bets = predictions[predictions['Bet_Recommended'] == False]
rejected_bets.to_csv('Daily_Bets/Rejected_12.07.2025 Copy.csv', index=False)

# ✅ Summary
print("\n✅ Today's scaled bets saved to Daily_Bets/Selection_12.07.2025 Copy.csv")
print("📋 Preview:\n")
print(final_bets)
