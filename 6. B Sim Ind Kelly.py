import pandas as pd
import matplotlib.pyplot as plt

# ✅ Load and prepare data
data = pd.read_csv('Betting_Simulation/predicted_win_probabilities.23-25.csv')
data['Place'] = pd.to_numeric(data['Place'], errors='coerce')
data = data.sort_values(by=['Date of Race', 'Time']).reset_index(drop=True)
data['Race_ID'] = data['Date of Race'].astype(str) + "_" + data['Time'].astype(str)
data['Odds_To_Use'] = data['Industry SP']
data['Predicted_Win_Probability'] = data.groupby('Race_ID')['Predicted_Win_Probability'].transform(lambda x: x / x.sum())
data['Expected_Value'] = (data['Predicted_Win_Probability'] * (data['Odds_To_Use'] - 1)) - (1 - data['Predicted_Win_Probability'])

# ✅ Simulation settings
initial_bankroll = 1000
current_bankroll = initial_bankroll
bankroll_perc = 0.25
min_ev_threshold = 0.00
min_kelly_fraction = 0.05
max_odds_threshold = 100.0
updated_rows, rejected_rows = [], []

# ✅ Race-by-race simulation
for race_id, race_df in data.groupby('Race_ID', sort=False):
    stake_pool = current_bankroll * bankroll_perc
    race_df = race_df.copy()

    b = race_df['Odds_To_Use'] - 1
    p = race_df['Predicted_Win_Probability']
    q = 1 - p
    race_df['Kelly_Fraction'] = ((b * p) - q) / b

    race_df['Reject_Reason'] = ''
    race_df.loc[race_df['Kelly_Fraction'] <= min_kelly_fraction, 'Reject_Reason'] += 'kelly_low|'
    race_df.loc[race_df['Expected_Value'] <= min_ev_threshold, 'Reject_Reason'] += 'ev_low|'
    race_df.loc[race_df['Odds_To_Use'] > max_odds_threshold, 'Reject_Reason'] += 'odds_high|'

    race_df['Bet_Placed'] = (
        (race_df['Kelly_Fraction'] > min_kelly_fraction) &
        (race_df['Expected_Value'] > min_ev_threshold) &
        (race_df['Odds_To_Use'] <= max_odds_threshold)
    )

    race_df['Stake'] = 0
    race_df.loc[race_df['Bet_Placed'], 'Stake'] = race_df.loc[race_df['Bet_Placed'], 'Kelly_Fraction'] * stake_pool

    total_stake = race_df['Stake'].sum()
    if total_stake > stake_pool and total_stake > 0:
        race_df.loc[race_df['Bet_Placed'], 'Stake'] *= stake_pool / total_stake

    race_df['Actual_Result'] = (race_df['Place'] == 1).astype(int)
    race_df['Bet_Return'] = 0
    win_mask = race_df['Bet_Placed'] & (race_df['Actual_Result'] == 1)
    lose_mask = race_df['Bet_Placed'] & (race_df['Actual_Result'] == 0)

    race_df.loc[win_mask, 'Bet_Return'] = (race_df.loc[win_mask, 'Odds_To_Use'] - 1) * race_df.loc[win_mask, 'Stake']
    race_df.loc[lose_mask, 'Bet_Return'] = -race_df.loc[lose_mask, 'Stake']

    current_bankroll += race_df['Bet_Return'].sum()
    race_df['Bankroll_After_Race'] = current_bankroll

    updated_rows.append(race_df)
    rejected_rows.append(race_df[~race_df['Bet_Placed']])

# ✅ Post-simulation
data = pd.concat(updated_rows).reset_index(drop=True)
rejected_data = pd.concat(rejected_rows).reset_index(drop=True)
data['Max_Bankroll'] = data['Bankroll_After_Race'].cummax()
data['Drawdown'] = data['Max_Bankroll'] - data['Bankroll_After_Race']

# ✅ R-Multiple
data['R_Multiple'] = 0
data.loc[data['Bet_Placed'] & (data['Stake'] > 0), 'R_Multiple'] = data['Bet_Return'] / data['Stake']

# ✅ Summary Stats
total_bets = data['Bet_Placed'].sum()
total_staked = data.loc[data['Bet_Placed'], 'Stake'].sum()
total_profit = data['Bet_Return'].sum()
final_bankroll = current_bankroll
average_R = data.loc[data['Bet_Placed'], 'R_Multiple'].mean()
total_winning_R = data.loc[data['R_Multiple'] > 0, 'R_Multiple'].sum()
total_losing_R = data.loc[data['R_Multiple'] < 0, 'R_Multiple'].sum()
max_drawdown = data['Drawdown'].max()

print("\n✅ Total Bets Placed:", int(total_bets))
print("✅ Total Amount Staked: ${:.2f}".format(total_staked))
print("✅ Total Profit/Loss: ${:.2f}".format(total_profit))
print("🏦 Final Bankroll: ${:.2f}".format(final_bankroll))
print("✅ Avg R-Multiple: {:.4f}".format(average_R))
print("📈 Total Winning R: {:.2f}".format(total_winning_R))
print("📉 Total Losing R: {:.2f}".format(total_losing_R))
print("📉 Max Drawdown: ${:.2f}".format(max_drawdown))

# ✅ Save
data.to_csv('betting_simulation/betting_simulation_kelly_R.csv', index=False)
rejected_data.to_csv('betting_simulation/rejected_bets.csv', index=False)

# ✅ Grouped R-stats
def group_r_stats(df, group_col):
    grouped = df[df['Bet_Placed']].groupby(group_col)
    stats = grouped['R_Multiple'].agg(
        Total_Bets='count',
        Total_R='sum',
        Winning_R=lambda x: x[x > 0].sum(),
        Losing_R=lambda x: x[x < 0].sum()
    )
    return stats

# Odds Bin
data['Odds_Bin'] = pd.cut(data['Odds_To_Use'], bins=[0, 5, 10, 15, 25, 50, 100])
r_stats_odds = group_r_stats(data, 'Odds_Bin')
print("\n📊 R-Metrics by Odds Bin:\n", r_stats_odds)

# Race Class
def clean_class(val):
    try: return int(''.join(filter(str.isdigit, str(val))))
    except: return None

data['Race_Class'] = data['Class'].apply(clean_class) if 'Class' in data.columns else None
if data['Race_Class'].notna().any():
    r_stats_class = group_r_stats(data, 'Race_Class')
    print("\n📊 R-Metrics by Race Class:\n", r_stats_class)

# Field Size
if 'Horse' in data.columns:
    field_sizes = data.groupby('Race_ID')['Horse'].count().rename('Field_Size')
    data = data.merge(field_sizes, on='Race_ID', how='left')
    data['Field_Size_Bin'] = pd.cut(data['Field_Size'], bins=[0, 7, 10, 13, 20], labels=['5–7', '8–10', '11–13', '14+'])
    r_stats_field = group_r_stats(data, 'Field_Size_Bin')
    print("\n📊 R-Metrics by Field Size:\n", r_stats_field)

# Track
if 'Track' in data.columns and data['Track'].notna().any():
    r_stats_track = group_r_stats(data, 'Track').sort_values('Total_R', ascending=False)
    print("\n📊 R-Metrics by Track:\n", r_stats_track)

# ✅ Plot bankroll
data['Race_DateTime'] = pd.to_datetime(data['Date of Race'] + ' ' + data['Time'])

# ✅ Export grouped R-stats to Excel (one sheet per group)
with pd.ExcelWriter('betting_simulation/grouped_r_metrics.xlsx', engine='openpyxl') as writer:
    r_stats_odds.to_excel(writer, sheet_name='By_Odds_Bin')
    if 'Race_Class' in data.columns and data['Race_Class'].notna().any():
        r_stats_class.to_excel(writer, sheet_name='By_Race_Class')
    if 'Field_Size_Bin' in data.columns:
        r_stats_field.to_excel(writer, sheet_name='By_Field_Size')
    if 'Track' in data.columns and data['Track'].notna().any():
        r_stats_track.to_excel(writer, sheet_name='By_Track')


plt.figure(figsize=(12, 6))
plt.plot(data['Race_DateTime'], data['Bankroll_After_Race'], label='Bankroll', marker='o', linewidth=1, markersize=2)
plt.axhline(initial_bankroll, color='gray', linestyle='--', label='Starting Bankroll')
plt.title('📈 Bankroll Over Time (R-Multiple Evaluation)')
plt.xlabel('Date')
plt.ylabel('Bankroll ($)')
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
