import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Import data
df = pd.read_csv(
    'medical_examination.csv',
    header=0, # first row has headers
    index_col=[0], # 'id' column
    sep=',',
    decimal='.',
    na_values=["", "?", "-", None])

# Add 'overweight' column
height_in_meters = df['height']/100
bmi = df['weight'] / (height_in_meters ** 2)
maskNotOverweight = bmi <= 25
maskOverweight = bmi > 25
df.loc[maskNotOverweight, 'overweight'] = 0
df.loc[maskOverweight, 'overweight'] = 1

# Normalize data by making 0 always good and 1 always bad. If the value of 'cholesterol' or 'gluc' is 1, make the value 0. If the value is more than 1, make the value 1.
maskCholOne = df['cholesterol'] == 1  # if cholesterol == 1, then ZERO
maskCholGreaterThanOne = df['cholesterol'] > 1  # if cholesterol > 1, then ONE
df.loc[maskCholOne, 'cholesterol'] = 0
df.loc[maskCholGreaterThanOne, 'cholesterol'] = 1

maskGlucOne = df['gluc'] == 1  # if glucose == 1, then ZERO
maskGlucGreaterThanOne = df['gluc'] > 1  # if glucose > 1, then ONE
df.loc[maskGlucOne, 'gluc'] = 0
df.loc[maskGlucGreaterThanOne, 'gluc'] = 1


# Draw Categorical Plot
def draw_cat_plot():
    # Create DataFrame for cat plot using `pd.melt` using just the values from 'cholesterol', 'gluc', 'smoke', 'alco', 'active', and 'overweight'.
    id = ['cardio']
    cols = ['cholesterol','gluc','smoke','alco','active', 'overweight']
    df_cat = pd.melt(df, id_vars=id, value_vars=cols)
  
    # Group and reformat the data to split it by 'cardio'. Show the counts of each feature. You will have to rename one of the columns for the catplot to work correctly.
    df_cat = df_cat.value_counts().to_frame()
    df_cat.reset_index(inplace=True)
    df_cat.rename(columns={0:'total'}, inplace=True)
    df_cat.sort_values(by=['variable'], inplace=True) # dataframe sorted alphabetically by 'variable' column
    
    # Draw the catplot with 'sns.catplot()'
    cat_grid = sns.catplot(data=df_cat, kind='bar', height=5, aspect=1.0, col='cardio', hue='value', x='variable', y='total')
  
    # Get the figure for the output
    fig = cat_grid.fig
  
    # Do not modify the next two lines
    fig.savefig('catplot.png')
    return fig


# Draw Heat Map
def draw_heat_map():
    # Clean the data
    df2 = df.copy()
    df_heat = df2[ (df2['ap_lo'] <= df2['ap_hi']) & ((df2['height'] >= df2['height'].quantile(.025))&(df2['height'] <= df2['height'].quantile(.975))) &
  ((df2['weight'] >= df2['weight'].quantile(.025))&(df2['weight'] <= df2['weight'].quantile(.975))) ]

    # gets "id" column back (from index)
    df_heat.reset_index(inplace=True)

    # Calculate the correlation matrix
    corr = df_heat.corr()

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(13, 10))
  
    # Draw the heatmap with 'sns.heatmap()'
    heatmap_axes = sns.heatmap(corr, mask=mask, vmin=-0.16, vmax=0.32, square=True, linewidths=.7, center=0, annot=True, fmt='.1f', cbar_kws={ "orientation": "vertical", "shrink": 0.5, "spacing": "proportional", "ticks": np.arange(-0.08, 0.25, 0.08)}, cmap="icefire")
  
    # sets the ticks "upright" (0) and sideways (90)
    plt.yticks(rotation=0)
    plt.xticks(rotation=90)
  
    # sets the tick marks "-" to appear on the left and on the bottom of the plot
    ax.tick_params(left=True, bottom=True)

    fig = heatmap_axes.figure

    # Do not modify the next two lines
    fig.savefig('heatmap.png')
    return fig
