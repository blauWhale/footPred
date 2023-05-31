import pandas as pd
df = pd.read_html('https://fbref.com/en/comps/19/schedule/Europa-League-Scores-and-Fixtures')
df[0].to_csv('data.csv', index=False)
