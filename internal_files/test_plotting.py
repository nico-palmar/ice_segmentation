import seaborn as sns 
import pandas as pd

dict = {"Val 1": [1, 2, 3, 4], "Val 2": [7, 8, 9, 10]}
info_df = pd.DataFrame(dict)
sns.set_style('darkgrid')
test_ax = sns.lineplot(data=info_df[['Val 1', 'Val 2']])
test_ax.set_title('testing_ax')
test_ax.figure.savefig('testing_ax.png')
print("Done")