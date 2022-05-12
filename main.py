import os
import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns
sns.set_theme(style='darkgrid')

def main():
    dataset = pd.read_csv(os.path.join('dataset', 'SAT to GPA.csv'))
    print(dataset.head())
    plot = sns.relplot(x = 'GPA', y = 'SAT Score', data=dataset)
    plot.fig.savefig('out.png')

if __name__ == "__main__":
    main()