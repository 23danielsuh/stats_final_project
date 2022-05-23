import os
import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns

#for ml model training
import torch

sns.set_theme(style='darkgrid')

dataset = pd.read_csv(os.path.join('dataset', 'SAT to GPA.csv'))

def preprocess_data():
    print('Preprocessing Data')


def train_model():
    print('Training Model')


def visualize_data():
    print(dataset.head())
    plot = sns.relplot(x = 'GPA', y = 'SAT Score', data=dataset)
    plot.fig.savefig('out.png')


def main():
    print('Initiating Program')
    visualize_data()
    preprocess_data()
    train_model()


if __name__ == "__main__":
    main()
