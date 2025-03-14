def print_imports():
    """Prints import statements for all installed libraries."""
    imports = [
        "# Core Libraries",
        "import numpy as np",
        "import pandas as pd",
        "import scipy",
        "import sklearn",
        
        "# Data Preprocessing",
        "from sklearn.model_selection import train_test_split",
        "from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder",
        "from sklearn.impute import SimpleImputer",
        
        "# Machine Learning Models",
        "from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso",
        "from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor",
        "from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier",
        "from sklearn.svm import SVC, SVR",
        "from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor",
        "from sklearn.naive_bayes import GaussianNB",
        "from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering",
        
        "# Model Evaluation",
        "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_squared_error",
        
        "# NLP",
        "import nltk",
        "import spacy",
        
        "# Visualization",
        "import plotly.express as px",
        "import matplotlib.pyplot as plt",
        "import seaborn as sns",
        
        "# Deep Learning",
        "import tensorflow as tf",
        "import torch",
        
        "# Data Manipulation & Querying",
        "import pandasql",
        "import polars",
        
        "# Advanced ML Models",
        "import xgboost as xgb",
        "import lightgbm as lgb",
        "import catboost as cb",
        
        "# Web Frameworks & APIs",
        "import fastapi",
        "import streamlit",
        "import flask",
        
        "# Web Scraping & Automation",
        "import selenium",
        "import beautifulsoup4 as bs4",
        
        "# Databases & SQL",
        "import mysql.connector",
        "import pyodbc",
        "import sqlalchemy",
        
        "# Utility Libraries",
        "import tqdm",
        "import openpyxl",
        "import pyarrow",
        "import yaml",
        "import networkx",
    ]

    print("\n🚀 **MLEssentials Installed Successfully!**\n")
    print("✅ To use the installed libraries, copy-paste these imports:\n")
    print("\n".join(imports))
    print("\n🎯 Happy Coding! 🚀\n")


if __name__ == "__main__":
    print_imports()
