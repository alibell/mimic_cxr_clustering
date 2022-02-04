"""
    This script download a sample of radiography data from MIMIC CRX and generate a compressed lower resolution of them
"""

#%%
import wget

#%%
# Parameters
p_sample = 0.01
random_seed = 42

# %%
# Getting the data file list
