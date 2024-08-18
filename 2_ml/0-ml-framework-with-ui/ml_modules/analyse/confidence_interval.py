import numpy as np
import pandas as pd
from scipy import stats


education_districtwise = pd.read_csv("education_districtwise.csv")
education_districtwise = education_districtwise.dropna()
sampled_data = education_districtwise.sample(n=50, replace=True, random_state=31208)
sampled_data
sample_mean = sampled_data['OVERALL_LI'].mean()
estimated_standard_error = sampled_data['OVERALL_LI'].std() / np.sqrt(sampled_data.shape[0])


stats.norm.interval(alpha=0.95, loc=sample_mean, scale=estimated_standard_error)
stats.norm.interval(alpha=0.99, loc=sample_mean, scale=estimated_standard_error)

