import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

#load  the data table from task 2
data_set = pd.read_csv('data.csv')

#Compute means band power per subject.
#Each subject should have one representative value per band (or per selected channel/region).
patient_mean = data_set.groupby(['patient','group']).mean().reset_index()
print(patient_mean.head())

#Separate the AD and the control groups
#first the ad group
ad_class = patient_mean[patient_mean['class'] == 'AD']
#then the control group
control_class = patient_mean[patient_mean['class'] == 'Control']

#box plot of the Alpha power comparison
plt.boxplot([ad_class ['Alpha'], control_class['Alpha']], labels=['AD', 'Control'])

plt.title('Alpha power (AD vs Control)')
plt.show()

#Box plot of Theta power comparison
plt.boxplot([ad_class ['Theta'], control_class['Theta']], labels=['AD', 'Control'])

plt.title('Theta Power (AD vs Control)')
plt.show()

#Perform statistical testing (report simple difference in means or t-test): Report your observations.
#choosing the T-test
#T-test for the alpha band, alp_t = t value for the alpha band, alp_p = the p value for the alpha band
alp_t, alp_p = ttest_ind(ad_class['Alpha'], control_class['Alpha'])

#print results for the alpha t-test
print("The Alpha T-test results: t value = ", alp_t, ", p value = ", alp_p)

#T-test for the theta band. th_t = t-value for theta, th_p = p value for theta
th_t, th_p = ttest_ind(ad_class['Theta'], control_class['Theta'])

#print the results for the theta t-test
print("The Theta T-test results: t value = ", th_t, ", p value = ", th_p)
