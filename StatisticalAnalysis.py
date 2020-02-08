#Statistical Analysis

'''
Statistical analysis of log returns of DJIA and S&P500 index and hypothesis 
testing:
- Skewness
- Kurtosis
- Correlation
- Coefficient of determination
- Adjusted coefficient of determination
- F-test
- Jarque-Bera test statistic
- Durbin-Watson statistic
'''
#------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import scipy.stats as ss
from sklearn.linear_model import LinearRegression
#------------------------------------------------------------------------------
#Practical Task 1

DJIA = pd.read_csv('^DJI.csv',index_col=0, parse_dates=True)
ax1 = DJIA.Close.plot()
ax1.set_ylabel('Price')
ax1.set_title('Time Series of DJIA Index')
ax1.autoscale(enable=True, axis='x', tight=True)
ax1.autoscale(enable=True, axis='y', tight=True)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
plt.show()

SP500 = pd.read_csv('^GSPC.csv',index_col=0, parse_dates=True)
ax2 = SP500.Close.plot()
ax2.set_ylabel('Price')
ax2.set_title('Time Series of SP500 Index')
ax2.autoscale(enable=True, axis='x', tight=True)
ax2.autoscale(enable=True, axis='y', tight=True)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
plt.show()

DJIA['log_ret'] = np.log(DJIA.Close) - np.log(DJIA.Close.shift(1))
SP500['log_ret'] = np.log(SP500.Close) - np.log(SP500.Close.shift(1))
print('### DJIA Daily Log Return ###')
print(DJIA['log_ret'])
print('### SP500 Daily Log Return ###')
print(SP500['log_ret'])

ax3 = DJIA['log_ret'].plot()
ax3.set_ylabel('Log Return')
ax3.set_title('Time Series of DJIA Index Log Return')
ax3.autoscale(enable=True, axis='x', tight=True)
ax3.autoscale(enable=True, axis='y', tight=True)
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
plt.show()

ax4 = SP500['log_ret'].plot()
ax4.set_ylabel('Log Return')
ax4.set_title('Time Series of SP500 Index Log Return')
ax4.autoscale(enable=True, axis='x', tight=True)
ax4.autoscale(enable=True, axis='y', tight=True)
ax4.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
plt.show()

DJIA_m=np.mean(DJIA.log_ret)
DJIA_s2=np.var(DJIA.log_ret,ddof=1)
print('DJIA Sample Mean = ', DJIA_m)
print('DJIA Sample Variance = ', DJIA_s2)
SP500_m=np.mean(SP500.log_ret)
SP500_s2=np.var(SP500.log_ret,ddof=1)
print('S&P500 Sample Mean = ', SP500_m)
print('S&P500 Sample Variance = ', SP500_s2)

DJIA_annave=round(252*DJIA_m*100,2)
DJIA_annvol=round(np.sqrt(252*DJIA_s2)*100,2)
print('DJIA Annualized Average = ',DJIA_annave,'%')
print('DJIA Annualized Volatility = ',DJIA_annvol,'%')
SP500_annave=round(252*SP500_m*100,2)
SP500_annvol=round(np.sqrt(252*SP500_s2)*100,2)
print('S&P500 Annualized Average = ',SP500_annave,'%')
print('S&P500 Annualized Volatility = ',SP500_annvol,'%')

DJIA_sk=ss.skew(DJIA['log_ret'][1:].values)
print('DJIA Sample Skewness = ',DJIA_sk)
DJIA_k=ss.kurtosis(DJIA['log_ret'][1:].values,fisher=False)
print('DJIA Sample Kurtosis = ',DJIA_k)
SP500_sk=ss.skew(SP500['log_ret'][1:].values)
print('S&P500 Sample Skewness = ',SP500_sk)
SP500_k=ss.kurtosis(SP500['log_ret'][1:].values,fisher=False)
print('S&P500 Sample Kurtosis = ',SP500_k)

DJIA_JB=ss.jarque_bera(DJIA['log_ret'][1:].values)
print('DJIA JB Statistic = ',DJIA_JB[0])
print('DJIA JB Test p-value = ',DJIA_JB[1])
print('Comments: Therefore we reject the null hypothesis of JB=0. This shows 
that DJIA index does not follow Normal Distribution.')
SP500_JB=ss.jarque_bera(SP500['log_ret'][1:].values)
print('S&P500 JB Statistic = ',SP500_JB[0])
print('S&P500 JB Test p-value = ',SP500_JB[1])
print('Comments: Therefore we reject the null hypothesis of JB=0. This shows 
that S&P500 index does not follow Normal Distribution.')
#------------------------------------------------------------------------------
DJIA['ave'] = DJIA.Close/np.mean(DJIA.Close)
SP500['ave'] = SP500.Close/np.mean(SP500.Close)
ax5 = DJIA.ave.plot(label='DJIA')
ax5 = SP500.ave.plot(label='SP500')
ax5.set_ylabel('Normalized Price')
ax5.set_title('Time Series of DJIA & SP500 Index')
ax5.autoscale(enable=True, axis='both', tight=True)
ax5.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
ax5.legend()
plt.show()

Corr=ss.pearsonr(DJIA['log_ret'][1:].values,SP500['log_ret'][1:].values)
print("Pearson's Correlation between the log returns of DJIA & S&P500
indexes = ", Corr[0])

N1=len(DJIA.index)-1
N2=len(SP500.index)-1
m1=DJIA_m
m2=SP500_m
s1=DJIA_s2
s2=SP500_s2
t=(m1-m2)/np.sqrt(s1/N1+s2/N2)
df=(s1/N1+s2/N2)**2/((s1/N1)**2/(N1-1)+(s2/N2)**2/(N2-1))

#Test Scenario: Two samples have equal mean
print('Test Scenario: Two samples have equal mean')
p=(1-ss.t.cdf(t,df))*2
cv_l=ss.t.ppf(.025,df)
cv_h=ss.t.ppf(.975,df)
print('t = ',t)
print('p-value =',p)
print('critical-value = ',cv_l,'and ', cv_h)
print('Comments: Since -1.96 < t = 0.14 < 1.96, we do not have sufficient 
evidence to reject the null hypothesis. Hence the two samples have equal mean 
at 5% level of significance.')

#Test Scenario: Two samples have equal variances
print('Test Scenario: Two samples have equal variances')
F=s1/s2
p_F=ss.f.cdf(F,N1-1,N2-1)*2
cvF_l=ss.f.ppf(.025,N1-1,N2-1)
cvF_h=ss.f.ppf(.975,N1-1,N2-1)
print('F =', F)
print('p-value =',p_F)
print('critical-value = ',cvF_l,'and ', cvF_h)
print('Comments: Since F = 0.956 < 0.959, we have sufficient evidence to reject 
the null hypothesis. Hence the two samples do not equal variance at 5% level 
of significance.')
#------------------------------------------------------------------------------
x=SP500['log_ret'][1:].values.reshape((-1,1))
y=DJIA['log_ret'][1:].values
model=LinearRegression().fit(x,y)
a_hat=model.intercept_
b_hat=model.coef_[0]
print('intercept, a_hat:', a_hat)
print('slope, b_hat:', b_hat)

y_hat=a_hat+b_hat*x
e=y.reshape((-1,1))-y_hat
SSR=np.sum(e**2)/(len(e)-2)
sigma_u=np.sqrt(SSR)
print('Residual Variance, sigma_u:',sigma_u)

se_a=sigma_u*np.sqrt(1/len(e)+m2**2/(s2*(N2-1)))
se_b=sigma_u*np.sqrt(1/(s2*(N2-1)))
t_a=a_hat/se_a
t_b=b_hat/se_b
df_a=N2-2
df_b=N2-2

#Test Scenario: a_hat = 0
print('Test Scenario: a_hat = 0')
p_a=(1-ss.t.cdf(t_a,df_a))*2
cva_l=ss.t.ppf(.025,df_a)
cva_h=ss.t.ppf(.975,df_a)
print('t = ',t_a)
print('p-value =',p_a)
print('critical-value = ',cva_l,'and ', cva_h)
print('Comments: Since -1.96 < t = 1.38 < 1.96, we do not have sufficient 
evidence to reject the null hypothesis. Hence the intercept, a_hat = 0 at 5% 
level of significance.')
#Test Scenario: b_hat = 0
print('Test Scenario: b_hat = 0')
p_b=(1-ss.t.cdf(t_b,df_b))*2
cvb_l=ss.t.ppf(.025,df_b)
cvb_h=ss.t.ppf(.975,df_b)
print('t = ',t_b)
print('p-value =',p_b)
print('critical-value = ',cvb_l,'and ', cvb_h)
print('Comments: Since t = 345.95 > 1.96, we have sufficient evidence to reject 
the null hypothesis. Hence the slope, b_hat does not equal to zero at 5% level 
of significance.')
 
R_sq=np.sum((y_hat-m1)**2)/np.sum((y-m1)**2)
print('R_squared = ', R_sq)
Ra_sq=1-((N2-1)/(N2-2)*(1-R_sq))
print('Adjusted R_squared = ', Ra_sq)

e_JB=ss.jarque_bera(e)
print('Residuals JB Statistic = ',e_JB[0])
print('Residuals JB Test p-value = ',e_JB[1])
print('Comments: Therefore we reject the null hypothesis of JB=0. This shows 
that residuals do not follow Normal Distribution.')

DW = (np.sum((e[1:]-e[:-1])**2))/np.sum(e*e)
phi=np.sum(e[1:]*e[:-1])/np.sum(e*e)
st=(np.sum(e[1:]**2))/np.sum(e*e)
nd=(np.sum(e[:-1]**2))/np.sum(e*e)
print('1st term = ',st)
print('2nd term = ',nd)
print('phi =', phi)
print('Durbin Watson Test = ',DW)
print('Comments: Since phi is close to 0, DW is close to 2, we do not reject 
the null hypothesis. Hence we can say that there is no evidence of autocorrelation 
on the residuals.')
