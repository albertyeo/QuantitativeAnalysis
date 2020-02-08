#S&P500 Analysis

'''
Analysis of historical S&P 500 index's performance. Constructed two long-short
strategies, based on:
1. Jensen's alpha
2. Momentum 
'''
#------------------------------------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import statsmodels.api as sm
import scipy.stats as stats
#------------------------------------------------------------------------------
#2018 Data

data_list = pd.read_csv('Project_603.csv', header = 0, index_col = 0)
data_team1 = data_list.iloc[0:100, :]
data_2018 = [pd.read_csv('2018/'+i, header = 0, index_col = 0, parse_dates = True).resample('BM').last() for i in data_team1.index]
n = len(data_2018)
#------------------------------------------------------------------------------
#F-F Data

data_french = pd.read_csv('F-F_Research_Data_Factors.csv', header = 0, index_col = 0)
data_french = data_french.apply(lambda x: x/100)
data_french.index = pd.to_datetime(data_french.index, format='%Y%m')
#------------------------------------------------------------------------------
#Stock Data

stocks = [data_2018[i].loc['1999-04-1':'2018-12-31',['Close']] for i in range(n)]
for i in range(n):
    stocks[i]['Simple Return'] = (stocks[i]['Close'] - stocks[i]['Close'].shift(1).values)/(stocks[i]['Close'].shift(1).values)
    stocks[i]['ri-rf'] = stocks[i]['Simple Return'] - data_french.loc['1999-04-1':'2018-12-31', 'RF'].values
    stocks[i]['rm-rf'] = data_french.loc['1999-04-1':'2018-12-31', 'Mkt-RF'].values
    
sp500_return = data_french.loc['1999-05-1':'2018-12-31', 'Mkt-RF'].values + data_french.loc['1999-05-1':'2018-12-31', 'RF'].values
#------------------------------------------------------------------------------
#Linear Regression

def linear_reg(x, y, ind):
    model = [LinearRegression().fit(x, y[i]) for i in range(n)]
    #r_sq = [model[i].score(x, y[i]) for i in range(n)]
    alpha = [model[i].intercept_ for i in range(n)]
    beta = [model[i].coef_[0] for i in range(n)]
    return pd.DataFrame({'Alpha': alpha, 'Beta': beta}, index = ind)
#------------------------------------------------------------------------------
#Stock List
    
x_i = stocks[0]['rm-rf'].iloc[1:].values.reshape((-1, 1))
y_i = [stocks[i]['ri-rf'].iloc[1:].values for i in range(n)]

stocks_list = pd.DataFrame(linear_reg(x_i, y_i, data_team1.index))
stocks_list.sort_values('Alpha', axis=0, ascending=False, inplace=True)
alpha_top10 = stocks_list.iloc[0:10, [0]]
alpha_bot10 = stocks_list.iloc[90:100, [0]]

#Top 10 Alpha
data_top10 = [pd.read_csv('2018/'+i, header = 0, index_col = 0, parse_dates = True).resample('BM').last() for i in alpha_top10.index]
stocks_top10 = [data_top10[i].loc['1999-04-1':'2018-12-31',['Close']] for i in range(10)]
for i in range(10):
    stocks_top10[i]['Simple Return'] = (stocks_top10[i]['Close'] - stocks_top10[i]['Close'].shift(1).values)/(stocks_top10[i]['Close'].shift(1).values)
concat_top10 = pd.concat([pd.Series(i['Simple Return']) for i in stocks_top10], axis=1)
return_top10 = pd.DataFrame(concat_top10.values, index = concat_top10.index, columns = alpha_top10.index).dropna()
return_top10['Long 10'] = return_top10.apply(np.mean, axis=1)

#Bot 10 Alpha
data_bot10 = [pd.read_csv('2018/'+i, header = 0, index_col = 0, parse_dates = True).resample('BM').last() for i in alpha_bot10.index]
stocks_bot10 = [data_bot10[i].loc['1999-04-1':'2018-12-31',['Close']] for i in range(10)]
for i in range(10):
    stocks_bot10[i]['Simple Return'] = (stocks_bot10[i]['Close'] - stocks_bot10[i]['Close'].shift(1).values)/(stocks_bot10[i]['Close'].shift(1).values)    
concat_bot10 = pd.concat([pd.Series(i['Simple Return']) for i in stocks_bot10], axis=1)
return_bot10 = pd.DataFrame(concat_bot10.values, index = concat_bot10.index, columns = alpha_bot10.index).dropna()
return_bot10['Short 10'] = return_bot10.apply(np.mean, axis=1)

#SD for Top 10 Alpha - Bot 10 Alpha
topbot10 = pd.concat([return_top10.iloc[:,0:10], return_bot10.iloc[:,0:10]], axis = 1)
topbot10_cov = topbot10.cov()
top10_w = [0.1]*10
bot10_w = [-0.1]*10
topbot10_w = np.array([*top10_w, *bot10_w])
topbot10_var = topbot10_w.T @ topbot10_cov @ topbot10_w
#------------------------------------------------------------------------------
#Performance Metrix

def perf_metrix(x, y, rp, col, a, b, std):
    reg_model = LinearRegression().fit(x, y)
    #reg_rsquare = reg_model.score(x, y)
    reg_alpha = reg_model.intercept_
    reg_beta = reg_model.coef_[0]
    treynor = np.mean(rp-data_french.loc[a:b, 'RF'].values)/reg_beta
    sharpe = np.mean(rp-data_french.loc[a:b, 'RF'].values)/std
    value = rp-sp500_return
    information = np.mean(value)/np.std(value)
    sv = np.sum((np.minimum((rp-data_french.loc[a:b, 'RF'].values), 0))**2/n)
    sortino = np.mean(rp-data_french.loc[a:b, 'RF'].values)/np.sqrt(sv)
    m2 = np.mean(rp-data_french.loc[a:b, 'RF'].values)*np.std(sp500_return)/std - np.mean(data_french.loc[a:b, 'Mkt-RF'].values)
    mean_return = np.mean(rp)*100
    return pd.DataFrame({'Alpha': reg_alpha, 'Beta': reg_beta, 'Treynor Ratio': treynor, 'Sharpe Ratio': sharpe, 'Information Ratio': information, 'Sortino Ratio': sortino, 'M2': m2, 'Return':mean_return}, index = [col])
#------------------------------------------------------------------------------
#Sharpe Maximization
r_vector = topbot10.apply(np.mean, axis = 0)
R = r_vector.values
x0 = [0.5]*20
R0 = np.mean(sp500_return)
a = '1999-05-1'
b = '2018-12-31'

def sharpe_fun(w, a, b, data, cov):
    variance = w.T @ cov @ w
    rp = data.apply(np.mean, axis=0) @ w
    return -(np.mean(rp-data_french.loc[a:b, 'RF'].values)/np.sqrt(variance))

cons = ({'type':'eq', 'fun': lambda x: sum(x)},
        {'type':'ineq', 'fun': lambda x: x @ R - R0},
        {'type':'ineq', 'fun': lambda x: x.T @ topbot10_cov @ x})
bnds = ((-1,1),) * 20

res = minimize(sharpe_fun, x0, args=(a, b, topbot10, topbot10_cov), bounds = bnds, constraints =cons)

sharpe_w = res.x
sharpe_var = sharpe_w.T @ topbot10_cov @ sharpe_w
#------------------------------------------------------------------------------
#Momentum Strategy

#2018 Data
data_2018_2 = [pd.read_csv('2018/'+i, header = 0, index_col = 0, parse_dates = True) for i in data_team1.index]
n = len(data_2018_2)

#Stock Data
stocks_2 = [data_2018_2[i].loc['1999-04-1':'2018-12-31',['Close']] for i in range(n)]

concat_stocks = pd.concat([pd.Series(i['Close']) for i in stocks_2], axis=1)
price_stocks = pd.DataFrame(concat_stocks.values, index = concat_stocks.index, columns = data_team1.index)

#Linear Regression
def linear_reg(x, y, ind):
    model = [LinearRegression().fit(x, y[i]) for i in range(n)]
    #r_sq = [model[i].score(x, y[i]) for i in range(n)]
    alpha = [model[i].intercept_ for i in range(n)]
    beta = [model[i].coef_[0] for i in range(n)]
    return pd.DataFrame({'Alpha': alpha, 'Beta': beta}, index = ind)

#Define normalized momentum
def momentum(dataDf, period):
    return dataDf.sub(dataDf.shift(period), fill_value=0) / dataDf.iloc[-1]

#Let's load momentum score and returns into separate dataframes
index = price_stocks.index
mscores = pd.DataFrame(index=index,columns=price_stocks.columns)
mscores = momentum(price_stocks, 30)
returns = pd.DataFrame(index=index,columns=price_stocks.columns)
day = 30

# Calculate Forward returns
forward_return_day = 5
returns = price_stocks.shift(-forward_return_day)/price_stocks -1
returns.dropna(inplace = True)

# Calculate correlations between momentum and returns
correlations = pd.DataFrame(index = returns.columns, columns=['Scores','pvalues'])
mscores = mscores[mscores.index.isin(returns.index)]

for i in correlations.index:
    score, pvalue = stats.spearmanr(mscores[i], returns[i])
    correlations['pvalues'].loc[i] = pvalue
    correlations['Scores'].loc[i] = score

correlations.dropna(inplace = True)
correlations.sort_values('Scores', inplace=True)
l = correlations.index.size
plt.figure(figsize=(15,7))
plt.bar(range(1,1+l),correlations['Scores'])
plt.xlabel('Stocks')
plt.xlim((1, l+1))
plt.legend(['Correlation over All Data'])
plt.ylabel('Correlation between %s day Momentum Scores and %s-day forward returns by Stock'%(day,forward_return_day));
plt.show()

correl_scores = pd.DataFrame(index = returns.index.intersection(mscores.index), columns = ['Scores', 'pvalues'])
for i in correl_scores.index:
    score, pvalue = stats.spearmanr(mscores.loc[i], returns.loc[i])
    correl_scores['pvalues'].loc[i] = pvalue
    correl_scores['Scores'].loc[i] = score
correl_scores.dropna(inplace = True)
l = correl_scores.index.size

monthly_mean_correl =correl_scores['Scores'].astype(float).resample('M').mean()
plt.figure(figsize=(15,7))
plt.bar(range(1,len(monthly_mean_correl)+1), monthly_mean_correl)
plt.hlines(np.mean(monthly_mean_correl), 1,len(monthly_mean_correl)+1, colors='r', linestyles='dashed')
plt.xlabel('Month')
plt.xlim((1, len(monthly_mean_correl)+1))
plt.legend(['Mean Correlation over All Data', 'Monthly Rank Correlation'])
plt.ylabel('Rank correlation between %s day Momentum Scores and %s-day forward returns'%(day,forward_return_day));
plt.show()

def compute_basket_returns(factor, forward_returns, number_of_baskets, index):
    data = pd.concat([factor.loc[index],forward_returns.loc[index]], axis=1)
    # Rank the equities on the factor values
    data.columns = ['Factor Value', 'Forward Returns']
    data.sort_values('Factor Value', inplace=True)
    # How many equities per basket
    equities_per_basket = np.floor(len(data.index) / number_of_baskets)
    basket_returns = np.zeros(number_of_baskets)
    # Compute the returns of each basket
    for i in range(number_of_baskets):
        start = i * equities_per_basket
        if i == number_of_baskets - 1:
            # Handle having a few extra in the last basket when our number of equities doesn't divide well
            end = len(data.index) - 1
        else:
            end = i * equities_per_basket + equities_per_basket
        basket_returns[i] = data.iloc[int(start):int(end)]['Forward Returns'].mean()
        
    return basket_returns

number_of_baskets = 10
mean_basket_returns = np.zeros(number_of_baskets)
resampled_scores = mscores.astype(float).resample('2D').last()
resampled_prices = price_stocks.astype(float).resample('2D').last()
resampled_scores.dropna(inplace=True)
resampled_prices.dropna(inplace=True)
forward_returns = resampled_prices.shift(-1)/resampled_prices -1
forward_returns.dropna(inplace = True)
for m in forward_returns.index.intersection(resampled_scores.index):
    basket_returns = compute_basket_returns(resampled_scores, forward_returns, number_of_baskets, m)
    mean_basket_returns += basket_returns
mean_basket_returns /= l    

plt.figure(figsize=(15,7))
plt.bar(range(number_of_baskets), mean_basket_returns)
plt.ylabel('Returns')
plt.xlabel('Basket')
plt.legend(['Returns of Each Basket'])
plt.show()

total_months = mscores.resample('M').last().index
months_to_plot = len(total_months)
monthly_index = total_months[:months_to_plot+1]
mean_basket_returns = np.zeros(number_of_baskets)
strategy_returns = pd.Series(index = monthly_index)

top_corr=[]
bot_corr=[]
for month in range(1, monthly_index.size):
    temp_returns = forward_returns.loc[monthly_index[month-1]:monthly_index[month]]
    temp_scores = resampled_scores.loc[monthly_index[month-1]:monthly_index[month]]
    for m in temp_returns.index.intersection(temp_scores.index):
        basket_returns = compute_basket_returns(temp_scores, temp_returns, number_of_baskets, m)
        mean_basket_returns += basket_returns
    
    strategy_returns[monthly_index[month-1]] = mean_basket_returns[0] - mean_basket_returns[ number_of_baskets-1]
    top_corr.append(mean_basket_returns[0])
    bot_corr.append(mean_basket_returns[ number_of_baskets-1])
    mean_basket_returns /= temp_returns.index.intersection(temp_scores.index).size
    
plt.figure(figsize=(15,7))
plt.plot(strategy_returns, color ='C8')
plt.ylabel('Returns')
plt.xlabel('Year')
plt.plot(strategy_returns.cumsum(), color ='C7')
plt.title('Time Series of Portfolio 3')
plt.legend(['Monthly Strategy Returns','Cumulative Strategy Returns'])
plt.show()

total_return = strategy_returns.sum()
ann_return = 100*((1 + total_return)**(12.0 /float(strategy_returns.index.size))-1)
print('Annual Returns: %.2f%%'%ann_return)
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
momentum_corr = strategy_returns.shift(periods = 1)
momentum_corr = momentum_corr.dropna()
momentum_corr = pd.DataFrame(momentum_corr, columns = ['Simple Return'])
momentum_corr['Top10 Corr'] = top_corr
momentum_corr['Bot10 Corr'] = bot_corr

topbot10corr = momentum_corr.iloc[:,1:3]
topbot10corr_cov = topbot10corr.cov()
top10corr_w = [1]
bot10corr_w = [-1]
topbot10corr_w = np.array([*top10corr_w, *bot10corr_w])
topbot10corr_var = topbot10corr_w.T @ topbot10corr_cov @ topbot10corr_w
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Portfolio Return
result = pd.DataFrame(return_top10['Long 10']-return_bot10['Short 10'], columns = ['Top10Bot10 Alpha'])
result['Top10Bot10 ri-rf'] = result['Top10Bot10 Alpha'] - data_french.loc['1999-05-1':'2018-12-31', 'RF'].values

result['Max Sharpe'] = topbot10 @ res.x
result['Max Sharpe ri-rf'] = result['Max Sharpe'] - data_french.loc['1999-05-1':'2018-12-31', 'RF'].values

result['Top10Bot10Corr'] = momentum_corr.iloc[:,0].values
result['Top10Bot10Corr ri-rf'] = result['Top10Bot10Corr'] - data_french.loc['1999-05-1':'2018-12-31', 'RF'].values

result['S&P 500'] = sp500_return
result['S&P 500 ri-rf'] = sp500_return - data_french.loc['1999-05-1':'2018-12-31', 'RF'].values

#Portfolio Performance
performance = pd.DataFrame(perf_metrix(x_i, result['Top10Bot10 ri-rf'].values, result['Top10Bot10 Alpha'], result.columns[0], '1999-05-1','2018-12-31', np.sqrt(topbot10_var)))
performance = performance.append(perf_metrix(x_i, x_i, sp500_return, 'S&P500', '1999-05-1','2018-12-31', np.std(sp500_return)))
performance = performance.append(perf_metrix(x_i, result['Max Sharpe ri-rf'].values, result['Max Sharpe'], result.columns[2], '1999-05-1','2018-12-31', np.sqrt(sharpe_var)))
performance = performance.append(perf_metrix(x_i, result['Top10Bot10Corr ri-rf'].values, result['Top10Bot10Corr'], result.columns[4], '1999-05-1','2018-12-31', np.sqrt(topbot10corr_var)))
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Time Series of Price
SP500 = pd.read_csv('^GSPC.csv', header = 0, index_col = 0, parse_dates = True).resample('BM').last()
SP500 = SP500.loc['1999-04-1':'2018-12-31',:]

stocks_top10 = [data_top10[i].loc['1999-04-1':'2018-12-31',['Close']] for i in range(10)]
stocks_bot10 = [data_bot10[i].loc['1999-04-1':'2018-12-31',['Close']] for i in range(10)]

concat_top10_c = pd.concat([pd.Series(i['Close']) for i in stocks_top10], axis=1)
concat_bot10_c = pd.concat([pd.Series(i['Close']) for i in stocks_bot10], axis=1)

price_top10 = pd.DataFrame(concat_top10_c.values, index = concat_top10_c.index, columns = alpha_top10.index)
price_bot10 = pd.DataFrame(concat_bot10_c.values, index = concat_bot10_c.index, columns = alpha_bot10.index)
price_top10bot10 = pd.concat([price_top10, price_bot10], axis = 1)
price_top10bot10['Portfolio1']=price_top10bot10@topbot10_w
price_top10bot10['Portfolio2']=price_top10bot10.iloc[:,0:20]@sharpe_w

price_top10bot10['ave1'] = price_top10bot10.Portfolio1/np.mean(price_top10bot10.Portfolio1)
SP500['ave'] = SP500.Close/np.mean(SP500.Close)
ax1 = price_top10bot10.ave1.plot(label='Portfolio 1', color='C0')
ax1 = SP500.ave.plot(label='S&P500', color='C1')
ax1.set_ylabel('Normalized Price')
ax1.set_xlabel('Year')
ax1.set_title('Time Series of Portfolio 1 and S&P500 Index')
ax1.autoscale(enable=True, axis='both', tight=True)
ax1.legend()
plt.show()

price_top10bot10['ave2'] = price_top10bot10.Portfolio2/np.mean(price_top10bot10.Portfolio2)
ax2 = price_top10bot10.ave2.plot(label='Portfolio 2', color='C2')
ax2 = SP500.ave.plot(label='S&P500', color='C1')
ax2.set_ylabel('Normalized Price')
ax2.set_xlabel('Year')
ax2.set_title('Time Series of Portfolio 2 and S&P500 Index')
ax2.autoscale(enable=True, axis='both', tight=True)
ax2.legend()
plt.show()

ax3 = price_top10bot10.ave1.plot(label='Portfolio 1', color='C0')
ax3 = price_top10bot10.ave2.plot(label='Portfolio 2', color='C2')
ax3 = SP500.ave.plot(label='S&P500', color='C1')
ax3.set_ylabel('Normalized Price')
ax3.set_xlabel('Year')
ax3.set_title('Time Series of Portfolio 1, 2, and S&P500 Index')
ax3.autoscale(enable=True, axis='both', tight=True)
ax3.legend()
plt.show()
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Times Series of Returns
ax4 = result['Top10Bot10 Alpha'].plot(label='Portfolio 1', color='C0')
ax4 = result['S&P 500'].plot(label='S&P500', color ='C1')
ax4.set_ylabel('Monthly Returns')
ax4.set_xlabel('Year')
ax4.set_title('Time Series of Portfolio 1 and S&P500 Index')
ax4.autoscale(enable=True, axis='both', tight=True)
ax4.legend(loc='upper right')
plt.show()

ax5 = result['Max Sharpe'].plot(label='Portfolio 2', color ='C2')
ax5 = result['S&P 500'].plot(label='S&P500', color = 'C1')
ax5.set_ylabel('Monthly Returns')
ax5.set_xlabel('Year')
ax5.set_title('Time Series of Portfolio 2 and S&P500 Index')
ax5.autoscale(enable=True, axis='both', tight=True)
ax5.legend(loc='upper right')
plt.show()

ax6 = result['Top10Bot10 Alpha'].plot(label='Portfolio 1', color='C0')
ax6 = result['Max Sharpe'].plot(label='Portfolio 2', color='C2')
ax6 = result['S&P 500'].plot(label='S&P500', color='C1')
ax6.set_ylabel('Monthly Returns')
ax6.set_xlabel('Year')
ax6.set_title('Time Series of Portfolio 1, 2, and S&P500 Index')
ax6.autoscale(enable=True, axis='both', tight=True)
ax6.legend()
plt.show()

ax7 = result['Top10Bot10Corr'].plot(label='Portfolio 3', color='C7')
ax7 = result['S&P 500'].plot(label='S&P500', color='C1')
ax7.set_ylabel('Monthly Returns')
ax7.set_xlabel('Year')
ax7.set_title('Time Series of Portfolio 3 and S&P500 Index')
ax7.autoscale(enable=True, axis='both', tight=True)
ax7.legend()
plt.show()
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Bar Charts
bar_label = ['S01','S02','S03','S04','S05','S06','S07','S08','S09','S10','S91','S92','S93','S94','S95','S96','S97','S98', 'S99','S100']

p1_weight_data = pd.DataFrame(topbot10_w, columns = ['Portfolio 1'], index=bar_label)
ax7 = p1_weight_data.plot.bar(title = 'Stock Weightage of Portfolio 1', color='C0')
ax7.set_xlabel('Stock Name')
ax7.set_ylabel('Stock Weight')
plt.show()

p2_weight_data = pd.DataFrame(sharpe_w, columns = ['Portfolio 2'], index=bar_label)
ax8 = p2_weight_data.plot.bar(title = 'Stock Weightage of Portfolio 2', color='C2')
ax8.set_xlabel('Stock Name')
ax8.set_ylabel('Stock Weight')
plt.show()

weight_data = pd.DataFrame(np.c_[topbot10_w,sharpe_w], columns = ['Portfolio 1', 'Portfolio 2'], index=bar_label)
ax9 = weight_data.plot.bar(title = 'Stock Weightage of Portfolio 1 and 2', color=['C0', 'C2'])
ax9.set_xlabel('Stock Name')
ax9.set_ylabel('Stock Weight')
plt.show()
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
plt.figure(figsize=(15,7))
plt.plot(result['S&P 500'], color='C5')
plt.ylabel('Returns')
plt.xlabel('Year')
plt.plot(result['S&P 500'].cumsum(), color='C1')
plt.title('Time Series of S&P 500')
plt.legend(['Monthly S&P 500 Returns','Cumulative S&P 500 Returns'])
plt.show()

total_returnsp = result['S&P 500'].sum()
ann_return_sp = 100*((1 + total_returnsp)**(12.0 /float(strategy_returns.index.size))-1)
print('Annual Returns: %.2f%%'%ann_return_sp)

plt.figure(figsize=(15,7))
plt.plot(result['S&P 500'], color='C5')
plt.ylabel('Returns')
plt.xlabel('Year')
plt.plot(result['S&P 500'].cumsum(), color='C1')
plt.title('Time Series of Portfolio 3 and S&P 500')
plt.plot(momentum_corr.iloc[:,0], color='C8')
plt.plot(momentum_corr.iloc[:,0].cumsum(), color='C7')
plt.legend(['Monthly S&P 500 Returns','Cumulative S&P 500 Returns', 'Monthly Strategy Returns','Cumulative Strategy Returns'])
plt.show()

print(result)
print(performance)

x_reg = result['S&P 500 ri-rf'].values
x_reg = sm.add_constant(x_reg)
y_reg1 = result['Top10Bot10 ri-rf'].values

model_1 = sm.OLS(y_reg1, x_reg)
results_1 = model_1.fit()
print(results_1.summary())

y_reg2 = result['Max Sharpe ri-rf'].values
model_2 = sm.OLS(y_reg2, x_reg)
results_2 = model_2.fit()
print(results_2.summary())

y_reg3 = result['Top10Bot10Corr ri-rf'].values
model_3 = sm.OLS(y_reg3, x_reg)
results_3 = model_3.fit()
print(results_3.summary())

print(performance.iloc[:,2:5])
print(performance.iloc[:,5:])
