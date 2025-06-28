from multiprocessing import pool
import numpy as np 
import matplotlib.pyplot as plt 
import random 
import math 
import json
import scipy.stats as stats  

def writeJson(data, filename):
    with open(filename,'w',encoding='utf-8') as f:
        json.dump(data,f,ensure_ascii=False,indent=4)




def allocateTransaction(transaction_value,transaction_size,block_limit,strategy,gamma):
    '''
        We are assuming the transaction value is sorted according in increasing order 
    '''
    assert(transaction_value.shape == transaction_size.shape)
    assert(gamma != 0)
    block_limit = (block_limit*np.sum(transaction_size))/1000
    transaction_value = np.sort(transaction_value)
    included_transaction = np.zeros(transaction_size.shape)
    n = transaction_size.shape[0]
    index = -1
    if strategy == 'greedy':
        csize = 0
        while csize < block_limit and index >= -n:
            included_transaction[index] = 1
            csize += transaction_size[index]
            index -= 1

    elif strategy == 'softmax':
        csize = 0
        iterNo = 0
        index = -1
        sampling_probability = np.exp(transaction_value/gamma)
        while csize <= block_limit and iterNo <= 100*n and np.sum(sampling_probability) > 0:
            # normalize sampling probability
    #        try:
            # print(sampling_probability)
            sampling_probability /= np.sum(sampling_probability) 
     #       except:
      #          print(np.sum(sampling_probability),'Err')
       #         exit()

            '''
            if sampling_probability[index] != 0 and csize + transaction_size[index] <= block_limit:
                if np.random.random() <= sampling_probability[index]:
                    included_transaction[index] = 1
                    csize += transaction_size[index]
                    # removing index from possible smaping set 
                    sampling_probability[index] = 0
            '''
            randomIndex = np.random.choice(range(n),size=(1),replace=False,p=sampling_probability)
            included_transaction[randomIndex] = 1
            sampling_probability[randomIndex] = 0
            csize = csize + transaction_size[index]
            for tindex in range(n):
                if sampling_probability[tindex] != 0 and csize + transaction_size[tindex] > block_limit:
                    sampling_probability[tindex] = 0

            iterNo += 1
            index -= 1
            if index + n < 0:
                index = -1
    return included_transaction


def getTransactionPool(transaction_count:int, zero_frac,distribution,distribution_params):
    assert(zero_frac <= 1)
    assert(transaction_count >= 0)
    assert(distribution in ['normal','chi2','exponential','uniform'])

    zero_fee_count = int(zero_frac*transaction_count)
    normal_transaction_count = int((1 - zero_frac)*transaction_count)

    if distribution is 'exponential':
        lambda_inverse = distribution_params[0]
        transaction_fees = np.random.exponential(lambda_inverse,normal_transaction_count)
    elif distribution is 'uniform':
        max_txf = distribution_params[0]
        min_txf = 0
        transaction_fees = np.random.uniform(min_txf,max_txf,normal_transaction_count)        
    elif distribution is 'normal':
        mean = distribution_params[0]
        sd = distribution_params[0]
        # transaction_fees = np.random.normal(mean,sd,normal_transaction_count)
        transaction_fees = stats.truncnorm((0-5)/3,(10-5)/3,loc=5,scale=3).rvs(normal_transaction_count)

    # do for normal and chi2

    transaction_size = np.random.exponential(1,transaction_count)
    
    if zero_fee_count == 0:
        return transaction_fees,transaction_size
    else:
        return np.concatenate((np.zeros(zero_fee_count),transaction_fees),axis=0),transaction_size

def calculateRevenue(allocatedTeansactionList,revenueStrategy):
    assert(revenueStrategy in ['EIP-1559-rb','FPA'])
    BURN_FRAC = 0.1
    revene = np.sum(allocatedTeansactionList)
    if revenueStrategy == 'EIP-1559-rb':
        revenue *= (1 - BURN_FRAC)
    return revene

def zeroFracFind(allocatedTxVector,transaction_value,transaction_size,blksize):
    threshold = 0.5
    count = 0
    tcount = blksize
    for i in range(allocatedTxVector.shape[0]):
        if transaction_value[i] <= threshold:
            if allocatedTxVector[i] == 1:
                count += transaction_size[i]
    return count/tcount

def simulate(nturns,gamma_max,gamma_count,pool_size,block_size,tx_sampling_scheme,revenueScheme):
    if tx_sampling_scheme is 'uniform':
        tx_value,tx_size = getTransactionPool(pool_size,0.2,'uniform',[5])
    elif tx_sampling_scheme is 'exponential':
        tx_value,tx_size = getTransactionPool(pool_size,0.2,'exponential',[1])
    elif tx_sampling_scheme is 'normal':
        tx_value, tx_size = getTransactionPool(pool_size,0.2,'normal',[5,4])
    tx_value = np.sort(tx_value)
    # possibleGamma = np.linspace(0.1,gamma_max+0.1,gamma_count)
    possibleGamma = np.array([0.1,0.25,0.50,0.75,1,2.5,5,10,25,50])
    revenueRatio = []
    probZero = []
    for gamma in possibleGamma:
        print('Gamma = ',gamma)    
        agg_rev = 0
        agg_prob = 0
        for j in range(nturns):
            print('\r[','='*int((j+1)*100/nturns),'-'*int((nturns-j-1)*100/nturns),']',sep='',end='')
            selected_transactions_greedy = allocateTransaction(tx_value,tx_size,block_size,'greedy',gamma)
            selected_transactions_softmax = allocateTransaction(tx_value,tx_size,block_size,'softmax',gamma)
            revenue_greedy = calculateRevenue(np.multiply(selected_transactions_greedy,tx_value),revenueScheme)
            revenue_softmax = calculateRevenue(np.multiply(selected_transactions_softmax,tx_value),revenueScheme)
            agg_rev += revenue_greedy/(revenue_softmax + 0.001)
            agg_prob += zeroFracFind(selected_transactions_softmax,tx_value,tx_size,block_size)
        revenueRatio.append(agg_rev/nturns)
        probZero.append(agg_prob/nturns)
        print()
    writeJson({'gamma':list(possibleGamma),'ratio':list(revenueRatio),'prob':list(probZero)},str(tx_sampling_scheme)+str(gamma_count)+str(block_size)+'.json')
    plt.title('Sampled [Ratio-plot]')
    plt.xlabel('Gamma')
    plt.ylabel('Greedy:Softmax')
    plt.plot(possibleGamma,np.ones(possibleGamma.shape),label='Optimal')
    plt.plot(possibleGamma,revenueRatio)
    plt.savefig(str(tx_sampling_scheme)+'SamplingRatio_'+str(gamma_count)+str(block_size)+'.png')
    plt.clf()
    plt.title('Sampled [Probability]')
    plt.xlabel('Gamma')
    plt.ylabel('P(zero)')
    # plt.plot(possibleGamma,np.ones(possibleGamma.shape),label='Optimal')
    plt.plot(possibleGamma,probZero)
    plt.savefig(str(tx_sampling_scheme)+'SamplingProbability_'+str(gamma_count)+str(block_size)+'.png')
    plt.clf()

C = [100,250,350,500,750,850,900]
for alpha in C:
    print('For C = ',alpha)
    simulate(20,10,30,1000,alpha,'uniform','FPA')
    simulate(20,10,30,1000,alpha,'exponential','FPA')
    simulate(20,10,30,1000,alpha,'normal','FPA')
'''
simulate(20,10,30,1000,1000,'uniform','FPA')
'''
#print('1a')
#simulate(20,10,30,1000,100,'uniform','FPA')
#print('1b')
#simulate(20,10,30,1000,100,'exponential','FPA')
#print('1c')
#simulate(20,4,30,1000,100,'normal','FPA')
#print('1d')
#simulate(20,4,30,1000,200,'uniform','FPA')
#print('1e')
#simulate(20,4,30,1000,200,'exponential','FPA')
#print('1e')
#simulate(20,4,30,1000,200,'normal','FPA')
#print('1f')
#simulate(20,4,30,1000,400,'uniform','FPA')
#print('1g')
#simulate(20,4,30,1000,400,'exponential','FPA')
#print('1h')
#simulate(20,4,30,1000,400,'normal','FPA')
#print('Done')

#simulate(5,2,5,1000,100,'exponential','FPA')

