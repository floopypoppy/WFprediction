#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
hyperparameter tuning

"""

import operator
wferr = pickle.load(open('/users/xliu/dropbox/expout/WFerr_#node.p','rb'))
sorted_dic = sorted(wferr.items(), key=operator.itemgetter(1))
bwe = []
first_layer_nodes = [21, 55, 90, 134, 167, 191]
for nodes in first_layer_nodes:
    bwe.append(wferr['%d_0_0'%nodes])

plt.scatter(first_layer_nodes, bwe, s=5, color='C1', label='prediction')
xleft, xright = plt.xlim()
plt.xticks(first_layer_nodes)
plt.xlim(xleft, xright)
plt.hlines(rsmd_delay,xleft,xright,linestyle='dotted',color='C0',label='delay: 111.6nm')
plt.legend()
plt.xlabel('number of hidden nodes in a single-hidden-layer LSTM')
plt.ylabel('bandwidth error (nm) with prediction')
plt.title('BW error (nm) with prediction vs. size of a single-hidden-layer LSTM')
plt.savefig('../error analysis/HyperTune-BWErrvs.#nodes(1-layer).png',dpi=600)
    
wfes = pickle.load(open('/users/xliu/dropbox/expout/HyperTune.p','rb'))
rsmd_delay = rsmd(wfes['112_153_0'][5], wfes['112_153_0'][3])

histories = pickle.load(open('/users/xliu/dropbox/expout/histories.p','rb'))
a = histories['133_172_0_lr=0.00036506']
plt.plot(a['loss'])
plt.plot(a['val_loss'])
plt.legend(['loss','val loss'])

'''# of layers'''
mean = []
std = []
key = '0_0_0_0_lr=0.001'
for i in range(1,5):
    new_key = key.replace('0','72',i)
    value = []
    for j in range(3):
        value.append(wferr[new_key+'(%d)'%j])
    mean.append(lmean(value))
    std.append(lstd(value))
    
plt.errorbar(range(1,5), mean, yerr=std, fmt='o', capsize=3, ms=2)
plt.xticks(range(1,5))
plt.xlabel('# of hidden layers')
plt.ylabel('Bandwidth error (nm)')
plt.title('Errorbar of bandwidth error vs. number of hidden layers \n (# of hidden nodes = 72, # of models each = 3)')
plt.savefig('../error analysis/Errorbar of BW error vs. #layer.png', dpi=600)
            
'''# of nodes'''
wferr_node = pickle.load(open('/users/xliu/dropbox/expout/WFerr_#node.p','rb'))            
keys = wferr_node.keys()
leng = len(keys)
first = []
second = []
for i in range(leng):
    if int(keys[i].split('_')[0]) not in first:
        first.append(int(keys[i].split('_')[0]))
        second.append(int(keys[i].split('_')[1]))
nsample = len(first)
mean = []
std = []
for j in range(nsample):
    sublst = [wferr['%d_%d_0_0_lr=0.001(%d)'%(first[j],second[j],i)] for i in range(3)]
    mean.append(lmean(sublst))
    std.append(lstd(sublst))
wferr_layer = pickle.load(open('/users/xliu/dropbox/expout/WFerr_#layer.p','rb'))            
sublst = [wferr_layer['72_72_0_0_lr=0.001(%d)'%(i)] for i in range(3)]
mean.append(lmean(sublst))
std.append(lstd(sublst))
first.append(72)
second.append(72)

mean, std, first, second = zip(*sorted(zip(mean, std, first, second)))
plt.scatter(first, second, c=std)
plt.xlim([50, 250])
plt.ylim([50, 250])
plt.xticks(range(50,300,50))
plt.yticks(range(50,300,50))
plt.hlines(200, 50, 200, linestyle='dotted')
plt.vlines(200, 50, 200, linestyle='dotted')
plt.colorbar()
plt.xlabel('# of first-layer nodes')
plt.ylabel('# of second-layer nodes')
plt.title('standard deviation of bandwidth error (nm) \n with respect to # of hidden nodes')
plt.savefig('../error analysis/Std of BW error vs #of nodes.png', dpi=600)         

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(first, second, mean, marker='^')
ax.set_xlabel('# of nodes in the first hidden layer')
ax.set_ylabel('# of nodes in the second hidden layer')
ax.set_zlabel('Bandwidth error (nm)')

'''(1) diff'''
diff = [abs(first[i]-second[i]) for i in range(13)]
plt.errorbar(diff, mean, std, fmt='o', capsize=3, ms=2)
plt.xlabel('|# of nodes (first layer) - # of nodes (second layer)|')
plt.ylabel('Bandwidth error (nm)')
plt.savefig('Errorbar of BW error vs. #node_diff.png', dpi=600)

stddic = {}
for j in range(13):
    stddic['%d_%d'%(first[j],second[j])] = std[j]
'''(2) # of parameters'''
def npara(m, n):
    return 4*(n*m+n**2+n)
def allpara(m, n):
    return npara(m, n)+npara(n, 72)
n_para = []
for j in range(13):
    n_para.append(allpara(first[j], second[j]))
all_para = [n_para[i] + 72*73 for i in range(13)]
plt.errorbar(n_para, mean, std, fmt='o', capsize=3, ms=2)
plt.xlabel('# of overall parameters')
plt.ylabel('Bandwidth error (nm)')
plt.savefig('../error analysis/Errorbar of BW error vs. #overall_node.png', dpi=600, overwrite=True)

'''learning rate'''
wferr_lr = pickle.load(open('/users/xliu/dropbox/expout/WFerr_lr.p','rb'))
lr = [0.00021393, 0.00150170, 0.00129919, 0.00020887, 0.00028436, 0.00021300, 0.00139351, 0.00016631, 0.00019422, 0.00150740, 0.00195873, 0.00048007, 0.00022831, 0.00062151, 0.00053479, 0.00115656, 0.00072576, 0.00033232]
leng = len(lr)
mean = []
std = []
for j in range(leng):
    sublst = [wferr_lr['72_72_0_0_lr=%.8f(%d)'%(lr[j],i)] for i in range(3)]
    mean.append(lmean(sublst))
    std.append(lstd(sublst))
wferr_layer = pickle.load(open('/users/xliu/dropbox/expout/WFerr_#layer.p','rb'))            
sublst = [wferr_layer['72_72_0_0_lr=0.001(%d)'%(i)] for i in range(3)]
mean.append(lmean(sublst))
std.append(lstd(sublst))
lr.append(0.001)
lr, mean, std = zip(*sorted(zip(lr, mean, std))) # sort the lists in the same order
fig = plt.figure()
ax = fig.add_subplot(111)
ax.errorbar(lr, mean, std, fmt='o', capsize=3, ms=2)
ax.set_xscale('log')
ax.set_xlim([10**(-4), 10**(-2.7)])
#ax.set_xticks([10**(-i) for i in [4, 3, 2.7]])
plt.xlabel('initial learning rate')
plt.ylabel('Bandwidth error (nm)')
plt.title('Errorbar of bandwidth error vs. initial learning rate')
plt.grid(True)

plt.savefig('../error analysis/Errorbar of BW error vs. lr.png', dpi=600, overwrite=True)


    