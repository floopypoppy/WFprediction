#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
ATMOS test -- compare set r0 and r0 measured from WFS slopes

"""

L0 = 10
#L0_new = 1e10
l0 = 0.01
k0 = 2*pi/L0
#k0_new = 2*pi/L0_new

km = 5.92/l0
k = np.linspace(0.01, 1000, 2000)
k1 = np.linspace(0.1, 1000, 2000)

phi_k = 0.49 * k**(-11/3.)
phi_mvk = 0.49 * np.e**(-k**2/km**2) / (k**2 + k0**2)**(11/6.)
phi_mvk1 = 0.49 * np.e**(-k1**2/km**2) / (k1**2 + k0**2)**(11/6.)
#phi_mvk_new = 0.49 * np.e**(-k**2/km**2) / (k**2 + k0_new**2)**(11/6.)

plt.figure()
plt.loglog(k, phi_k)
plt.loglog(k, phi_mvk, linestyle='dotted')
plt.loglog(k1, phi_mvk1, linestyle='dashed')
plt.vlines([k0,km], 10e-13, 10e4, colors='k', linestyle='dashed', linewidth=0.5)

dic = pickle.load(open('/users/xliu/dropbox/Dell/soapytest-master/measuredR0_N\n.p', 'rb'))
R0s = dic['R0s']

# convert measured r0 to r0 @ 500nm
keys = ['4', '8', '16',]# '256']
colors = ['C0', 'C1', 'C2']
# scaled_measuredR0 = (wvl/500e-9) ** (-6/5.) *measuredR0
pyplot.figure()
for i in range(3):
    pyplot.plot(R0s, dic[keys[i]], color=colors[i], label=keys[i])
#     pyplot.plot(R0s, dic2[keys[i]], marker='*', color=colors[i], label=keys[i])
pyplot.plot(R0s, R0s, color="k", linestyle=":", label='ideal')
# pyplot.plot(R0s, scaled_measuredR0, label='ideal')
pyplot.grid()
pyplot.xlabel("Phase Screen $r_0$ @ 500 nm")
pyplot.ylabel("$r_0$ measured from WFS slopes")
pyplot.legend(loc='upper left')
# pyplot.xlim(0.1, 0.26)
# pyplot.ylim(0.1, 0.26)
# pyplot.gca().set_aspect('equal', adjustable='box')
# print(R0s)
# print(scaled_measuredR0)
