# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 11:48:02 2021

@author: asdg
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

data = np.random.multivariate_normal((0, 0), [[0.8, 0.05], [0.05, 0.7]], 1000)
x = data[:, 0]
y = data[:, 1]
xmin, xmax = -2, 2
ymin, ymax = -2, 2
# Peform the kernel density estimate
xx, yy = np.mgrid[xmin:xmax:0.4, ymin:ymax:0.4]
positions = np.vstack([xx.ravel(), yy.ravel()])
values = np.vstack([x, y])
kernel = st.gaussian_kde(values)
f = np.reshape(kernel(positions).T, xx.shape)
fig = plt.figure()
ax = fig.gca()
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
# Contourf plot
cfset = ax.contourf(xx, yy, f, cmap='jet')
## Or kernel density estimate plot instead of the contourf plot
#ax.imshow(np.rot90(f), cmap='Blues', extent=[xmin, xmax, ymin, ymax])
# Contour plot
cset = ax.contour(xx, yy, f, colors='k')
# Label plot
ax.clabel(cset, inline=1, fontsize=10)
ax.set_xlabel('Y1')
ax.set_ylabel('Y0')
plt.show();
























































