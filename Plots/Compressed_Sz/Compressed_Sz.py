import numpy as np
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import h5py as h5
sys.path.append("C:/Users/maxim/Documents/GitHub/Few_body_operators/Plots/") ## Important to bind the modules
import figures_module

        
## LateX properties of the plot
fig, ax = figures_module.prepare_standard_figure(ncols=1,nrows=1, width=2.75,aspect_ratio=1.3)

L = 30
V = 0.2

## ---------------------------------------------------------------
## Plot the spectrum for a given U
## ---------------------------------------------------------------

filename = "C:/Users/maxim/Documents/GitHub/Few_body_operators/Plots/Compressed_Sz/IRLM_Sz_L30_dt0.01_alpha-1_Truncationglobal_chi1024_Sect30_TrottOrder4_Nsteps1010_phys_dims1_d2_Uint0.2_V0.2_gamma0.5_ed0_J0_Jz0.h5"
with h5.File(filename, 'r') as f:
    BondDim = np.array(f['BondDim'])[::2]
    BondDimRot = np.array(f['BondDimRot'])[::2]
    Time = np.array(f['t'])[1::3]

m = ['v','o','s']
shade = 0.6
for i, t in enumerate(Time):

	x = np.arange(L+1)
    
	y = BondDim[i]
	figures_module.plot1d(ax, x, y, color=mpl.colormaps["Blues"](shade), marker=m[i], markersize=1., lw=0.3, label='$t=%s$'%np.round(t))
    
	y = BondDimRot[i]
	figures_module.plot1d(ax, x, y, color=mpl.colormaps["Reds"](shade), marker=m[i], markersize=1., lw=0.3)

	shade += 0.2


## Properties of the plot
ax.set_xlabel(r"$n$")
ax.set_ylabel(r"$\chi_n$")

namefile='C:/Users/maxim/Documents/GitHub/Few_body_operators/Plots/Compressed_Sz/Compression_Sz.pdf'

fig.tight_layout(pad=0.8, h_pad=0.8, w_pad=0.8)
plt.draw()
plt.savefig(namefile,bbox_inches='tight', dpi=300)
