"""Script for making various DLA-related plots"""

import matplotlib.pyplot as plt
import os.path as path
import dla_plots as dp
from save_figure import *

base="/home/spb/data/ComparisonProject/512_20Mpc"
outdir=path.join(base,"plots")
snapnum=124

#Fig 6
plt.figure()
gdir=path.join(base,"Gadget")
adir=path.join(base,"Arepo_ENERGY")
ahalo=dp.PrettyHalo(adir,snapnum,"Arepo",halo=0)
ahalo.plot_pretty_halo()
save_figure(path.join(outdir,"Arepo_"+str(snapnum)+"pretty_halo"))

plt.figure()
ghalo=dp.PrettyHalo(gdir,snapnum,"Arepo",halo=0)
ghalo.plot_pretty_halo()
save_figure(path.join(outdir,"Gadget_"+str(snapnum)+"pretty_halo"))

#Fig 9
plt.figure()
dp.plot_totalHI(base,snapnum)
save_figure(path.join(outdir,"total_HI_"+str(snapnum)))

#Fig 10
hplots=dp.HaloHIPlots(base,snapnum,1000)
plt.figure()
hplots.plot_sigma_DLA()
save_figure(path.join(outdir,"sigma_DLA_"+str(snapnum)))

#Fig 11
plt.figure()
hplots.plot_dN_dla()
save_figure(path.join(outdir,"dNdz_"+str(snapnum)))

#Fig 12
plt.figure()
hplots.plot_column_density()
save_figure(path.join(outdir,"columden_"+str(snapnum)))
