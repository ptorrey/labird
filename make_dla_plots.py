"""Script for making various DLA-related plots"""

import matplotlib
matplotlib.use('PDF')

import matplotlib.pyplot as plt
import os.path as path
import dla_plots as dp
import numpy as np
from save_figure import *
import sys

#Argument: 1 => totalHI plots
#          2 => pretty density plots
#          3 => sigma_DLA plots
#          4 => Column density plots
bases=[
# "/home/spb/data/ComparisonProject/128_20Mpc",
# "/home/spb/data/ComparisonProject/256_20Mpc",
"/home/spb/data/ComparisonProject/512_20Mpc",
]
minpart=400
snaps=[
90,
124,
191,
]

#Plots with all the halo particles
if len(sys.argv) < 2 or int(sys.argv[1]) == 1:
    for (base,snapnum) in [(bb,ss) for bb in bases for ss in snaps]:
        outdir=path.join(base,"plots")
        print "Saving total plots for snapshot ",snapnum," to ",outdir
        #Fig 9
        tot=dp.TotalHIPlots(base,snapnum,minpart)
        plt.figure()
        tot.plot_totalHI()
        save_figure(path.join(outdir,"total_HI_"+str(snapnum)))

        plt.clf()
        tot.plot_MHI()
        save_figure(path.join(outdir,"MHI_vs_Mgas"+str(snapnum)))
        plt.clf()

        tot.plot_gas()
        save_figure(path.join(outdir,"halo_vs_gas_"+str(snapnum)))
        plt.clf()

#Plots with the nearest 200kpc
for (base,snapnum) in [(bb,ss) for bb in bases for ss in snaps]:
    outdir=path.join(base,"plots")
    print "Saving plots for snapshot ",snapnum," to ",outdir

    plt.figure(1)

    if len(sys.argv) < 2 or int(sys.argv[1]) == 2:
        #Load only the gas grids
        hplots=dp.HaloHIPlots(base,snapnum,minpart=minpart,skip_grid=1)
        #Find a smallish halo
        a_shalo=np.min(np.where(hplots.ahalo.sub_mass < 1e10))
        #Get the right halo for a smaller halo
        s_mass=hplots.ahalo.sub_mass[a_shalo]
        s_pos=hplots.ahalo.sub_cofm[a_shalo,:]
        g_shalo = hplots.ghalo.identify_eq_halo(s_mass,s_pos)
        #Make sure it has a gadget counterpart
        if np.size(g_shalo) == 0:
            for i in np.where(hplots.ahalo.sub_mass < 1e10)[0]:
                s_mass=hplots.ahalo.sub_mass[i]
                s_pos=hplots.ahalo.sub_cofm[i,:]
                g_shalo = hplots.ghalo.identify_eq_halo(s_mass,s_pos)
                if np.size(g_shalo) != 0 :
                    a_shalo=i
                    break
        #Get the right halo
        g_halo_0 = hplots.ghalo.identify_eq_halo(hplots.ahalo.sub_mass[0],hplots.ahalo.sub_cofm[0,:])[0]

        hplots.ahalo.plot_pretty_gas_halo(0)
        save_figure(path.join(outdir,"Arepo_"+str(snapnum)+"pretty_gas_halo"))
        plt.clf()
        hplots.ghalo.plot_pretty_gas_halo(g_halo_0)
        save_figure(path.join(outdir,"Gadget_"+str(snapnum)+"pretty_gas_halo"))
        plt.clf()
        hplots.ahalo.plot_pretty_gas_halo(a_shalo)
        save_figure(path.join(outdir,"Arepo_"+str(snapnum)+"_small_pretty_gas_halo"))
        plt.clf()
        hplots.ghalo.plot_pretty_gas_halo(g_shalo[0])
        save_figure(path.join(outdir,"Gadget_"+str(snapnum)+"_small_pretty_gas_halo"))
        #Radial profiles
        plt.clf()
        hplots.plot_radial_profile(maxR=15.)

        plt.figure(2)
        #low-mass halo radial profile
        hplots.plot_radial_profile(minM=7e9, maxM=7.5e9,maxR=5.)

        del hplots

    #Load only the nHI grids
    hplots=dp.HaloHIPlots(base,snapnum,minpart=minpart,skip_grid=2)

    if len(sys.argv) < 2 or int(sys.argv[1]) == 2:
        #low-mass halo radial profile
        hplots.plot_radial_profile(minM=7e9, maxM=7.5e9,maxR=5.)
        save_figure(path.join(outdir,"radial_profile_halo_low_"+str(snapnum)))
        plt.clf()

        plt.figure(1)
        hplots.plot_radial_profile(maxR=15.)
        save_figure(path.join(outdir,"radial_profile_halo_0_"+str(snapnum)))
        #Fig 6
        plt.clf()
        hplots.ahalo.plot_pretty_halo(0)
        save_figure(path.join(outdir,"Arepo_"+str(snapnum)+"pretty_halo"))
        plt.clf()
        hplots.ghalo.plot_pretty_halo(g_halo_0)
        save_figure(path.join(outdir,"Gadget_"+str(snapnum)+"pretty_halo"))

        plt.clf()
        hplots.ahalo.plot_pretty_cut_halo(0)
        save_figure(path.join(outdir,"Arepo_"+str(snapnum)+"pretty_cut_halo"))
        plt.clf()
        hplots.ghalo.plot_pretty_cut_halo(g_halo_0)
        save_figure(path.join(outdir,"Gadget_"+str(snapnum)+"pretty_cut_halo"))

        plt.clf()
        hplots.ahalo.plot_pretty_halo(a_shalo)
        save_figure(path.join(outdir,"Arepo_"+str(snapnum)+"_small_pretty_halo"))
        plt.clf()
        hplots.ghalo.plot_pretty_halo(g_shalo[0])
        save_figure(path.join(outdir,"Gadget_"+str(snapnum)+"_small_pretty_halo"))
        plt.clf()

    if len(sys.argv) < 2 or int(sys.argv[1]) == 3:
        #Fig 10
        hplots.plot_sigma_DLA()
        save_figure(path.join(outdir,"sigma_DLA_"+str(snapnum)))

        #Fig 10
        plt.clf()
        hplots.plot_sigma_DLA(17)
        save_figure(path.join(outdir,"sigma_DLA_17_"+str(snapnum)))

        #Same but against nHI mass
        plt.clf()
        hplots.plot_sigma_DLA_nHI()
        save_figure(path.join(outdir,"sigma_DLA_nHI"+str(snapnum)))

        #Same but against nHI mass
        plt.clf()
        hplots.plot_sigma_DLA_nHI(17)
        save_figure(path.join(outdir,"sigma_DLA_17_nHI"+str(snapnum)))

        #Same but against gas mass
        plt.clf()
        hplots.plot_sigma_DLA_gas()
        save_figure(path.join(outdir,"sigma_DLA_gas"+str(snapnum)))

        #Same but against gas mass
        plt.clf()
        hplots.plot_sigma_DLA_gas(17)
        save_figure(path.join(outdir,"sigma_DLA_17_gas"+str(snapnum)))

        #Fig 10
        plt.clf()
        hplots.plot_rel_sigma_DLA()
        save_figure(path.join(outdir,"rel_sigma_DLA_"+str(snapnum)))
        plt.clf()

    if len(sys.argv) < 2 or int(sys.argv[1]) == 4:
    #Fig 11
#     plt.clf()
#     hplots.plot_dN_dla()
#     save_figure(path.join(outdir,"dNdz_"+str(snapnum)))

        hplots.plot_central_density()
        save_figure(path.join(outdir,"central_den_"+str(snapnum)))

        #Fig 12
        plt.clf()
        hplots.plot_column_density()
        save_figure(path.join(outdir,"columden_"+str(snapnum)))

        #Fig 12
        plt.clf()
        hplots.plot_rel_column_density()
        save_figure(path.join(outdir,"columden_rel_"+str(snapnum)))

        #Fig 5
        plt.clf()
        hplots.plot_halo_mass_func()
        save_figure(path.join(outdir,"halo_func_"+str(snapnum)))
        plt.clf()
