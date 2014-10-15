import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt




def plot_lls_metal_hist(column, met, min_col=17.0, max_col=19.5, nbins=30, xmin=-3, xmax=0):
    column = column.flatten()
    met = np.log10(10.0**(met.flatten()) / 0.0127)
    select = (column > min_col) & (column < max_col)

    print met.min(), met.max()

    hist, edges = np.histogram(met[select],  bins=nbins, range=[xmin,xmax])

    print hist, edges

    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(1, 1, 1)
    new_edges = np.zeros( edges.shape[0]-1 )
    for index,min in enumerate(edges[:-1]):
      max = edges[index+1]
      new_edges[index] = (min + max)/2.0

    print new_edges.shape,hist.shape
    ax.plot(new_edges,hist)
    ax.set_xlim([edges[0],edges[-1]])
    fig.subplots_adjust(left=0.15, right=0.97, top=0.97, bottom=0.15)

    ax.set_xlabel(r'Z/Z${}_\odot$')
    fig.savefig('met_pdf.pdf')


    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(1, 1, 1)
    met = met[select]			# select only LLS's
    met.sort()			# sort their metallicity

    cum_val = np.arange( met.shape[0] ) / (1.0* met.shape[0] )
    cum_val = cum_val[::-1]

    ax.plot(met,cum_val, label='Sim all LLS')

    ax.set_xlim([edges[0],edges[-1]])
    fig.subplots_adjust(left=0.15, right=0.97, top=0.97, bottom=0.15)

    cdf1=np.array( [0.188,0.250,0.386,0.474,0.579,0.790])   # metalpoor
    err1=np.array( [0.098,0.108,0.124,0.134,0.143,0.165] )
    z1  =np.array( [-2.0,-2.2,-2.5,-2.7,-2.9,-3.45] )

    cdf2=np.array( [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8] )		# ;blind
    z2  =np.array( [-0.75,-1.55,-1.8,-2.0,-2.05,-2.1,-2.2,-2.6] ) 
    err2=np.array( [0.095,0.126,0.145,0.155,0.158,0.155,0.145,0.126] )

    ax.scatter(z1, cdf1, color='b', label='Metal Poor')
    ax.scatter(z2, cdf2, color='r', label='Blind')

    ax.legend()

    ax.set_ylabel('CDF (>[M/H])')
    ax.set_xlabel(r'[M/H]')
    fig.savefig('met_cum_dist.pdf')

    f = open('lss_metallicity_cdf.txt','w')
    for index in np.arange(cum_val.shape[0]):
        line = '{:.8}  {:.8}\n'.format(met[index], cum_val[index])
        f.write(line)
    f.close()



def plot_column_vs_met(column, met, nbins=100, xmin=14, xmax=22, ymin=-6, ymax=1, weights=None):

  my_x = column.flatten()	# already in log(N_H)
  my_y = np.log10(10.0**(met.flatten()) / 0.0127)	# already in log(Z)

  if weights==None:
    weights=np.zeros( my_x.shape )+1.0


  print my_x.shape
  print my_y.shape

  hist, xedges, yedges = np.histogram2d(my_x, my_y, bins=nbins, range=[[xmin,xmax],[ymin,ymax]], weights=weights)
  extent = [xedges[0], xedges[-1], yedges[0], yedges[-1] ]

  hist = np.log10(hist+1)

  fig = plt.figure(figsize=(5,5))
  ax = fig.add_subplot(1, 1, 1)
  ax.imshow(hist.T,extent=extent,interpolation='nearest',origin='lower')

#  ax.plot(np.log10([n_h_down, n_h_down]), [-100, 100],linewidth=3, color='k')
#  ax.plot(np.log10([n_h_up, n_h_up]), [-100, 100],linewidth=3, color='k')
#  ax.plot([-100,100], [-2.5, -2.5], linewidth=3, color='k')

  ax.set_xlim([xedges[0],xedges[-1]])
  ax.set_ylim([yedges[0],yedges[-1]])

  ax.set(aspect=(1.0*(xmax-xmin))/(1.0*(ymax-ymin)))

  fig.subplots_adjust(left=0.15, right=0.97, top=0.97, bottom=0.15)

  ax.set_xlabel(r'N${}_{HI}$ [cm${}^{-2}$]')
  ax.set_ylabel(r'Z/Z${}_\odot$')

#  redshift_label=str(redshifts[snap])
#  redshift_label='z='+redshift_label[:3]
#  ax.text(0, -5.5, redshift_label, color='w' )

  fig.savefig('col_vs_met.pdf')




def plot_column_map(column, slice=0, nmin=14, nmax=22, savebase='HI_map_slice'):
    
    this_slice=column[slice,:,:]

    fig = plt.figure(figsize=(15,15))

    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

# first map for low density material
    norm = mpl.colors.Normalize(vmin=10, vmax=nmin)
    cmap = plt.cm.Blues
    mapper = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    rgba   = mapper.to_rgba(this_slice)
    rgba[:,:,3] = 0.8
    ax1.imshow(rgba)

# now make LLS and up image with transparent background   
    norm = mpl.colors.Normalize(vmin=nmin, vmax=nmax)
    cmap = plt.cm.spring
    mapper = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    rgba   = mapper.to_rgba(this_slice)
   
    low_pixel_index = this_slice < nmin
    rgba[low_pixel_index,3] = 0.0
    ax2.imshow(rgba, vmin=nmin, vmax=nmax)    

    position=fig.add_axes([0.85,0.1,0.02,0.8])  ## the parameters are the specified position you set 
#    plt.imshow(this_slice)
    cmap = plt.cm.spring
    cmap = plt.cm.spring

    mini_ax = fig.add_axes([0.000, 0.001, 0.000, 0.001])
    dummy = np.zeros( (5,5) )
    f = mini_ax.imshow(dummy, cmap='spring', vmin=nmin, vmax=nmax)
    mini_ax.axes.get_xaxis().set_visible(False)
    mini_ax.axes.get_yaxis().set_visible(False)

    cbar = plt.colorbar(f, cmap=cmap, cax=position)                #, ticks=np.arange(5)+15)
    cbar.set_cmap('spring')
    cbar.ax.tick_params(labelsize=25)
    cbar.ax.set_ylabel(r'log(N${}_{HI}$ [cm${}^{-2}$])', fontsize=30)


    ax1.axes.get_xaxis().set_visible(False)
    ax1.axes.get_yaxis().set_visible(False)
    ax2.axes.get_xaxis().set_visible(False)
    ax2.axes.get_yaxis().set_visible(False)

#    position=fig.add_axes([0.85,0.1,0.02,0.8])  ## the parameters are the specified position you set 
#    position.set_ylabel(r'log(N${}_{HI}$)')
#a    cbar = plt.colorbar(cax=position, cmap=cmap)		#, ticks=np.arange(5)+15)
#    cbar.ax.tick_params(labelsize=25) 
#    cbar.ax.set_ylabel(r'log(N${}_{HI}$ [cm${}^{-2}$])', fontsize=30)

    left = 0.03
    delta= 0.8
    fig.subplots_adjust(left=left, right=left+delta, top=1.0 - (1.0-delta)/2.0, bottom=(1.0-delta)/2.0 )
  
    fig.savefig(savebase+'_'+str(slice)+'.pdf', dpi=512)




