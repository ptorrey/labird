import numpy as np
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

  #  ax.set(aspect=(1.0*(xmax-xmin))/(1.0*(ymax-ymin)))

    fig.subplots_adjust(left=0.15, right=0.97, top=0.97, bottom=0.15)

    ax.set_xlabel(r'Z/Z${}_\odot$')
#    ax.set_ylabel(r'Z/Z${}_\odo')

    fig.savefig('met_pdf.pdf')

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
    ax = fig.add_subplot(1, 1, 1)
    
    plt.imshow(this_slice, vmin=nmin, vmax=nmax)
    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)

    position=fig.add_axes([0.85,0.1,0.02,0.8])  ## the parameters are the specified position you set 
    cbar = plt.colorbar(cax=position, label=r'log(N${}_{HI}$)')		#, ticks=np.arange(5)+15)
    cbar.ax.tick_params(labelsize=10) 

    left = 0.03
    delta= 0.8
    fig.subplots_adjust(left=left, right=left+delta, top=1.0 - (1.0-delta)/2.0, bottom=(1.0-delta)/2.0 )
  
    fig.savefig(savebase+'_'+str(slice)+'.pdf', dpi=1024)




