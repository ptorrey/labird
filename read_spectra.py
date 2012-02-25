"""Module for reading spectra generated by the flux_extractor, and for generating
LOS tables.
        Functions:
                gen_los_table - save LOS table on a regular grid.
        Classes: 
                Spectra - class for loading Lyman-alpha spectra generated by the flux_extractor
                Subfind - saves LOS table through halo centers.
"""
import numpy as np
import math
import matplotlib.pyplot as plt
import os.path as path
import readsubf
import fieldize


def gen_los_table(filename,nbins,box=20.):
    """Function to generate and save a LOS table on a regular grid
    Inputs:
            filename - filename to save LOS table to.
            nbins - grid spacing
            box - size of grid
    """
    los_table=np.empty([3*nbins**2,4])
    sc=(1.*box)/nbins
    for j in range(0,nbins):
        for i in range(0,nbins):
            los_table[nbins*j+i,:]=[1,0,i*sc,j*sc]
    for j in range(0,nbins):
        for i in range(0,nbins):
            los_table[nbins**2+nbins*j+i,:]=[2,i*sc,0,j*sc]
    for j in range(0,nbins):
        for i in range(0,nbins):
            los_table[2*nbins**2+nbins*j+i,:]=[3,i*sc,j*sc,0]
    np.savetxt(filename,los_table,fmt="%d %.3e %.3e %.3e")

class Spectra:
    """Class for loading Lyman-alpha spectra generated by the flux_extractor, 
    with their LOS tables."""
    def __init__(self,spec_file,los_table="",nlos=16000,nbins=1024,box=20,no_header=1):
        self.nlos=nlos
        self.nbins=nbins
        self.box=box
        (self.zz, self.n_HI)=self.read_spectra(spec_file,no_header)
        if los_table != "":
            self.los_table=np.loadtxt(los_table)
        return

    def read_spectra(self,spec_file,no_header=1):
        """Load a file of spectra as output by flux_extractor.
           Inputs: 
                spec_file - File to load
                no_header - Is the file a new-style file with a header, 
                            or an old-style raw file.
           Outputs:
                Returns zz - redshift of file
                        n_HI - neutral hydrogen density.
        """
        fd=open(spec_file,'rb')
        zz=np.fromfile(fd,dtype=np.float64,count=1)
        if not no_header:
            self.box=np.fromfile(fd,dtype=np.float64,count=1)
            #This is an integer
            head=np.fromfile(fd,dtype=np.int32,count=2)
            self.nbins=head[0]
            self.nlos=head[1]
            #Pad to 128 bytes
            fd.seek(128,0)
        size=self.nbins*self.nlos
        #Density of hydrogen
        rho_H=np.reshape(np.fromfile(fd,dtype=np.float64,count=size),(self.nlos,self.nbins))
        #neutral hydrogen fraction
        rho_HI=np.reshape(np.fromfile(fd,dtype=np.float64,count=size),(self.nlos,self.nbins))
        n_HI=np.nansum(rho_HI/rho_H,axis=1)
        #temperature
        #temp_HI=np.reshape(np.fromfile(fd,dtype=np.float64,count=size),(self.nlos,self.nbins))
        #velocity
        #vel_HI=np.reshape(np.fromfile(fd,dtype=np.float64,count=size),(self.nlos,self.nbins))
        #optical depth
        #tau_HI=np.reshape(np.fromfile(fd,dtype=np.float64,count=size),(self.nlos,self.nbins))
        return (zz,n_HI)

    def get_los(self,axis):
        """Load a table of sightlines associated with the spectra.
        Inputs: axis - Sightline axis to consider
        Returns: sightlines along axis and their raw indices."""
        ind=np.where(self.los_table[:,0] == axis)
        axes=np.where(np.array([axis,1,2,3]) != axis)
        los=np.empty((np.size(self.los_table[ind,axes[0][0]]),2))
        los[:,0]=self.los_table[ind,axes[0][0]]
        los[:,1]=self.los_table[ind,axes[0][1]]
        return (ind,los)

    def get_int_hi(self,axis=1):
        """Get the nHI values for sightlines along axis"""
        (ind,pos)=self.get_los(axis)
        sn_HI=self.n_HI[ind]
        return sn_HI

    def get_spec_pos(self,axis=1):
        """Get the positions of sightlines along axis"""
        (ind,pos)=self.get_los(axis)
        return (pos[:,0],pos[:,1])

    def get_lims(self):
        """Get tuple of limits for imshow"""
        return(0,self.box,0,self.box)

    def plot_int_hi(self,axis,vmax=None):
        """Use matplotlib's imshow to draw the HI grid from various spectra
        Inputs:
                axis - Axis to consider sightlines along
                vmax - Maximum nHI to plot"""
        nHI=self.grid_int_hi(axis)
        #Set zero values to Nan.
        nHI[np.where(nHI == 0)]*=np.NaN
        if vmax == None:
            plt.imshow(nHI,origin='lower',extent=self.get_lims(),aspect='auto')
        else:
            plt.imshow(nHI,origin='lower',extent=self.get_lims(),aspect='auto',vmax=vmax,vmin=0)
        plt.colorbar()

    def grid_int_hi(self,axis):
        """Put the n_HI from spectra along axis onto a grid.
        Inputs:
                axis - axis to consider
        Outputs:
                grid_n_HI - a grid of HI values"""
        n_HI=self.get_int_hi(axis)
        (x,y)=self.get_spec_pos(axis)
        nbins=math.sqrt(self.nlos/3)/2
        grid_n_HI = np.zeros([nbins,nbins])
        pos=fieldize.convert((x,y),nbins,self.box)
        grid_n_HI=fieldize.ngp(pos,n_HI,grid_n_HI)
        return grid_n_HI



class Subfind(readsubf.subfind_catalog):
    """Generates a LOS or position table from a halo catalogue.
    Sightlines get thrown through the center of every halo above a certain mass.
    Masses are all in M_sun"""
    def __init__(self,sim_dir,snapnum):
        """Inputs:
                sim_dir - Directory where the subfind catalogue is stored.
                snapnum - Snapshot to load
        """
        self.sim_dir=sim_dir
        self.snapnum=snapnum
        readsubf.subfind_catalog.__init__(self,sim_dir,snapnum,masstab=True,long_ids=True)

    def gen_halo_los_table(self, minmass=3e8):
        """Generate and save a LOS table of sightlines going through a halo center. 
        Input: 
                minmass - Minimum mass halo to consider, in solar masses.
        Output:
                $sim_dir/los_$snapnum.txt 
                formatted as:
                axis, x, y, z
                'axis' is the axis the sightline goes along. 
                Two of the next three columns show the coordinates of the sightline.
        """
        #Look at above-average mass halos only
        ind=np.where(self.sub_mass > minmass/1e10)
        nsubs=np.size(ind)
        #Make table of sightlines, one going through 
        #the center of each halo in each direction
        los_table=np.empty([3*nsubs,4])
        #x-axis
        los_table[0:nsubs,1:4]=self.sub_pos[ind]/1000
        los_table[0:nsubs,0]=1
        #y-axis
        los_table[nsubs:2*nsubs,1:4]=self.sub_pos[ind]/1000
        los_table[nsubs:2*nsubs,0]=2
        #z-axis
        los_table[2*nsubs:3*nsubs,1:4]=self.sub_pos[ind]/1000
        los_table[2*nsubs:3*nsubs,0]=3
        np.savetxt(path.join(self.sim_dir,"los_"+str(self.snapnum)+".txt"),los_table,fmt="%d %.3e %.3e %.3e")
    
    def gen_halo_table(self,  minmass=3e8):
        """Generate and save a list of halo positions.
        Input: 
                minmass - Minimum mass halo to consider, in solar masses.
        Output:
                $sim_dir/halo_$snapnum.txt 
                formatted as:
                x, y, z  - Coordinates of the halo CofM
        """
        #Look at above-average mass halos only
        ind=np.where(self.sub_mass > minmass/1e10)
        #Make table of sightlines, one going through
        #the center of each halo in each direction
        np.savetxt(path.join(self.sim_dir,"halo_"+str(self.snapnum)+".txt"),self.sub_pos[ind]/1000,fmt="%.3e %.3e %.3e")

    def get_halo_mass(self,  dimension=1e10):
        """Convert internal gadget masses to solar masses.
        Default conversion is code units where one mass unit is 1e10 solar masses"""
        return self.sub_mass*dimension
