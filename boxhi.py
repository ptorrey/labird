# -*- coding: utf-8 -*-
"""Derived class for computing the integrated HI across the whole box.

   Author:  Simeon Bird

   Modifications:
   9/10/14  -- Edits by P. Torrey for MPI reading and addition/modification of some LLS specific routines

"""
import numpy as np
import os.path as path
import fieldize
import numexpr as ne
import spb_common.cold_gas as cold_gas
import spb_common.subfindhdf as subfindhdf
from halohi import HaloHI,calc_binned_median,calc_binned_percentile
import spb_common.hdfsim as hdfsim
from brokenpowerfit import powerfit
import h5py
import spb_common.hsml as hsml
from _fieldize_priv import _find_halo_kernel,_Discard_SPH_Fieldize

from mpi4py import MPI
import time
import gc

class BoxHI(HaloHI):
    """Class for calculating a large grid encompassing the whole simulation.
    Stores a big grid projecting the neutral hydrogen along the line of sight for the whole box.

    Parameters:
        dir - Simulation directory
        snapnum - Number of simulation
        reload_file - Ignore saved files if true
        nslice - number of slices in the z direction to divide the box into.
    """
    def __init__(self,snap_dir,snapnum,nslice=1,reload_file=False,savefile=None, savepath=None, gas=False, molec=True, start=0, end=3000, ngrid=16384, comm=None, this_task=0, n_tasks=1):
        self.snapnum=snapnum
        self.snap_dir=snap_dir
        self.molec = molec
        self.set_units()
        self.start = int(start)
        self.end = int(end)
        if savefile==None:
            savefile = "boxhi_grid_H2.hdf5"
	if savepath==None:
	    savepath = "./"

        self.savefile = path.join(savepath,savefile)
        self.tmpfile = self.savefile+"."+str(self.start)+".tmp"
        if gas:
            self.tmpfile+=".gas"
        self.sub_mass = 10.**12*np.ones(nslice)
        self.nhalo = nslice
        if reload_file:
            self.load_header()
            self.sub_cofm=0.5*np.ones([nslice,3])
            self.sub_cofm[:,0]=(np.arange(0,nslice)+0.5)/(1.*nslice)*self.box
            self.sub_radii=self.box/2.*np.ones(nslice)
            self.ngrid=ngrid*np.ones(self.nhalo)
            self.sub_nHI_grid=np.zeros([self.nhalo, ngrid,ngrid])
	    self.sub_nTotal_grid=np.zeros([self.nhalo, ngrid,ngrid])
            self.sub_nZ_grid=np.zeros([self.nhalo, ngrid,ngrid])

            try:
                thisstart = self.load_tmp()
            except IOError:
                print "Could not load file"
                thisstart = self.start

            self.set_nHI_grid(gas=False, start=0, comm=comm, this_task=this_task, n_tasks=n_tasks )			# want to move this here... no need for halohi...
	    comm.Barrier()
	    if this_task==0:
	        self.save_nHI_grid(gas=False, this_task=this_task, n_tasks=n_tasks )

	    self.set_nHI_grid(gas=True,  start=0, comm=comm, this_task=this_task, n_tasks=n_tasks ) 
	    comm.Barrier()
	    if this_task==0:
	        self.save_nHI_grid(gas=True, this_task=this_task, n_tasks=n_tasks )

	    self.set_nHI_grid(gas=True,  start=0, metallicity=True, comm=comm, this_task=this_task, n_tasks=n_tasks )
	    comm.Barrier()
	    if this_task==0:
		self.sub_nZ_grid[ self.sub_nZ_grid < -30 ] = -30
		self.sub_nZ_grid[ np.isnan(self.sub_nZ_grid) ] = -30
	        self.save_nHI_grid(gas=True, metallicity=True, this_task=this_task, n_tasks=n_tasks )
	  

	    print "saving file with full grid.  This could take a while if using the whole grid!"
	    if this_task==0:
	        self.save_file(save_grid=True)

            #Account for molecular fraction
            #This is done on the HI density now
            #self.set_stellar_grid()
            #+ because we are in log space
            #self.sub_nHI_grid+=np.log10(1.-self.h2frac(10**self.sub_nHI_grid, self.sub_star_grid))
        else:
            #try to load from a file
	    print "loading from save file"
            self.load_savefile(self.savefile)
        return


    def save_nHI_grid(self, gas=False, metallicity=False, this_task=0, n_tasks=1):
        try:
            self.sub_nHI_grid
        except AttributeError:
            self.load_hi_grid()

        for i in xrange(0,self.nhalo):
#	    if ((i % n_tasks) == this_task):		# this wont work until we take Reduce -> AllReduce in grid comm step
							# note also the this_task==0 condition before this routine is called...

                try:
		    if gas:
			if metallicity:
			    this_savefile=self.savefile[  : self.savefile.index('H2')+2]+'.Z_'+str(i)+'.hdf5'
                            f=h5py.File(this_savefile,'w')
                            grp_grid = f.create_group("GridZData")
                            grp_grid.attrs["nslice"]    = self.nhalo
                            grp_grid.attrs["ngrid"]     = self.ngrid
                            grp_grid.create_dataset(str(i),data=self.sub_nZ_grid[i].astype('f4'))
                            f.close()
			else:
                            this_savefile=self.savefile[  : self.savefile.index('H2')+2]+'.Total_'+str(i)+'.hdf5'
                            f=h5py.File(this_savefile,'w')
                            grp_grid = f.create_group("GridTotData")
                            grp_grid.attrs["nslice"]    = self.nhalo
                            grp_grid.attrs["ngrid"]     = self.ngrid
                            grp_grid.create_dataset(str(i),data=self.sub_nTotal_grid[i].astype('f4'))
                            f.close()
	    	    else:
                        this_savefile=self.savefile[  : self.savefile.index('H2')+2]+'.HI_'+str(i)+'.hdf5'
                        f=h5py.File(this_savefile,'w')
                        grp_grid = f.create_group("GridHIData")
                        grp_grid.attrs["nslice"]    = self.nhalo
                        grp_grid.attrs["ngrid"]     = self.ngrid
                        grp_grid.create_dataset(str(i),data=self.sub_nHI_grid[i].astype('f4'))
                        f.close()

                except AttributeError:
                    print "failed to write GridData"
                    pass


    def save_file(self, save_grid=True, LLS_cut = 17., DLA_cut = 20.3):
        """Save the file, by default without the grid"""
#        HaloHI.save_file(self,save_grid)

        f=h5py.File(self.savefile,'w')
        grp = f.create_group("HaloData")

        grp.attrs["redshift"]	=self.redshift
        grp.attrs["hubble"]	=self.hubble
        grp.attrs["box"]	=self.box
        grp.attrs["npart"]	=self.npart
        grp.attrs["omegam"]	=self.omegam
        grp.attrs["omegal"]	=self.omegal
        grp.create_dataset("ngrid",data=self.ngrid)
        grp.create_dataset('sub_cofm',data=self.sub_cofm)
        grp.create_dataset('sub_radii',data=self.sub_radii)
        try:
            grp.attrs["minpart"]=self.minpart
            grp.create_dataset('sub_mass',data=self.sub_mass)
            grp.create_dataset('halo_ind',data=self.ind)
        except AttributeError:
            pass
        try:
            grp.attrs["pDLA"]=self.pDLA
            grp.attrs["Rho_DLA"]=self.Rho_DLA
            grp.attrs["Omega_DLA"]=self.Omega_DLA
            grp.create_dataset('cddf_bins',data=self.cddf_bins)
            grp.create_dataset('cddf_f_N',data=self.cddf_f_N)
        except AttributeError:
            pass

	f.close()

        #Save a list of DLA positions instead
        f=h5py.File(self.savefile,'r+')
        ind = np.where(self.sub_nHI_grid > DLA_cut)
        ind_LLS = np.where((self.sub_nHI_grid > LLS_cut)*(self.sub_nHI_grid < DLA_cut))
        grp = f.create_group("abslists")
        grp.create_dataset("DLA",data=ind)
        grp.create_dataset("DLA_val",data=self.sub_nHI_grid[ind])
        grp.create_dataset("LLS",data=ind_LLS)
        grp.create_dataset("LLS_val",data=self.sub_nHI_grid[ind_LLS])
        f.close()


    def load_savefile(self,savefile=None):
        """Load data from a file"""
        #Name of savefile
        try:
            f=h5py.File(savefile,'r')
        except IOError:
            raise IOError("Could not open "+savefile)
        grid_file=f["HaloData"]
        self.redshift=grid_file.attrs["redshift"]
        self.omegam=grid_file.attrs["omegam"]
        self.omegal=grid_file.attrs["omegal"]
        self.hubble=grid_file.attrs["hubble"]
        self.box=grid_file.attrs["box"]
        self.npart=grid_file.attrs["npart"]
        self.ngrid = np.array(grid_file["ngrid"])
        try:
            self.sub_mass = np.array(grid_file["sub_mass"])
            self.ind=np.array(grid_file["halo_ind"])
            self.nhalo=np.size(self.ind)
            self.minpart = grid_file.attrs["minpart"]
        except KeyError:
            pass
        try:
            self.pDLA = grid_file.attrs["pDLA"]
            self.Rho_DLA = grid_file.attrs["Rho_DLA"]
            self.Omega_DLA = grid_file.attrs["Omega_DLA"]
            self.cddf_bins = np.array(grid_file["cddf_bins"])
            self.cddf_f_N = np.array(grid_file["cddf_f_N"])
        except KeyError:
            pass

        self.sub_cofm=np.array(grid_file["sub_cofm"])
        self.sub_radii=np.array(grid_file["sub_radii"])

        f.close()
        del grid_file
        del f



	try:
	    print "loading GridHIData "
	    this_savefile=self.savefile[  : self.savefile.index('H2')+2]+'.HI_'+str(0)+'.hdf5'
	    f=h5py.File(this_savefile,'r')
	    grp_grid=f["GridHIData"]
            self.nhalo = grp_grid.attrs["nslice"]
            self.ngrid = grp_grid.attrs["ngrid"]
	    self.sub_nHI_grid=np.zeros([self.nhalo, self.ngrid[0], self.ngrid[0] ])
	    f.close()

	    for iii in np.arange(self.nhalo):	
		print iii
	        this_savefile=self.savefile[  : self.savefile.index('H2')+2]+'.HI_'+str(iii)+'.hdf5'
	        f=h5py.File(this_savefile,'r')
	        grp_grid=f["GridHIData"]

	        tmp = np.array(grp_grid[str(iii)])
	        self.sub_nHI_grid[iii,:,:] = tmp		# need to fill in slice-by-slice.  Means I need to set once I figure out "nhaloes" (really nslices)

		f.close()
	except:
	    print "fail."
	    pass


        try:
            print "loading GridTotData (WARNING NSLICE HARD WIRED!)"
	    this_savefile=self.savefile[  : self.savefile.index('H2')+2]+'.Total_'+str(0)+'.hdf5'
            f=h5py.File(this_savefile,'r')
            grp_grid=f["GridTotData"]
            self.nhalo = grp_grid.attrs["nslice"]
            self.ngrid = grp_grid.attrs["ngrid"]
            self.sub_nTotal_grid=np.zeros([self.nhalo, self.ngrid[0], self.ngrid[0] ])
            f.close()

	    for iii in np.arange(self.nhalo):
		print iii
		this_savefile=self.savefile[  : self.savefile.index('H2')+2]+'.Total_'+str(iii)+'.hdf5'
                f=h5py.File(this_savefile,'r')
                grp_grid=f["GridTotData"]

                tmp = np.array(grp_grid[str(iii)])
                self.sub_nTotal_grid[iii,:,:] = tmp                # need to fill in slice-by-slice.  Means I need to set once I figure out "nhaloes" (really nslices)

		f.close()
        except:
            print "fail."
            pass

        try:
            print "loading GridZData"
            this_savefile=self.savefile[  : self.savefile.index('H2')+2]+'.Z_'+str(0)+'.hdf5'
            f=h5py.File(this_savefile,'r')
            grp_grid=f["GridZData"]
            self.nhalo = grp_grid.attrs["nslice"]
            self.ngrid = grp_grid.attrs["ngrid"]
            self.sub_nZ_grid=np.zeros([self.nhalo, self.ngrid[0], self.ngrid[0] ])
            f.close()

            for iii in np.arange(self.nhalo):
                print iii
                this_savefile=self.savefile[  : self.savefile.index('H2')+2]+'.Z_'+str(iii)+'.hdf5'
                f=h5py.File(this_savefile,'r')
                grp_grid=f["GridZData"]

                tmp = np.array(grp_grid[str(iii)])
                self.sub_nZ_grid[iii,:,:] = tmp                # need to fill in slice-by-slice.  Means I need to set once I figure out "nhaloes" (really nslices)

                f.close()
        except:
            print "fail."
            pass

#===============================#
    def rho_LLS(self, thresh=17.0):
        """Compute rho_LLS, the sum of the mass in DLAs. 
           Units are 10^8 M_sun / Mpc^3 (comoving), like 0811.2003
        """
        try:
            return self.Rho_LLS
        except AttributeError:
            rho_LLS = self._rho_LLS(thresh)  						#Avg density in g/cm^3 (comoving) / a^3 = physical
            conv = 1e8 * self.SolarMass_in_g / (1e3 * self.UnitLength_in_cm)**3		# 1 g/cm^3 (physical) in 1e8 M_sun/Mpc^3	
            self.Rho_LLS = rho_LLS / conv
            return rho_LLS / conv

    def omega_LLS(self, thresh=17.0, upthresh=20.3, fact=1, gas=False, zup=100.0, zdown=-1000.0):
        """Compute Omega_LLS, the sum of the neutral (or total, if gas=True) gas in LLSs, divided by the critical density.
            Ω_LLS =  m_p * avg. column density / (1+z)^2 / length of column / rho_c / X_H
            Note: If we want the neutral hydrogen density rather than the gas hydrogen density, multiply by 0.76,
            the hydrogen mass fraction.			 """

	val = fact*self._rho_LLS(thresh=thresh, upthresh=upthresh, gas=gas, zup=zup, zdown=zdown)/self.rho_crit()		# gas=False=neutral H, gas=true=tot gas mass
	if gas:
	    self.Omega_LLS_tot=val
	else:
	    self.Omega_LLS = val

        return val


    def _rho_LLS(self, thresh=17.0, upthresh=20.3, gas=False, zup=100.0, zdown=-100.0):
        """Find the average density in DLAs in g/cm^3 (comoving). Helper for omega_DLA and rho_DLA."""
        #Average column density of HI in atoms cm^-2 (physical)
        try:
            self.sub_nHI_grid
        except AttributeError:
	    print "did not find nHI grid, need to load...."
            self.load_hi_grid()

        if thresh > 0:
	    condition_grid=self.sub_nHI_grid
	    if gas:
		mass_grid=self.sub_nTotal_grid
	    else:
                mass_grid=self.sub_nHI_grid
            mass = np.sum(10**mass_grid[np.where((condition_grid < upthresh)*(condition_grid > thresh)*(self.sub_nZ_grid > zdown)*(self.sub_nZ_grid < zup))])/np.size(mass_grid)
        else:
            mass = np.mean(10**self.sub_nHI_grid)
        mass *= self.protonmass /(1+self.redshift)**2			# Avg. Column density of HI in g cm^-2 (comoving)
        length = (self.box*self.UnitLength_in_cm/self.hubble/self.nhalo)	#Length of column in comoving cm
        return mass/length							#Avg density in g/cm^3 (comoving)


    def line_density_LLS(self, thresh=17.0):
        """Compute the line density, the total cells in LLSs divided by the total area, multiplied by d L / dX. This is dN/dX = l_DLA(z)
        """
        #P(hitting a DLA at random)
        LLSs = 1.*np.sum(self.sub_nHI_grid > thresh)
        size = 1.*np.sum(self.ngrid**2)
        pLLS = LLSs/size/self.absorption_distance()
        self.pLLS = pLLS
        return pLLS

    def line_density2_LLS(self,thresh=17.0, upthresh=20.3):
        """Compute the line density the other way, by summing the cddf. This is dN/dX = l_LLS(z)"""
        (_,  fN) = self.column_density_function(minN=thresh,maxN=upthresh)
        NHI_table = 10**np.arange(thresh, upthresh, 0.2)
        width =  np.array([NHI_table[i+1]-NHI_table[i] for i in range(0,np.size(NHI_table)-1)])
        return np.sum(fN*width)

    def omega_LLS2(self,thresh=17.0, upthresh=20.3, fact=1.0, dlogN=0.2):
        """Compute Omega_LLS the other way, by summing the cddf."""
        (center,  fN) = self.column_density_function(minN=thresh,maxN=upthresh, dlogN=dlogN)	  
        #f_N* NHI is returned, in amu/cm^2/dX
        NHI_table = 10**np.arange(thresh, upthresh, dlogN)
        width =  np.array([NHI_table[i+1]-NHI_table[i] for i in range(0,np.size(NHI_table)-1)])			# this guy has shape of 16
        dXdcm = self.absorption_distance()/((self.box/self.nhalo)*self.UnitLength_in_cm/self.hubble)
        return fact*self.protonmass*np.sum(fN*center*width)*dXdcm/self.rho_crit()/(1+self.redshift)**2

#==================================#
    def _find_particles_in_slab(self,ii,ipos,ismooth, mHI):
        """Find particles in the slab and convert their units to grid units"""
        jpos = self.sub_cofm[ii,0]
        jjpos = ipos[:,0]
        grid_radius = self.box/self.nhalo/2.
        indj = np.where(ne.evaluate("abs(jjpos-jpos) < grid_radius"))

        if np.size(indj) == 0:
            return (None, None, None)

        ipos = ipos[indj]
        ismooth = ismooth[indj]
        mHI = mHI[indj]

        #coords in grid units
        coords=fieldize.convert(ipos,self.ngrid[0],self.box)
        # Convert each particle's density to column density by multiplying by the smoothing length once (in physical cm)!
        cellspkpc=(self.ngrid[0]/self.box)
        if self.once:
            avgsmth=np.mean(ismooth)
            print ii," Av. smoothing length is ",avgsmth," kpc/h ",avgsmth*cellspkpc, "grid cells min: ",np.min(ismooth)*cellspkpc
            self.once=False
        #Convert smoothing lengths to grid coordinates.
        return (coords, ismooth*cellspkpc, mHI)

    def load_tmp(self):
        """	Load a partially completed file	"""
        print "Starting loading tmp file: "+str(self.tmpfile)
        f = h5py.File(self.tmpfile,'r')
        grp = f["GridHIData"]
        [ grp[str(i)].read_direct(self.sub_nHI_grid[i]) for i in xrange(0,self.nhalo)]
        location = f.attrs["file"]
        f.close()
        print "Successfully loaded tmp file. Next to do is:",location+1
        return location+1

    def _set_nHI_grid_single_file(self, file, gas=False, metallicity=False):
	star=cold_gas.RahmatiRT(self.redshift, self.hubble, molec=self.molec)

	start_time = time.time()
	f = h5py.File(file,"r")
        bar=f["PartType0"]
        ipos=np.array(bar["Coordinates"])
        mass=np.array(bar["Masses"])
        smooth = hsml.get_smooth_length(bar)
        if not gas:                 # Hydrogen mass fraction
            try:
                mass *= np.array(bar["GFM_Metals"][:,0])
            except KeyError:
                mass *= self.hy_mass
            mass *= star.get_reproc_HI(bar)
	else:
	    if metallicity:
		try:
		    mass *= np.array(bar["GFM_Metallicity"])
		except:
		    mass *= 0.0127

#        ipos   = ipos[   :10000, :]
#        smooth = smooth[ :10000   ]
#        mass   = mass[   :10000   ]

	end_time = time.time()

	print "Snapshot loading done!  took "+str(end_time-start_time)+" seconds (starting to gridize)"
	
	if not gas:		# normal, neutral hydrogen projections
            [self.sub_gridize_single_file(ii,ipos,smooth,mass,self.sub_nHI_grid) for ii in xrange(0,self.nhalo)]
	else:
	    if metallicity:
		[self.sub_gridize_single_file(ii,ipos,smooth,mass,self.sub_nZ_grid) for ii in xrange(0,self.nhalo)]
	    else:
	        [self.sub_gridize_single_file(ii,ipos,smooth,mass,self.sub_nTotal_grid) for ii in xrange(0,self.nhalo)]

        f.close()
        del ipos
        del mass
        del smooth


    def set_nHI_grid(self, gas=False, metallicity=False, start=0, comm=None, this_task=0, n_tasks=1):
        """Set up the grid around each halo where the HI is calculated.
        """
        self.once=True
        files = hdfsim.get_all_files(self.snapnum, self.snap_dir)
        files.reverse()
        end = np.min([np.size(files),self.end])

        for xx in xrange(start, end):
            if ((xx % n_tasks) == this_task):
                ff = files[xx]
	        print "Starting file "+str(ff)+" for nHI grid setup on task "+str(this_task)
	        self._set_nHI_grid_single_file(ff, gas=gas, metallicity=metallicity)


	print "task "+str(this_task)+" ready for MPI comm..."
	comm.Barrier()

	print self.sub_nHI_grid.shape

	for this_slice in np.arange(self.nhalo):				# for each slice	
	    if not gas:
		local_grid   = self.sub_nHI_grid[this_slice,:,:]
	    else:
		if metallicity:
		    local_grid   = self.sub_nZ_grid[this_slice,:,:]
		else:
		    local_grid   = self.sub_nTotal_grid[this_slice,:,:]

            global_grid  = np.zeros( (self.ngrid[this_slice], self.ngrid[this_slice]) )		# create a global grid
            comm.Barrier()
	    comm.Reduce(   local_grid ,	global_grid,	op=MPI.SUM,  root=0)

	    if this_task==0:
                if not gas:         # normal operation, looking at neutral hydrogen only
                    self.sub_nHI_grid[this_slice,:,:] = global_grid		# only makes sense for root task, but can be done for all
                else:
		    if metallicity:
			self.sub_nZ_grid[this_slice,:,:] = global_grid
		    else:
                        self.sub_nTotal_grid[this_slice,:,:] = global_grid



	if this_task==0:
	    print "final units (root task only)..."
            #Deal with zeros: 0.1 will not even register for things at 1e17.
            #Also fix the units:
            #we calculated things in internal gadget /cell and we want atoms/cm^2
            #So the conversion is mass/(cm/cell)^2

	    massg=self.UnitMass_in_g/self.hubble/self.protonmass
	    for ii in xrange(0,self.nhalo):
	        epsilon=2.*self.sub_radii[ii]/(self.ngrid[ii])*self.UnitLength_in_cm/self.hubble/(1+self.redshift)
	        if not gas:
                    self.sub_nHI_grid[ii]*=(massg/epsilon**2)
                    self.sub_nHI_grid[ii]+=0.1
                    np.log10(self.sub_nHI_grid[ii],self.sub_nHI_grid[ii])
	        else:
		    if metallicity:
                        self.sub_nZ_grid[ii]*=(massg/epsilon**2)
                        self.sub_nZ_grid[ii]+=0.2

			self.sub_nZ_grid[ii] /= 10.0**(self.sub_nTotal_grid[ii])
                        np.log10(self.sub_nZ_grid[ii],self.sub_nZ_grid[ii])
		    else:
                        self.sub_nTotal_grid[ii]*=(massg/epsilon**2)
                        self.sub_nTotal_grid[ii]+=0.1
                        np.log10(self.sub_nTotal_grid[ii],self.sub_nTotal_grid[ii])

	print "done with nHI loading routine..."
	
        return


    def sub_gridize_single_file(self,ii,ipos,ismooth,mHI,sub_nHI_grid,weights=None):
        """Helper function for sub_nHI_grid
            that puts data arrays loaded from a particular file onto the grid.
            Arguments:
                pos - Position array
                rho - Density array to be interpolated
                smooth - Smoothing lengths
                sub_grid - Grid to add the interpolated data to
        """
	start_time = time.time()
	end_time = time.time()
        (coords, ismooth, mHI) = self._find_particles_in_slab(ii,ipos,ismooth, mHI)
        if coords == None:
            return

        fieldize.sph_str(coords,mHI,sub_nHI_grid[ii],ismooth,weights=weights, periodic=True)

	npart=mHI.shape[0]
	del coords
	del ismooth
	del mHI
	gc.collect()

	end_time = time.time()
        print "fieldize done!  took "+str(end_time-start_time)+" seconds ("+str((end_time-start_time)/npart)+" seconds per particle)"

        return


    def set_stellar_grid(self):
        """Set up a grid around each halo containing the stellar column density
        """
        self.sub_star_grid=np.zeros([self.nhalo, self.ngrid[0],self.ngrid[0]])
        self.once=True
        #Now grid the HI for each halo
        files = hdfsim.get_all_files(self.snapnum, self.snap_dir)
        #Larger numbers seem to be towards the beginning
        files.reverse()
        for ff in files:
            f = h5py.File(ff,"r")
            print "Starting file for stellar grid setup",ff
            bar=f["PartType4"]
            ipos=np.array(bar["Coordinates"])
            #Get HI mass in internal units
            mass=np.array(bar["Masses"])
            smooth = np.array(bar["SubfindHsml"])
            [self.sub_gridize_single_file(ii,ipos,smooth,mass,self.sub_star_grid) for ii in xrange(0,self.nhalo)]
            f.close()
            #Explicitly delete some things.
            del ipos
            del mass
            del smooth
        #we calculated things in internal gadget /cell and we want atoms/cm^2
        #So the conversion is mass/(cm/cell)^2
        for ii in xrange(0,self.nhalo):
            massg=self.UnitMass_in_g/self.hubble*self.hy_mass/self.protonmass
            epsilon=2.*self.sub_radii[ii]/(self.ngrid[ii])*self.UnitLength_in_cm/self.hubble/(1+self.redshift)
            self.sub_star_grid[ii]*=(massg/epsilon**2)
        return

    def load_fast_tmp(self,start,key):
        """Not supported"""
        return start
    def save_fast_tmp(self,location,key):
        """Not supported"""
        return

    def set_zdir_grid(self, dlaind, gas=False, key="zpos", ion=-1):
        """Set up the grid around each halo where the HI is calculated.
        """
        star=cold_gas.RahmatiRT(self.redshift, self.hubble, molec=self.molec)
        self.once=True
        #Now grid the HI for each halo
        files = hdfsim.get_all_files(self.snapnum, self.snap_dir)
        #Larger numbers seem to be towards the beginning
        files.reverse()
        self.xslab = np.zeros_like(dlaind[0], dtype=np.float64)
        try:
            start = self.load_fast_tmp(self.start, key)
        except IOError:
            start = self.start
        end = np.min([np.size(files),self.end])
        for xx in xrange(start,end):
            ff = files[xx]
            f = h5py.File(ff,"r")
            print "Starting file for zdir grid setup",ff
            bar=f["PartType0"]
            ipos=np.array(bar["Coordinates"])
            #Get HI mass in internal units
            mass=np.array(bar["Masses"])
            if not gas:
                #Hydrogen mass fraction
                try:
                    mass *= np.array(bar["GFM_Metals"][:,0])
                except KeyError:
                    mass *= self.hy_mass
            nhi = star.get_reproc_HI(bar)
            ind = np.where(nhi > 1.e-3)
            ipos = ipos[ind,:][0]
            mass = mass[ind]
            #Get x * m for the weighted z direction
            if not gas:
                mass *= nhi[ind]
            if key == "zpos":
                mass*=ipos[:,0]
            elif key != "":
                mass *= self._get_secondary_array(ind,bar,key, ion)
            smooth = hsml.get_smooth_length(bar)[ind]
            for slab in xrange(self.nhalo):
                ind = np.where(dlaind[0] == slab)
                self.xslab[ind] += self.sub_list_grid_file(slab,ipos,smooth,mass,dlaind[1][ind], dlaind[2][ind])

            f.close()
            #Explicitly delete some things.
            del ipos
            del mass
            del smooth
            self.save_fast_tmp(start,key)

        #Fix the units:
        #we calculated things in internal gadget /cell and we want atoms/cm^2
        #So the conversion is mass/(cm/cell)^2
        massg=self.UnitMass_in_g/self.hubble/self.protonmass
        epsilon=2.*self.sub_radii[0]/(self.ngrid[0])*self.UnitLength_in_cm/self.hubble/(1+self.redshift)
        self.xslab*=(massg/epsilon**2)
        return self.xslab

    def _get_secondary_array(self, ind, bar, key, ion=1):
        """Get the array whose HI weighted amount we want to compute. Throws ValueError
        if key is not a desired species."""
        raise NotImplementedError("Not valid species")

    def sub_list_grid_file(self,ii,ipos,ismooth,mHI,yslab, zslab):
        """Like sub_gridize_single_file for set_zdir_grid
        """
        (coords, ismooth, mHI) = self._find_particles_in_slab(ii,ipos,ismooth, mHI)
        if coords == None:
            return np.zeros_like(yslab)

        slablist = yslab*int(self.ngrid[0])+zslab
        xslab = _Discard_SPH_Fieldize(slablist, coords, ismooth, mHI, np.array([0.]),True,int(self.ngrid[0]))
        return xslab

    def absorption_distance(self):
        """Compute X(z), the absorption distance per sightline (eq. 9 of Nagamine et al 2003)
        in dimensionless units, accounting for slicing the box."""
        #h * 100 km/s/Mpc in h/s
        h100=3.2407789e-18
        # in cm/s
        light=2.9979e10
        #Units: h/s   s/cm                        kpc/h      cm/kpc
        return h100/light*(1+self.redshift)**2*(self.box/self.nhalo)*self.UnitLength_in_cm


    def omega_DLA(self, thresh=20.3, upthresh=50., fact=1000):
        """Compute Omega_DLA, the sum of the neutral gas in DLAs, divided by the critical density.
            Ω_DLA =  m_p * avg. column density / (1+z)^2 / length of column / rho_c / X_H
            Note: If we want the neutral hydrogen density rather than the gas hydrogen density, multiply by 0.76,
            the hydrogen mass fraction.
            The Noterdaeme results are GAS MASS
        """
        #Avg density in g/cm^3 (comoving) divided by critical density in g/cm^3
        omega_DLA=fact*self._rho_DLA(thresh, upthresh)/self.rho_crit()
        self.Omega_DLA = omega_DLA
        return omega_DLA

    def _rho_DLA(self, thresh=20.3, upthresh=50.):
        """Find the average density in DLAs in g/cm^3 (comoving). Helper for omega_DLA and rho_DLA."""
        #Average column density of HI in atoms cm^-2 (physical)
        try:
            self.sub_nHI_grid
        except AttributeError:
            self.load_hi_grid()
        if thresh > 0:
            grids=self.sub_nHI_grid
            HImass = np.sum(10**grids[np.where((grids < upthresh)*(grids > thresh))])/np.size(grids)
        else:
            HImass = np.mean(10**self.sub_nHI_grid)
        #Avg. Column density of HI in g cm^-2 (comoving)
        HImass = self.protonmass * HImass/(1+self.redshift)**2
        #Length of column in comoving cm
        length = (self.box*self.UnitLength_in_cm/self.hubble/self.nhalo)
        #Avg density in g/cm^3 (comoving)
        return HImass/length

    def _rho_DLA_tot(self, thresh=20.3, upthresh=50.):
        """Find the average density in DLAs in g/cm^3 (comoving). Helper for omega_DLA and rho_DLA."""
        #Average column density of HI in atoms cm^-2 (physical)
        try:
            self.sub_nHI_grid
        except AttributeError:
            self.load_hi_grid()
        if thresh > 0:
            grids=self.sub_nTotal_grid
            HImass = np.sum(10**grids[np.where((grids < upthresh)*(grids > thresh))])/np.size(grids)
        else:
            HImass = np.mean(10**self.sub_nHI_grid)
        #Avg. Column density of HI in g cm^-2 (comoving)
        HImass = self.protonmass * HImass/(1+self.redshift)**2
        #Length of column in comoving cm
        length = (self.box*self.UnitLength_in_cm/self.hubble/self.nhalo)
        #Avg density in g/cm^3 (comoving)
        return HImass/length


    def get_omega_hi_mass_breakdown(self, rhohi=True):
        """Get the total HI mass density in DLAs in each halo mass bin.
        Returns Omega_DLA in each mass bin."""
        (halo_mass, _, _) = self._load_halo(0)
        self._get_sigma_DLA(0,2)
        ind = np.where(self.dla_halo >= 0)
        find = np.where(self.dla_halo < 0)
        masses = halo_mass[self.dla_halo[ind]]
        dlaval = self._load_dla_val(True)
        massbins = 10**np.arange(9,13)
        massbins[0] = 10**8
        massbins[-1] = 10**13
        nmassbins = np.size(massbins)-1
        fractions = np.zeros(nmassbins+2)
        for mm in xrange(nmassbins):
            mind = np.where((masses > massbins[mm])*(masses <= massbins[mm+1]))
            if rhohi:
                fractions[mm+1] = np.sum(10**dlaval[ind][mind])
            else:
                fractions[mm+1] = np.size(mind)
        #Field DLAs
        if rhohi:
            fractions[nmassbins+1] = np.sum(10**dlaval[find])
        else:
            fractions[nmassbins+1] = np.size(find)
        #Divide by total number of sightlines
        fractions /= (self.nhalo*self.ngrid[0]**2)
        if rhohi:
            #Avg. Column density of HI in g cm^-2 (comoving)
            fractions = self.protonmass * fractions/(1+self.redshift)**2
            #Length of column in comoving cm
            length = (self.box*self.UnitLength_in_cm/self.hubble/self.nhalo)
            #Avg Density in g/cm^3 (comoving) divided by rho_crit
            fractions = 1000*fractions/length/self.rho_crit()
        else:
            fractions /= self.absorption_distance()
        return (massbins, fractions)

    def rho_DLA(self, thresh=20.3):
        """Compute rho_DLA, the sum of the mass in DLAs. This is almost the same as the total mass in HI.
           Units are 10^8 M_sun / Mpc^3 (comoving), like 0811.2003
        """
        try:
            return self.Rho_DLA
        except AttributeError:
            #Avg density in g/cm^3 (comoving) / a^3 = physical
            rho_DLA = self._rho_DLA(thresh)  #*(1.+self.redshift)**3
            # 1 g/cm^3 (physical) in 1e8 M_sun/Mpc^3
            conv = 1e8 * self.SolarMass_in_g / (1e3 * self.UnitLength_in_cm)**3
            self.Rho_DLA = rho_DLA / conv
            return rho_DLA / conv

    def line_density(self, thresh=20.3):
        """Compute the line density, the total cells in DLAs divided by the total area, multiplied by d L / dX. This is dN/dX = l_DLA(z)
        """
        #P(hitting a DLA at random)
        try:
            return self.pDLA
        except AttributeError:
            DLAs = 1.*np.sum(self.sub_nHI_grid > thresh)
            size = 1.*np.sum(self.ngrid**2)
            pDLA = DLAs/size/self.absorption_distance()
            self.pDLA = pDLA
            return pDLA

    def line_density2(self,thresh=20.3):
        """Compute the line density the other way, by summing the cddf. This is dN/dX = l_DLA(z)"""
        (_,  fN) = self.column_density_function(minN=thresh,maxN=24)
        NHI_table = 10**np.arange(thresh, 24, 0.2)
        width =  np.array([NHI_table[i+1]-NHI_table[i] for i in range(0,np.size(NHI_table)-1)])
        return np.sum(fN*width)

    def omega_DLA2(self,thresh=20.3):
        """Compute Omega_DLA the other way, by summing the cddf."""
        (center,  fN) = self.column_density_function(minN=thresh,maxN=24)
        #f_N* NHI is returned, in amu/cm^2/dX
        NHI_table = 10**np.arange(thresh, 24, 0.2)
        width =  np.array([NHI_table[i+1]-NHI_table[i] for i in range(0,np.size(NHI_table)-1)])
        dXdcm = self.absorption_distance()/((self.box/self.nhalo)*self.UnitLength_in_cm/self.hubble)
        return 1000*self.protonmass*np.sum(fN*center*width)*dXdcm/self.rho_crit()/(1+self.redshift)**2

    def get_dndm(self,minM,maxM):
        """Get the halo mass function from the simulations,
        in units of h^4 M_sun^-1 Mpc^-3.
        Parameters:
            minM and maxM are the sides of the bin to use.
        """
        #Number of halos in this mass bin in the whole box
        Nhalo=np.shape(np.where((self.real_sub_mass <= maxM)*(self.real_sub_mass > minM)))[1]
        Mpch_in_cm=3.085678e24
        #Convert to halos per Mpc/h^3
        Nhalo/=(self.box*self.UnitLength_in_cm/Mpch_in_cm)**3
        #Convert to per unit mass
        return Nhalo/(maxM-minM)


    def save_sigLLS(self):
        """Generate and save sigma_LLS to the savefile"""
        (self.real_sub_mass, self.sigLLS, self.field_lls, self.lls_halo) = self.find_cross_section(False, 0, 2.)
        f=h5py.File(self.savefile,'r+')
        mgrp = f["CrossSection"]
        try:
            del mgrp["sigLLS"]
            del mgrp["LLS_halo"]
        except KeyError:
            pass
        mgrp.attrs["field_lls"] = self.field_lls
        mgrp.create_dataset("sigLLS",data=self.sigLLS)
        mgrp.create_dataset("LLS_halo",data=self.lls_halo)
        f.close()

    def load_sigLLS(self):
        """Load sigma_LLS from a file"""
        f=h5py.File(self.savefile,'r')
        try:
            mgrp = f["CrossSection"]
            self.real_sub_mass = np.array(mgrp["sub_mass"])
            self.sigLLS = np.array(mgrp["sigLLS"])
            self.lls_halo = np.array(mgrp["LLS_halo"])
            self.field_lls = mgrp.attrs["field_lls"]
        except KeyError:
            f.close()
            raise
        f.close()

    def save_sigDLA(self):
        """Generate and save sigma_DLA to the savefile"""
        (self.real_sub_mass, self.sigDLA, self.field_dla,self.dla_halo) = self.find_cross_section(True, 0, 1.)
        f=h5py.File(self.savefile,'r+')
        try:
            mgrp = f.create_group("CrossSection")
        except ValueError:
            mgrp = f["CrossSection"]
        try:
            del mgrp["sub_mass"]
            del mgrp["sigDLA"]
            del mgrp["DLAzdir"]
            del mgrp["DLA_halo"]
        except KeyError:
            pass
        mgrp.attrs["field_dla"] = self.field_dla
        mgrp.create_dataset("sub_mass",data=self.real_sub_mass)
        mgrp.create_dataset("sigDLA",data=self.sigDLA)
        mgrp.create_dataset("DLA_halo",data=self.dla_halo)
        mgrp.create_dataset("DLAzdir",data=self.dla_zdir)
        f.close()

    def load_sigDLA(self):
        """Load sigma_DLA from a file"""
        f=h5py.File(self.savefile,'r')
        try:
            mgrp = f["CrossSection"]
            self.real_sub_mass = np.array(mgrp["sub_mass"])
            self.sigDLA = np.array(mgrp["sigDLA"])
            self.dla_halo = np.array(mgrp["DLA_halo"])
            self.field_dla = mgrp.attrs["field_dla"]
        except KeyError:
            f.close()
            raise
        f.close()

    def find_cross_section(self, dla=True, minpart=0, vir_mult=1):
        """Find the number of DLA cells within dist virial radii of
           each halo resolved with at least minpart particles.
           If within the virial radius of multiple halos, use the most massive one."""
        (halo_mass, halo_cofm, halo_radii, sub_pos, sub_radii, sub_index) = self._load_halo(0, True)
        dlaind = self._load_dla_index(dla)
        #Computing z distances
        xslab = self._get_dla_zpos(dlaind,dla)
        self.dla_zdir=xslab
        dla_cross = np.zeros_like(halo_mass)
        celsz = 1.*self.box/self.ngrid[0]
        yslab = (dlaind[1]+0.5)*celsz
        zslab = (dlaind[2]+0.5)*celsz
        assigned_halo = np.zeros_like(yslab, dtype=np.int32)
        assigned_halo-=1
        print "Starting find_halo_kernel"
        field_dla = _find_halo_kernel(self.box, halo_cofm,halo_radii,halo_mass, sub_pos, sub_radii, sub_index, xslab, yslab, zslab,dla_cross, assigned_halo)
        print "max = ",np.max(dla_cross)," field dlas: ",100.*field_dla/np.shape(dlaind)[1]
        #Convert from grid cells to kpc/h^2
        dla_cross*=celsz**2
        return (halo_mass, dla_cross, 100.*field_dla/np.shape(dlaind)[1],assigned_halo)

    def _get_dla_zpos(self,dlaind,dla=True):
        """Load or compute the depth of the DLAs"""
        if dla == False:
            raise NotImplementedError("Does not work for LLS")
        f=h5py.File(self.savefile,'r')
        try:
            xslab = np.array(f["CrossSection"]["DLAzdir"])
        except KeyError:
            xhimass = self.set_zdir_grid(dlaind)
            xslab = xhimass/10**self._load_dla_val(dla)
        f.close()
        return xslab

    def _load_dla_index(self, dla=True):
        """Load the positions of DLAs or LLS from savefile"""
        #Load the DLA/LLS positions
        f=h5py.File(self.savefile,'r')
        grp = f["abslists"]
        #This is needed to make the dimensions right
        if dla:
            ind = (grp["DLA"][0,:],grp["DLA"][1,:],grp["DLA"][2,:])
        else:
            ind = (grp["LLS"][0,:],grp["LLS"][1,:],grp["LLS"][2,:])
        f.close()
        return ind

    def _load_dla_val(self, dla=True):
        """Load the values of DLAs or LLS from savefile"""
        #Load the DLA/LLS positions
        f=h5py.File(self.savefile,'r')
        grp = f["abslists"]
        #This is needed to make the dimensions right
        if dla:
            nhi = np.array(grp["DLA_val"])
        else:
            nhi = np.array(grp["LLS_val"])
        f.close()
        return nhi

    def _load_halo(self, minpart=0, subhalo=False):
        """Load a halo catalogue:
        minpart - does nothing
        subhalo - shall I load a subhalo catalogue"""
        #This is rho_c in units of h^-1 M_sun (kpc/h)^-3
        rhom = 2.78e+11* self.omegam / (1e3**3)
        #Open subfind catalogue
        subs=subfindhdf.SubFindHDF5(self.snap_dir, self.snapnum)
        #Store the indices of the halos we are using
        #Get particle center of mass, use group catalogue.
        halo_cofm=subs.get_grp("GroupPos")
        #halo masses in M_sun/h
        halo_mass=subs.get_grp("GroupMass")*self.UnitMass_in_g/self.SolarMass_in_g
        #r200 in kpc/h (comoving).
        halo_radii = subs.get_grp("Group_R_Crit200")
        if subhalo:
            sub_radii =  subs.get_sub("SubhaloHalfmassRad")
            sub_pos =  subs.get_sub("SubhaloPos")
            sub_index = subs.get_sub("SubhaloGrNr")
            return (halo_mass, halo_cofm, halo_radii, sub_pos, sub_radii, sub_index)
        else:
            return (halo_mass, halo_cofm, halo_radii)

    def column_density_function(self,dlogN=0.1, minN=16, maxN=24., maxM=None,minM=None):
        """
        This computes the DLA column density function, which is the number
        of absorbers per sight line with HI column densities in the interval
        [NHI, NHI+dNHI] at the absorption distance X.
        Absorption distance is simply a single simulation box.
        A sightline is assumed to be equivalent to one grid cell.
        That is, there is presumed to be only one halo in along the sightline
        encountering a given halo.

        So we have f(N) = d n_DLA/ dN dX
        and n_DLA(N) = number of absorbers per sightline in this column density bin.
                     1 sightline is defined to be one grid cell.
                     So this is (cells in this bins) / (no. of cells)
        ie, f(N) = n_DLA / ΔN / ΔX
        Note f(N) has dimensions of cm^2, because N has units of cm^-2 and X is dimensionless.

        Parameters:
            dlogN - bin spacing
            minN - minimum log N
            maxN - maximum log N
            maxM - maximum log M halo mass to consider		# what does this mean here?
            minM - minimum log M halo mass to consider

        Returns:
            (NHI, f_N_table) - N_HI (binned in log) and corresponding f(N)
        """
        NHI_table = 10**np.arange(minN, maxN, dlogN)
        if maxM == None and minM == None:
            try:
                if np.size(NHI_table)-1 == np.size(self.cddf_bins):
                    return (self.cddf_bins, self.cddf_f_N)
                else:
                    raise AttributeError
            except AttributeError:
                (self.cddf_bins, self.cddf_f_N)= self._calc_cddf(NHI_table, minN, maxM, minM)
                return (self.cddf_bins, self.cddf_f_N)
        else:
            return self._calc_cddf(NHI_table, minN, maxM, minM)


    def _calc_cddf(self,NHI_table, minN=17, maxM=None,minM=None):
        """Does the actual calculation for the CDDF function above"""
        try:
            grids = self.sub_nHI_grid
        except AttributeError:
            self.load_hi_grid()
            grids = self.sub_nHI_grid

        center = np.array([(NHI_table[i]+NHI_table[i+1])/2. for i in range(0,np.size(NHI_table)-1)])
        width  = np.array([NHI_table[i+1]-NHI_table[i] for i in range(0,np.size(NHI_table)-1)])
        #Grid size (in cm^2)
        dX=self.absorption_distance()
        tot_cells = np.sum(self.ngrid**2)
        if maxM != None:
            raise NotImplementedError("Splitting by mass no longer works")
            ind = np.where((self.halo_mass < 10.**maxM)*(self.halo_mass > 10.**minM))
            tot_f_N = np.histogram(np.ravel(grids[ind]),np.log10(NHI_table))[0]
        else:
            ind = np.where(grids >= minN)
            tot_f_N = np.histogram(np.ravel(grids[ind]),np.log10(NHI_table))[0]
        tot_f_N=(tot_f_N)/(width*dX*tot_cells)
        return (center, tot_f_N)

    def get_dla_metallicity(self):
        """Get the DLA metallicities from the save file, as Z/Z_sun.
        """
        try:
            return self.dla_metallicity-np.log10(self.solarz)
        except AttributeError:
            ff = h5py.File(self.savefile,"r")
            self.dla_metallicity = np.array(ff["Metallicities"]["DLA"])
            ff.close()
            return self.dla_metallicity-np.log10(self.solarz)

    def get_ion_metallicity(self, species,ion, dla=True):
        """Get the metallicity derived from an ionic species"""
        f=h5py.File(self.savefile,'r')
        grp = f[species][str(ion)]
        #This is needed to make the dimensions right
        if dla:
            spec = np.array(grp["DLA"])
        else:
            spec = np.array(grp["LLS"])
        f.close()
        #Divide by H column density
        hi = self._load_dla_val(dla)
        met = np.log10(spec+0.01)-hi-np.log10(self.solar[species])
        return met

    def get_lls_metallicity(self):
        """Get the LLS metallicities from the save file, as Z/Z_solar
        """
        try:
            return self.lls_metallicity-np.log10(self.solarz)
        except AttributeError:
            ff = h5py.File(self.savefile,"r")
            self.lls_metallicity = np.array(ff["Metallicities"]["LLS"])
            ff.close()
            return self.lls_metallicity-np.log10(self.solarz)

    def get_sDLA_fit(self):
        """Fit an broken power law profile based function to sigma_DLA as binned."""
        ind = np.where(self.real_sub_mass > 0)
        minM = np.min(self.real_sub_mass[ind])
        maxM = np.max(self.real_sub_mass)
        bins=30
        mass=np.logspace(np.log10(minM),np.log10(maxM),num=bins)
        bin_mass = np.array([(mass[i]+mass[i+1])/2. for i in xrange(0,np.size(mass)-1)])
        (sDLA,loq,upq)=self.get_sigma_DLA_binned(mass,sigma=68)
        ind = np.where((sDLA > 0)*(loq+upq > 0)*(bin_mass > 10**8.5))
        err = (upq[ind]+loq[ind])/2.
        #Arbitrary large values if err is zero
        return powerfit(np.log10(bin_mass[ind]), np.log10(sDLA[ind]), np.log10(err), breakpoint=10)

    def get_sigma_DLA_binned(self,mass,DLA_cut=20.3,DLA_upper_cut=42.,sigma=95):
        """Get the median and scatter of sigma_DLA against mass."""
        if DLA_cut < 17.5:
            sigs = self.sigLLS
        else:
            sigs = self.sigDLA
        aind = np.where(sigs > 0)
        #plt.loglog(self.real_sub_mass[aind], self.sigDLA[aind],'x')
        amed=calc_binned_median(mass, self.real_sub_mass[aind], sigs[aind])
        aupq=calc_binned_percentile(mass, self.real_sub_mass[aind], sigs[aind],sigma)-amed
        #Addition to avoid zeros
        aloq=amed - calc_binned_percentile(mass, self.real_sub_mass[aind], sigs[aind],100-sigma)
        return (amed, aloq, aupq)

    def _get_sigma_DLA(self, minpart, dist):
        """Helper for halo_hist to correctly populate sigDLA, from a savefile if possible"""
        if minpart == 0 and dist == 2.:
            try:
                self.sigDLA
            except AttributeError:
                try:
                    self.load_sigDLA()
                except KeyError:
                    self.save_sigDLA()
        else:
            (self.real_sub_mass, self.sigDLA, self.field_dla, self.dla_halo) = self.find_cross_section(True, minpart, dist)

    def _get_sigma_LLS(self, minpart, dist):
        """Helper for halo_hist to correctly populate sigLLS, from a savefile if possible"""
        if minpart == 0 and dist == 2.:
            try:
                self.sigLLS
            except AttributeError:
                try:
                    self.load_sigLLS()
                except KeyError:
                    self.save_sigLLS()
        else:
            (self.real_sub_mass, self.sigLLS, self.field_lls, self.lls_halo) = self.find_cross_section(False, minpart, dist)

    def get_dla_impact_parameter(self, minM, maxM):
        """Get the distance from the parent halo as a fraction of rvir for each DLA"""
        (halo_mass, halo_cofm, halo_radii) = self._load_halo(0)
        self._get_sigma_DLA(0,2)
        dlaind = self._load_dla_index(True)
        #Halo positions
        ind = np.where(self.dla_halo >= 0)
        halopos = halo_cofm[self.dla_halo[ind]]
        #Computing z distances
        xslab = self._get_dla_zpos(dlaind,True)
        yslab = (dlaind[1]+0.5)*self.box*1./self.ngrid[0]
        zslab = (dlaind[2]+0.5)*self.box*1./self.ngrid[0]
        #Total distance
        xdist = np.abs(xslab[ind]-halopos[:,0])
        ydist = np.abs(yslab[ind]-halopos[:,1])
        zdist = np.abs(zslab[ind]-halopos[:,2])
        #Deal with periodics
        ii = np.where(xdist > self.box/2.)
        xdist[ii] = self.box-xdist[ii]
        ii = np.where(ydist > self.box/2.)
        ydist[ii] = self.box-ydist[ii]
        ii = np.where(zdist > self.box/2.)
        zdist[ii] = self.box-zdist[ii]
        distance = np.sqrt(xdist**2 + ydist**2 + zdist**2)

        ind2 = np.where((halo_mass[self.dla_halo[ind]] > minM)*(halo_mass[self.dla_halo[ind]] < maxM))
        return (distance/halo_radii[self.dla_halo[ind]])[ind2]
