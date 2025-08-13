import numpy as np
import h5py
from scipy.interpolate import griddata
import scipy.constants

class analysis:
    """
    Object to analyse and visualize RIXS output. The object stores a 2D
    interpolation of the data on an excitation-emission grid and an
    excitation-loss grid.
    
    Args:
        rixs (object): rixs object
        w_core (np.array): 1D numpy array of the excitation energies of the RIXS
        calculation.
        grid (np.array): 2D numpy array of the number of interpolation points
        method (str): Optional method for the scipy.griddata interpolation.
        Defaults to 'linear'.
        fill_value(float): Optional fill_value for the scipy.griddata
        interpolation.  Defaults to 0.
        rescale (boolean): Optional rescale for scipy.griddate interpolation.
        Defaults to False.

        .. attribute:: w_core

            1D array of the excitation energies of the RIXS calulation.

        .. attribute:: xl
            
            1D array of the interpolated energy loss points.

        .. attribute:: xe
            
            1D array of the interpolated emission energy points.

        .. attribute:: y
            
            1D array of the interpolated excitation energy points.

        .. attribute:: zl

            2D array of the interpolated RIXS cross section on (xl,y)-grid
        
        .. attribute:: ze

            2D array of the interpolated RIXS cross section on (xe,y)-grid
    """

    def __init__(self, rixs, w_core, grid=np.array([]), method='linear',\
            fill_value=0, rescale=False):
        self.w_core=w_core
        if grid.shape[0] != 0:
            self.grid=grid
            self.method=method
            self.fill_value=fill_value
            self.rescale=rescale
            self.xl=np.linspace(min(rixs.w),max(rixs.w),num=grid[0])
            self.xe=np.linspace(min(self.w_core)-max(rixs.w),\
                    max(self.w_core)-min(rixs.w),grid[0])
            self.y=np.linspace(min(self.w_core),max(self.w_core),num=grid[1])
            #interpolate data
            self.__interpolate_data__(rixs)

    def __interpolate_data__(self,rixs):
        points_loss=np.zeros((rixs.w.shape[0]*self.w_core.shape[0],2))
        points_em=np.zeros((rixs.w.shape[0]*self.w_core.shape[0],2))
        z=np.zeros(rixs.w.shape[0]*self.w_core.shape[0])
        counter=0
        for i in range(self.w_core.shape[0]):
            for j in range(rixs.w.shape[0]):
                points_loss[counter,0]=rixs.w[j]
                points_loss[counter,1]=self.w_core[i]
                points_em[counter,0]=self.w_core[i]-rixs.w[j]
                points_em[counter,1]=self.w_core[i]
                z[counter]=rixs.spectrum[i,j]
                counter=counter+1

        #actual interpolation
        self.zl=griddata(points_loss, z, (self.xl[None,:], self.y[:,None]),\
                method=self.method, fill_value=self.fill_value, \
                rescale=self.rescale)
        
        self.ze=griddata(points_em, z, (self.xe[None,:], self.y[:,None]),\
                method=self.method, fill_value=self.fill_value, \
                rescale=self.rescale)
    
    def export(self,rixs,w_core,filepath):
        np.savez_compressed(filepath, xl=self.xl, xe=self.xe, zl=self.zl,\
                ze=self.ze, w=rixs.w, spectrum=rixs.spectrum, y=self.y,\
                w_emission=rixs.w_emission, w_core=w_core, grid=self.grid)
    
    @staticmethod
    def from_file(filepath):
        data_=np.load(filepath)
        rixs_=rixs()
        visual_=analysis(rixs=rixs_,w_core=data_['w_core'])
        visual_.xl=data_['xl']
        visual_.xe=data_['xe']
        visual_.y=data_['y']
        visual_.zl=data_['zl']
        visual_.ze=data_['ze']
        visual_.grid=data_['grid']
        rixs_.w=data_['w']
        rixs_.spectrum=data_['spectrum']
        rixs_.w_emission=data_['w_emission']

        return rixs_, visual_

class rixs:
    """
        Object to store output of BRIXS calculation and to generate RIXS spectra
        from the oscillator strength of the BRIXS calculation.

        Args:
            file (str):  rixs.h5 file path
            broad (float): Optional Lorentzian broadening for RIXS spectra in
            eV.  Defaults to None.
            freq (np.array): Optional 1D array of frequencies for energy loss in
            eV.  Defaults to [].

        .. attribute:: w

            1D numpy array of loss frequencies.

        .. attribute:: oscstr

            2D complex numpy array of RIXS oscillator strength. First dimension
            is the number of excitation energies, the second dimension the
            number of optical excitations.

        .. attribute:: spectrum

            2D real numpy array of RIXS spectrum. First dimension is the number
            of excitation energies, the second one is the number of frequencies
            for the energy loss, i.e. shape(spectrum)[1]=shape(freq)[0].
    """
    def __init__(self, file=None, broad=None, freq=np.array([])):
        self.delta_e=None
        self.oscstr=None
        self.w=None
        self.spectrum=None
        if file != None and broad !=None:
            self.file=file
            self.broad=broad
            self.__get_oscstr__()
        if freq.shape[0] != 0:
            self.w=freq
            self.set_spectrum()
            self.gen_spectrum()
        

    def __get_oscstr__(self):
        with h5py.File(self.file) as f:
            self.energy=np.asarray(list(f['evals']))
            nfreq=len(list(f['oscstr']))
            self.oscstr=[]
            for i in range(nfreq):
                nexciton=len(list(f['oscstr'][format(i+1,'04d')]))
                oscstr_p=[]
                for j in range(nexciton):
                    inter=f['oscstr'][format(i+1,'04d')][j][0]\
                            +1j*f['oscstr'][format(i+1,'04d')][j][1]
                    oscstr_p.append(inter)
                self.oscstr.append(oscstr_p)
            self.oscstr=np.asarray(self.oscstr)
        del oscstr_p
    
    def set_spectrum(self):
        #create matrix to hold Lorenztian broadening with broadening of
        #self.broad
        hartree=scipy.constants.physical_constants['Hartree energy in eV'][0]
        self.delta_e=np.zeros((self.w.shape[0],self.oscstr.shape[1]),\
                dtype=np.complex64)
        for i in range(self.delta_e.shape[0]):
            for j in range(self.delta_e.shape[1]):
                self.delta_e[i,j]=1.0/(self.w[i]/hartree-self.energy[j]\
                        +1j*self.broad/hartree)
    
    def gen_spectrum(self):
        self.spectrum=np.zeros((self.oscstr.shape[0],self.w.shape[0]))
        for i in range(self.spectrum.shape[0]):
            self.spectrum[i,:]=\
                    -1.0*np.matmul(self.delta_e,np.abs(self.oscstr[i,:])**2).imag
    
    def gen_emission_en(self,w_core):
        self.w_emission=[]
        #loop over core frequencies
        for i in range(len(w_core)):
            inter=[]
            #loop over energy losses
            for j in range(self.w.shape[0]):
                inter.append(w_core[i]-self.w[j])
            self.w_emission.append(inter)
        self.w_emission=np.asarray(self.w_emission)
