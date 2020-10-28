import numpy as np
from scipy.integrate import romb
from scipy.special import lpn

class Scf(object):
    """
    Hachisu's Self Consistent Field method (Hachisu 1986, ApJs, 61, 479)

    Notes
    -----
    The code works with following dimensionless variables:
    * rho = rho / rho_c
    * R   = R / R_A
    * Phi = Phi / (G * R_A^2 * rho_c)
    * omg2 = omg2 / (G * rho_c)
    * alpha2 = c_s^2 / (G * R_A^2 * rho_c)

    TODO
    ----
    """
    ra = 1.0
    def __init__(self, alpha2=0.01, Trat=None, rb=-0.8, rmin=0.6, rmax=1.2,
                  nr=513, nt=513, nl=50):
        """
        Parameters
        ----------
        alpha2 : a dimensionless isothermal sound speed defined by
                 alpha2 = c_s^2 / (G * R_A^2 * rho_c)
        Trat   : a temperature ratio between the hot and cold medium.
                 If Trat=None, no external medium is assumed.
        rb     : boundary point of the SCF method. Note that ra = 1.
                 rb > 0 : spheroidal   rb < 0 : toroidal
        rmin   : minimum radius of the SCF mesh
        rmax   : maximum radius of the SCF mesh
        nr     : number of radial grid point
        nt     : number of polar grid point
        nl     : maximum degree of the Legendre polynomial
        """
        self.alpha2 = alpha2
        self.Trat = Trat
        self.rb = rb
        self.nr = nr
        self.nt = nt
        self.nl = nl
        self.r   = np.linspace(rmin, rmax, self.nr)
        self.mu  = np.linspace(0, 1, self.nt)
        self.R = self.r[:,None]*np.sqrt(1.0-self.mu[None,:]**2)
        self.z = self.r[:,None]*self.mu[None,:]
        self.plgdr = np.empty((self.nl, self.nt))
        for i,mu in enumerate(self.mu):
            self.plgdr[:,i] = lpn(self.nl-1, mu)[0]
        l = np.arange(0, nl)
        self.rl00 = self.r[None,:]**l[:,None]
        self.rlp1 = self.rl00*self.r[None,:]
        self.rlp2 = self.rlp1*self.r[None,:]
        self.r1ml = self.r[None,:]/self.rl00
        self.idA = np.where(abs(self.r-self.ra)
                    == min(abs(self.r-self.ra)))[0][0]
        self.idB = np.where(abs(self.r-abs(self.rb))
                    == min(abs(self.r-abs(self.rb))))[0][0]
    def set_ic(self):
        """
        Set initial condition for the SCF iteration

        Notes
        -----
        For toroidal configuration (rb < 0), the initial guess is homogeneous
        circular torus centered at (ra-rb)/2 with radius (ra+rb)/2

        TODO
        ----
        Initial guess for spherical configuration
        """
        rc = 0.5*(self.ra-self.rb)
        eta = np.sqrt(self.r[:,None]**2 + rc**2 - 2.0*self.r[:,None]*rc*np.sqrt(1.-self.mu[None,:]**2))
        rho = np.zeros((self.nr, self.nt))
        rho[eta < 0.5*(self.ra+self.rb)] = 1.0
        return rho
    def solve_poisson(self,rho):
        """
        Solve poisson equation via multipole expansion

        Parameters
        ----------
        rho : dimensionless density distribution on a polar grid (nr,nt)

        Returns
        -------
        Phi : dimensionless gravitational potential on a polar grid (nr,nt)
        """
        rhol = np.zeros((self.nl,self.nr))
        iext = np.zeros((self.nl,self.nr))
        iint = np.zeros((self.nl,self.nr))
        dmu = self.mu[1]-self.mu[0]
        hdr = 0.5*(self.r[1]-self.r[0])
        rhol[::2,:] = 2.0*romb(self.plgdr[::2,None,:]*rho[None,:,:], dx=dmu, axis=-1)
        for i in range(1,self.nr):
            iext[:,i] = iext[:,i-1] + (self.rlp2[:,i-1]*rhol[:,i-1]+self.rlp2[:,i]*rhol[:,i])*hdr
        for i in range(self.nr-2,-1,-1):
            iint[:,i] = iint[:,i+1] + (self.r1ml[:,i+1]*rhol[:,i+1]+self.r1ml[:,i]*rhol[:,i])*hdr
        Phi = -2*np.pi*(self.plgdr[:,None,:]*(self.rl00[:,:,None]*iint[:,:,None]+iext[:,:,None]/self.rlp1[:,:,None])).sum(axis=0)
        Phi[0,:]=Phi[1,:]
        return Phi
    def solve_hydro(self,Phi):
        """
        Given gravitational potential, find equilibrium density distribution
        and rotation velocity by solving hydrostatic equilibrium

        Parameters
        ----------
        Phi : dimensionless gravitational potential on a polar grid (nr,nt)

        Returns
        rho : dimensionless density distribution on a polar grid (nr,nt)
        prs : dimensionless pressure on a polar grid (nr,nt)
        omg2 : square of the angular velocity
        mask : numpy mask for the dense (cold) material
        """
        if (self.rb >= 0):
            omg2 = 2.0*(Phi[self.idA,0]-Phi[self.idB,-1])/(self.ra**2)
        else:
            omg2 = 2.0*(Phi[self.idA,0]-Phi[self.idB,0])/(self.ra**2-self.rb**2)
        Psi = -0.5*omg2*self.R**2
        C = Phi[self.idA,0] + Psi[self.idA,0]
        H = C - Phi - Psi
        mask = (H >= H[self.idA,0])
        hmax = H[mask].max()
        rho = np.zeros((self.nr,self.nt))
        prs = np.zeros((self.nr,self.nt))
        rho[mask] = np.exp((H[mask]-hmax)/self.alpha2)
        prs[mask] = self.alpha2*rho[mask]
        if (self.Trat != None):
            rho[~mask] = np.exp((H[~mask]-hmax)/(self.Trat*self.alpha2))
            rho[~mask] *= rho[self.idA,0]/self.Trat/rho[self.idA+1,0]
            prs[~mask] = self.Trat*self.alpha2*rho[~mask]
        return rho, prs, omg2, mask
    def run(self, tol=2e-8):
        """
        Run SCF iteration

        Parameters
        ----------
        tol : tolerance for the convergence
        """
        rho = self.set_ic()
        omg2 = 0.0; pomg2 = -1.0
        while abs(omg2-pomg2) > tol:
            pomg2 = omg2
            Phi = self.solve_poisson(rho)
            rho, prs, omg2, mask = self.solve_hydro(Phi)
            print("omg = {:6.8f}".format(np.sqrt(omg2)))
        return {"dens":rho, "prs":prs, "Phi":Phi, "omg2":omg2, "mask":mask}
