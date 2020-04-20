from matplotlib import pyplot as plt
import numpy as np
import math
import json
import os
import time
def sample(x,y,z,points):
  (xp,yp,f)= points[:3] if len(points)>=3 else (*points[:2],1.0)
  flow=1/(np.pi*z)*np.sum(f*np.exp(-((xp-x)**2+(yp-y)**2)/z))
  return flow
 
def TriangularPattern(dx,RD=1,alpha=0,f=lambda x,y:1.0):
  RRD=RD*RD
  dy=dx*3**0.5
  sa,ca=np.sin(alpha),np.cos(alpha)
  return np.array([(ca*xa+sa*ya,-sa*xa+ca*ya,f(xa,ya)) for i in (0,1) for xa in np.arange(-(RD//dx)*dx-i*dx/2,RD,dx) for ya in np.arange(-(RD//dy)*dy-i*dy/2,RD,dy)  if (xa*xa+ya*ya)<RRD]).transpose()
  
def ideal_profile(r,z):
# calculations for the liquid distribution in an infinite packing
# for an initial distribution of a circular area with a radius of R=1
# and a superficial flow rate of L=1
# 
  RD=1
  RRD=RD*RD
#  dx,dy=np.array([0.05,0.05*3**0.5])*0.997267# ideal spacing to give 1459 points for R=1 do not change
#  alpha=49.837/180*np.pi #ideal rotation for transition between erf and sample: do not change
  distributor=TriangularPattern(0.05*0.997267,RD=1,alpha=49.837/180*np.pi,f=lambda x,y:1)
  z_erf_max=0.004 #z_erf_max=0.004 ideal transition point: do not change
  z_samp_min=0.0032 #z_samp_min=0.0032 ideal transition pont: do not change
  if z>z_samp_min:
    #the sampling method cannot be used for small values of z, because the 
    #individual distributor points are still distinguishable
    sampled_profile=np.ones(len(r))
    for i in range(len(r)-1,-1,-1):
      sampled_profile[i]=np.pi/len(distributor[0])*sample(r[i],0,z,distributor)
      if (sampled_profile[i]>1.0) or (((i+1)<(len(r)-1)) and (sampled_profile[i]<sampled_profile[i+1])):
#        print(f'Break at radius{r[i]}, z={z} sample={sampled_profile[i]}')
        sampled_profile[i]=1.0
        break#the remaining elements are already initialized to 1
    if not(z<z_erf_max):
       return sampled_profile
  if z<z_erf_max:
    #we can use the exact solution for the onedimensional case for small  values of z,
    #besause the curvature of the boundary is negligible
    calcd_profile=np.array([(math.erf(-(r_-1)/z**0.5)+1)/2 for r_ in r])
    if not(z>z_samp_min):
      return calcd_profile
  assert((z>z_samp_min) and (z<z_erf_max))
  # we are in the transition region between z_samp_min and z_erf_max
  # interpolate between the two methods and return the result
  dz=z_erf_max-z_samp_min
  return (z-z_samp_min)/dz*sampled_profile + (z_erf_max-z)/dz*calcd_profile 
  
def ideal_distribution(x_sample,y_sample,z,RD=1,max_R_sample=None):
  x_sample=np.array(x_sample)/RD
  y_sample=np.array(y_sample)/RD
  z=z/RD**2
  n=len(x_sample)
  m=len(y_sample)
  r_grid=(x_sample**2+y_sample[:,np.newaxis]**2)**0.5
  if max_R_sample==None:
    max_R_sample=np.amax(r_grid)
  else:
    max_R_sample=max_R_sample/RD
  n_samples=max(len(x_sample),len(y_sample),50)
  r=np.linspace(0,max_R_sample,n_samples)
  return(np.interp(r_grid,r,ideal_profile(r,z)))

#Run a test each time the module is imported, to make sure nothing has changed:
#Calculate the difference to the analytical one- dimensional solution near the wall (r=0.99)
#in the middle of the transition region (z=0.036).
#The expected result is 0.0001835956304843
assert ( (ideal_profile(r=[0.99],z=0.0036)[0]-(math.erf(-(0.99-1)/0.0036**0.5)+1)/2) - 0.0001835956304843).__abs__() < 1e-14, 'regression test failed: "ideal_profile(r=[0.99],z=0.0036)" returns an unexpected result' 
#execute the print statement below to print the assert statement for the current configuration      
#  print(f'assert abs((ideal_profile(r=[0.99],z=0.0036)[0]-(math.erf(-(0.99-1)/0.0036**0.5)+1)/2) -{ideal_profile([0.99],0.0036)[0]-(math.erf(-(0.99-1)/0.0036**0.5)+1)/2:.16f})<1e-14, \'regression test failed: "ideal_profile(r=[0.99],z=0.0036)" returns an unexpected result\' ')
 
if __name__=='__main__':
#test cases and graphical results for functions defined in this module
  r=np.linspace(0,2**0.5,100)
  z=np.exp(np.linspace(np.log(5e-4),np.log(50),100))
  dx=0.05*0.997267
  RD=1
  RRD=RD*RD
  distributor=TriangularPattern(dx,RD=RD,alpha=49.837/180*np.pi)
  print(f'Ideal number of points:{np.pi*RRD/(0.5*(dx*dx*3**0.5)):g} \n'
        f'                Actual:{len(distributor[0]):d}' )
        
  all_points=distributor
  plt.gca().set_aspect('equal')
  plt.xlim((-1.01,1.01))
  plt.ylim((-1.01,1.01))
  plt.plot(*(RD*f(np.linspace(0,2*np.pi,100)) for f in (np.cos,np.sin)),'black',lw=2)
  plt.scatter(distributor[0],distributor[1],marker='+')  
  plt.axis('off')
  plt.show()
  plt.close()   
  x_sample=np.arange(-0.025,1.25,dx/2) 
  y_sample=np.arange(-0.025,1.25,dx/2)
  R=RD
  RR=RRD
  points=distributor
  flowdistribution=np.array([[[x,y,np.pi*RR/len(points[0])*sample(x,y,0.0005,all_points)] for x in x_sample] for y in y_sample])
  
  pl1=plt
  plt.gca().set_aspect('equal')
  pl1.plot(*(R*f(np.linspace(0,2*np.pi,100)) for f in (np.cos,np.sin)),'black',lw=2)
  phi=0/180*np.pi
  pl1.plot([0,1.5*R*np.cos(phi)],[0,1.5*R*np.sin(phi)],'black')
  pl1.contourf(x_sample, y_sample, flowdistribution[:,:,2], np.arange(0.0,2,0.01))
  pl1.xlim((0,1.2))
  pl1.ylim((0,1.2))
  pl1.axis('off')
  plt.show()
  plt.close()
#  flow_profiles=np.array([[np.pi/len(points)*sample(r_,0,z,all_points,1) for r_ in r] for z in z])
  t1=time.time()
  flow_profiles=np.array([ideal_profile(r,z) for z in z])
  print(f'time for flow_profile:{(time.time()-t1)}')
  
  for fp,z_ in zip(flow_profiles,z):
    plt.plot(r,fp,'blue')
    plt.plot(r,[(math.erf(-(r_-1)/z_**0.5)+1)/2 for r_ in r],'red')
    plt.plot(r,[1/(z_+1)*np.exp(-r_*r_/(z_+1))for r_ in r],'green')
  #  plt.plot(r,[1-(1/(2000/z**4*np.exp(-r_*r_/z_**0.25)+1)) for r_ in r for z_ in (z+2**0.5,)],'blue')
  plt.semilogy()
  plt.ylim(ymin=1e-4,ymax=2)
  #plt.show() #skip showing results if commented out
  plt.close()
  for fp,r_ in zip(flow_profiles.transpose(),r):
    if r_>=0 and r_<2:
      plt.plot(z,fp,'blue')
  #    plt.plot(z[(z>0.001)*(z<10)],fp[(z>0.001)*(z<10)],'blue')
  #    plt.plot(z[z<0.02],[((math.erf(-(r_-1)/z_**0.5)+1)/2) for z_ in z[z<0.02]],'red')
  #    plt.plot(z[z>10],[1/(z_+0.4)*np.exp(-r_*r_/(z_+0.4))for z_ in z[z>10]],'green')
  #plt.semilogx()
  plt.loglog()
  plt.ylim(ymin=1e-4,ymax=2)
  plt.show() #skip if commented out
  plt.close()
  plt.plot(r,[(math.erf(-(r_-1)/z[14]**0.5)+1)/2-flow_profiles[14][i] for i,r_ in enumerate(r)],'lime')
  plt.plot(r,[(math.erf(-(r_-1)/z[15]**0.5)+1)/2-flow_profiles[15][i] for i,r_ in enumerate(r)],'green')
  plt.plot(r,[(math.erf(-(r_-1)/z[16]**0.5)+1)/2-flow_profiles[16][i] for i,r_ in enumerate(r)],'blue')
  plt.plot(r,[(math.erf(-(r_-1)/z[17]**0.5)+1)/2-flow_profiles[17][i] for i,r_ in enumerate(r)],'red')
  plt.plot(r,[(math.erf(-(r_-1)/z[18]**0.5)+1)/2-flow_profiles[18][i] for i,r_ in enumerate(r)],'black')
  plt.plot(r,[(math.erf(-(r_-1)/z[19]**0.5)+1)/2-flow_profiles[19][i] for i,r_ in enumerate(r)],'orange')
  plt.ylim((-0.001,0.001))
  plt.show()
  plt.close()
  for R in [1,1.5]:
    for z in [0.1]:
      x_sample=np.linspace(-2,2,100)
      y_sample=np.linspace(-1.8,1.8,90)
      plt.gca().set_aspect('equal')
      plt.plot(*(R*f(np.linspace(0,2*np.pi,100)) for f in (np.cos,np.sin)),'black',lw=2)
      plt.contourf(x_sample,y_sample,1-ideal_distribution(x_sample,y_sample,z,R),np.linspace(0.,2.,40))
      plt.xlim((-2,2))
      plt.ylim((-2,2))
      plt.axis('off')
      plt.show()
      plt.close()
      

