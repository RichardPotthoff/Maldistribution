# calculations for the liquid distribution in an infinite packing
# for an initial distribution of a circular area with a radius of R=1
# and a superficial flow rate of L=1
# 
from matplotlib import pyplot as plt
import numpy as np
import math
import json
import os
def sample(x,y,z,points,R):
  flow=1/(np.pi*z)*np.sum(np.exp(-np.sum((points.transpose()-(x,y))**2,1)/z))
  #To Do: need to add flow contribution from region outside sqrt(2)*R for large values of z
  return flow
def flow_profile(r,z,R=1,Rsample_max=None):
  if Rsample_max==None:
    Rsample_max=R
  RD=1
  RRD=RD*RD
#  dx,dy=np.array([0.05,0.05*3**0.5])*0.997267# ideal spacing to give 1459 points for R=1 do not change
  dx,dy=np.array([0.05,0.05*3**0.5])*0.997267
#  alpha=49.837/180*np.pi #ideal rotation for transition between erf and sample: do not change
  alpha=49.837/180*np.pi
  sa,ca=np.sin(alpha),np.cos(alpha)
  distributor=np.array([(ca*xa+sa*ya,-sa*xa+ca*ya) for i in (0,1) for xa in np.arange(-(RD//dx)*dx-i*dx/2,RD,dx) for ya in np.arange(-(RD//dy)*dy-i*dy/2,RD,dy)  if (xa*xa+ya*ya)<RRD]).transpose()
  z_erf_max=0.004 #z_erf_max=0.004 ideal transition point: do not change
  z_samp_min=0.0032 #z_samp_min=0.0032 ideal transition pont: do not change
  if z>z_samp_min:
    #the samplong method cannot be used for small values of z, because the 
    #individual distributor points are still distinguishable
    sampled_profile=np.ones(len(r))
    for i in range(len(r)-1,-1,-1):
      sampled_profile[i]=np.pi/len(distributor[0])*sample(r[i],0,z,distributor,1)
      if (sampled_profile[i]>1.0) or (((i+1)<(len(r)-1)) and (sampled_profile[i]<sampled_profile[i+1])):
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
  
datafilename=''#'flow_profiles.pkl'
if os.path.exists(datafilename):
  data=json.load(open(datafilename,'r',encoding='utf-8'))
  r=np.array(data['r'])/2**0.5
  z=np.array(data['z'])/2
  flow_profiles=np.array(data['flow_profiles'])
else:
  r=np.linspace(0,2**0.5,100)
  z=np.exp(np.linspace(np.log(5e-4),np.log(50),100))
  dx,dy=np.array([0.05,0.05*3**0.5])*0.997267
  RD=1
  RRD=RD*RD
  alpha=49.837/180*np.pi#49.85: transition z=0.003..0.004 49.83
  sa,ca=np.sin(alpha),np.cos(alpha)
  distributor=np.array([(ca*xa+sa*ya,-sa*xa+ca*ya) for i in (0,1) for xa in np.arange(-(RD//dx)*dx-i*dx/2,RD,dx) for ya in np.arange(-(RD//dy)*dy-i*dy/2,RD,dy)  if (xa*xa+ya*ya)<RRD])
  print(f'Ideal number of points:{np.pi*RRD/(0.5*(dx*dy)):g} \n'
        f'                Actual:{len(distributor):d}' )
        
  all_points=distributor.transpose()
  plt.gca().set_aspect('equal')
  plt.xlim((-1.01,1.01))
  plt.ylim((-1.01,1.01))
  plt.plot(RD*np.cos(np.linspace(0,2*np.pi,100)),RD*np.sin(np.linspace(0,2*np.pi,100)),'black',lw=2)
  #plt.plot(2**0.5*RD*np.cos(np.linspace(0,2*np.pi,100)),2**0.5*RD*np.sin(np.linspace(0,2*np.pi,100)))
  plt.scatter(distributor[:,0],distributor[:,1],marker='+')  
  plt.axis('off')
  plt.show()
  plt.close()   
  x_sample=np.arange(-0.025,1.25,dx/2) 
  y_sample=np.arange(-0.025,1.25,dy/4)
  R=RD
  RR=RRD
  points=distributor
  flowdistribution=np.array([[[x,y,np.pi*RR/len(points)*sample(x,y,0.0005,all_points,R)] for x in x_sample] for y in y_sample])
#  flowdistribution=np.array([[[x,y,np.pi*RR/len(points)*sample(x,y,0.004,all_points,R)] for x in x_sample] for y in y_sample])  
  pl1=plt
  plt.gca().set_aspect('equal')
  pl1.plot(R*np.cos(np.linspace(0,2*np.pi,100)),R*np.sin(np.linspace(0,2*np.pi,100)),'black',lw=2)
  phi=0/180*np.pi
  pl1.plot([0,1.5*R*np.cos(phi)],[0,1.5*R*np.sin(phi)],'black')
  pl1.contourf(x_sample, y_sample, flowdistribution[:,:,2], np.arange(0.0,2,0.01))
  pl1.xlim((0,1.2))
  pl1.ylim((0,1.2))
  pl1.axis('off')
  plt.show()
  plt.close()
#  flow_profiles=np.array([[np.pi/len(points)*sample(r_,0,z,all_points,1) for r_ in r] for z in z])
  flow_profiles=np.array([flow_profile(r,z) for z in z])

#  rra=np.ones((11,11))*xs**2+(np.ones((11,11))*ys**2).transpose()
#np.interp(rra,rr,fr)
  
#for row in flow_profiles:
#  for i in range(len(row)-2,-1,-1):
#    if row[i]<row[i+1]:
#      row[0:i+1]=1
#      break
#    elif row[i]>1:
#      row[i]=1
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
