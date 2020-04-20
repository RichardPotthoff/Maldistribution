import numpy as np
import matplotlib.pyplot as plt
import time
from  ideal_distributor import ideal_distribution
from polyline_circle_area import gridcellareas
def sample(x,y,z,points):
  (xp,yp,f)= points[:3] if len(points)>=3 else (*points[:2],1.0)
  flow=1/(np.pi*z)*np.sum(f*np.exp(-((xp-x)**2+(yp-y)**2)/z))
  return flow
  
def total_flow(distributor):
  return np.sum(distributor[2])
  
def flow_area(distribution,dx,dy,R):
  return np.minimum(1,np.maximum(0,0.4/dx*(R-(distribution[:,:,0]**2+distribution[:,:,1]**2)**0.5)+0.5))
  
dx,dy=0.1,0.05
RD=0.999750001#adjusted to match grid spacing to number of points: RD=(n_points*dx*dy/pi)**0.5
RRD=RD*RD
distributor=np.array([(xa,ya,1) for xa in np.arange(-(RD//dx)*dx-dx/2,RD,dx) for ya in np.arange(-(RD//dy)*dy-dy/2,RD,dy) if (xa*xa+ya*ya)<RRD*0.995]).transpose()#"if" condition adjusted to give 628 points
print(f'Ideal number of points:{np.pi*RRD/(dx*dy):g} \n'
        f'                Actual:{len(distributor[0]):d}' )
        
plt.gca().set_aspect('equal')
plt.xlim((-1.01,1.01))
plt.ylim((-1.01,1.01))
plt.plot(RD*np.cos(np.linspace(0,2*np.pi,100)),RD*np.sin(np.linspace(0,2*np.pi,100)),'black',lw=2)
#plt.plot(2**0.5*RD*np.cos(np.linspace(0,2*np.pi,100)),2**0.5*RD*np.sin(np.linspace(0,2*np.pi,100)))
plt.scatter(distributor[0],distributor[1],marker='+')  
plt.axis('off')
plt.show()
plt.close()   
for z in[0.02,100]:
  for i,offset in enumerate(np.array([[0,0],[0.0125,0]])):
    fig=plt.figure()
    for j,R in enumerate([RD,RD-0.0125,RD+0.0125]):
      RR=R*R
      points=np.copy(distributor)
      points[:2]+=np.ones(points.shape)[:2]*offset[:,np.newaxis]
      
      #The "mirrored points" simulate an impermeable wall by placing a mirror image
      #of the liquid distributor on the outside of the column wall.
      #This is only an approximation, but works well near the wall, where the wall can
      #be regarded as flat. Further away from the wall this method still works well,
      #either because the wall is too far away to have an effect, or, if the wall does have an effect, 
      #the errors cancel each other out (for the most part):
      mirrored_points=np.copy(points[:,(points[0]**2+points[1]**2)>0.5**2*RR])#leave out the points near the center
      mirrored_points[:2]*=(2*RR/(mirrored_points[0]**2+mirrored_points[1]**2)-1)**0.5
      
      all_points=np.hstack((points, mirrored_points))
      
      Δxs=0.025
      Δys=0.025
      rs=0.5*(Δxs**2+Δys**2)**0.5
      x_sample=np.arange(-R-Δxs,R+Δxs,Δxs) 
      y_sample=np.arange(-R-Δys,R+Δys,Δys)
      t1=time.time()
      flowdistribution=np.array([[[x,y,np.pi*RR/total_flow(points)*sample(x,y,z,all_points) if (x**2+y**2)<(R+rs)**2 else 1.0 ] for x in x_sample] for y in y_sample])
      t2=time.time()
      
      #Adding an infinite packing around the mirrored area with an initial uniform distribution 
      #counteracts 'leakage' through the wall at large values of 'z', and ensures
      #correct results for z->inf: 
      flowdistribution[:,:,2] += (1-ideal_distribution(x_sample,y_sample,z,RD=R*(1+total_flow(mirrored_points)/total_flow(points))**0.5,max_R_sample=R+rs))
      
      t3=time.time()
      print(f'execution time for "flowdistribution": {t2-t1:.3f}s, "ideal_distribution": {t3-t2:.3f}s')
      pl1=fig.add_subplot(2,3,j+1,adjustable='box', aspect='equal')
      for r in np.arange(R,1.5*R,0.01):
        pl1.plot(r*np.cos(np.linspace(0,2*np.pi,100)),r*np.sin(np.linspace(0,2*np.pi,100)),'white',lw=2)
        pass
      pl1.set_title(f'$D={2*R*1000:.0f}mm,$ $\Delta x={(offset[0]**2+offset[1]**2)**0.5*1000:.1f}mm$')
      pl1.plot(R*np.cos(np.linspace(0,2*np.pi,100)),R*np.sin(np.linspace(0,2*np.pi,100)),'black',lw=2)
      pl1.contourf(x_sample, y_sample, flowdistribution[:,:,2], np.arange(0.7,1.3,0.01),cmap='jet')
      pl1.set_xlim((-R*1.02,R*1.02))
      pl1.set_ylim((-R*1.02,R*1.02))
      pl1.axis('off')
      areas=gridcellareas(x_sample,y_sample,R,Δxs,Δys)
      flowdistribution=np.concatenate((flowdistribution,areas[:,:,np.newaxis]),axis=2)
      
      flow_spectrum=np.array(sorted([(f,a) for row in flowdistribution for x,y,f,a in row  if (x*x+y*y)<(R+rs)**2]))
      
      cummulative_area=np.cumsum(flow_spectrum[:,1])
      sampled_area=cummulative_area[-1]
      print(f'average flow:{sum(flow_spectrum[:,0]*flow_spectrum[:,1])/sampled_area}, sampled Diameter:{(4/np.pi*sampled_area)**0.5}')
      pl2=fig.add_subplot(2,3,j+3+1)
      pl2.set_title(f'$z = {z}$')
      pl2.contourf([i/len(flow_spectrum) for i in range(len(flow_spectrum))], [0.7,1.3], np.vstack((flow_spectrum[:,0],flow_spectrum[:,0])), np.arange(0.7,1.3,0.01),cmap='jet')
      
      pl2.plot(np.linspace(0,1,50),np.interp(np.linspace(0,1,50),cummulative_area/sampled_area,flow_spectrum[:,0]), 'black',lw=3)
      if j!=0: 
        pl2.yaxis.set_visible(False)
      else:
        pl2.set(ylabel='rel. liquid load')
      if j==1:
        pl2.set(xlabel='fraction of column cross-section')
        
    fig.tight_layout(pad=0.3)
    plt.show()
    plt.close()
    
