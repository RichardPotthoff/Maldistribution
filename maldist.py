import numpy as np
import matplotlib.pyplot as plt
def sample(x,y,z,points,R):
  flow=1/(np.pi*z)*np.sum(np.exp(-np.sum((points.transpose()-(x,y))**2,1)/z))
  #To Do: need to add flow contribution from region outside sqrt(2)*R for large values of z
  return flow

dx,dy=0.1,0.05
RD=0.99975
RRD=RD*RD
distributor=np.array([(xa,ya) for xa in np.arange(-(RD//dx)*dx-dx/2,RD,dx) for ya in np.arange(-(RD//dy)*dy-dy/2,RD,dy) if (xa*xa+ya*ya)<RRD*0.995])
print(f'Ideal number of points:{np.pi*RRD/(dx*dy):g} \n'
        f'                Actual:{len(distributor):d}' )
        
plt.gca().set_aspect('equal')
plt.xlim((-1.01,1.01))
plt.ylim((-1.01,1.01))
plt.plot(RD*np.cos(np.linspace(0,2*np.pi,100)),RD*np.sin(np.linspace(0,2*np.pi,100)),'black',lw=2)
#plt.plot(2**0.5*RD*np.cos(np.linspace(0,2*np.pi,100)),2**0.5*RD*np.sin(np.linspace(0,2*np.pi,100)))
plt.scatter(distributor[:,0],distributor[:,1],marker='+')  
plt.axis('off')
plt.show()
plt.close()   
  
for i,offset in enumerate([(0,0),(0.0125,0)]):
  fig=plt.figure()
  for j,R in enumerate([RD,RD-0.0125,RD+0.0125]):
    RR=R*R
    points=distributor+offset
    mirrored_points=[(a*x,a*y) for x,y in points for a in ((2*RR/(x*x+y*y)-1)**0.5,)]
    all_points=np.vstack((points, mirrored_points)).transpose()
    x_sample=np.arange(-1.05,1.05,0.025) 
    y_sample=np.arange(-1.05,1.05,0.025)
    flowdistribution=np.array([[[x,y,np.pi*RR/len(points)*sample(x,y,0.02,all_points,R)] for x in x_sample] for y in y_sample])
    
    pl1=fig.add_subplot(2,3,j+1,adjustable='box', aspect='equal')
    for r in np.arange(R,1.5*R,0.01):
      pl1.plot(r*np.cos(np.linspace(0,2*np.pi,100)),r*np.sin(np.linspace(0,2*np.pi,100)),'white',lw=2)
      pass
    pl1.set_title(f'$D={2*R*1000:.0f}mm,$ $\Delta x={(offset[0]**2+offset[1]**2)**0.5*1000:.1f}mm$')
    pl1.plot(R*np.cos(np.linspace(0,2*np.pi,100)),R*np.sin(np.linspace(0,2*np.pi,100)),'black',lw=2)
    pl1.contourf(x_sample, y_sample, flowdistribution[:,:,2], np.arange(0.7,1.3,0.01))
    pl1.set_xlim((-1.02,1.02))
    pl1.set_ylim((-1.02,1.02))
    pl1.axis('off')
    
    flow_densities=[f for row in flowdistribution for x,y,f in row  if (x*x+y*y)<RR]
    
    pl2=fig.add_subplot(2,3,j+3+1)
    pl2.contourf([i/len(flow_densities) for i in range(len(flow_densities))], [0.7,1.3], np.vstack((sorted(flow_densities),sorted(flow_densities))), np.arange(0.7,1.3,0.01))
    pl2.plot([i/len(flow_densities) for i in range(len(flow_densities))], sorted(flow_densities), 'black',lw=3)
    if j!=0: 
      pl2.yaxis.set_visible(False)
    else:
      pl2.set(ylabel='rel. liquid load')
    if j==1:
      pl2.set(xlabel='fraction of column cross-section')
      
  fig.tight_layout(pad=0.3)
  plt.show()
  plt.close()
  
#flow_density_distribution=np.histogram(flow_densities,bins=40)
#plt.plot(0.5*(flow_density_distribution[1][1:]+flow_density_distribution[1][:-1]),flow_density_distribution[0]/len(flow_densities))
#plt.xlim((0.7,1.3))
#plt.ylim((0,1))
#plt.show()
#plt.close()

