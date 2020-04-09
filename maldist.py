import numpy as np
import matplotlib.pyplot as plt
dx,dy=0.1,0.05
R=1
RR=R*R
points=[(xa,ya) for xa in np.arange(-(R//dx)*dx-dx/2,R,dx) for ya in np.arange(-(R//dy)*dy-dy/2,R,dy) if (xa*xa+ya*ya)<RR]
print(f'Ideal number of points:{np.pi*RR/(dx*dy):g} \n'
      f'                Actual:{len(points):d}' )
mirrored_points=[(a*x,a*y) for x,y in points for a in ((2*RR/(x*x+y*y)-1)**0.5,)]
all_points=np.array(points+mirrored_points).transpose()
plt.gca().set_aspect('equal')
plt.xlim(xmin=-1.5,xmax=1.5)
plt.ylim(ymin=-1.5,ymax=1.5)
plt.plot(R*np.cos(np.linspace(0,2*np.pi,100)),R*np.sin(np.linspace(0,2*np.pi,100)),'black',lw=2)
plt.plot(2**0.5*R*np.cos(np.linspace(0,2*np.pi,100)),2**0.5*R*np.sin(np.linspace(0,2*np.pi,100)))
plt.scatter(all_points[0],all_points[1],marker='+')  
plt.show()
plt.close()

def sample(x,y,z,points,R):
  flow=1/(np.pi*z)*np.sum(np.exp(-np.sum((points.transpose()-(x,y))**2,1)/z))
  #To Do: need to add flow contribution from region outside sqrt(2)*R for large values of z
  return flow
  
x_sample=np.arange(0.0125,1.5,0.025) 
y_sample=np.arange(0.0125,1.5,0.025)
fd_4=np.array([[[x,y,dx*dy*sample(x,y,0.01,all_points,R)] for x in x_sample] for y in y_sample])
fd_2=np.hstack((fd_4[:,::-1,2],fd_4[:,:,2]))
flowdistribution=np.vstack((fd_2[::-1],fd_2))
plt.gca().set_aspect('equal')
plt.plot(R*np.cos(np.linspace(0,2*np.pi,100)),R*np.sin(np.linspace(0,2*np.pi,100)),'black',lw=2)
plt.contourf( np.hstack((-x_sample[::-1],x_sample)), np.hstack(( -y_sample[::-1],y_sample)), flowdistribution,np.arange(0.7,1.3,0.01))
plt.show()
plt.close()

flow_densities=[f for row in fd_4 for x,y,f in row  if (x*x+y*y)<RR]
plt.xlabel('Fraction of Column Crossection')
plt.ylabel('Relative Flow Density')
plt.plot([i/len(flow_densities) for i in range(len(flow_densities))],sorted(flow_densities))
plt.show()
plt.close()

flow_density_distribution=np.histogram(flow_densities,bins=40)
plt.plot(0.5*(flow_density_distribution[1][1:]+flow_density_distribution[1][:-1]),flow_density_distribution[0]/len(flow_densities))
plt.show()
plt.close()
