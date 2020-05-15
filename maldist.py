import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time
from  ideal_distributor import ideal_distribution
from polyline_circle_area import gridcellareas
from ntp import plotMcCabe,y_eq,ntp_a,xb_yd,ntp_s
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
plt.close()        
A4=(lambda A:(2**((-A*0.5)-0.25),2**((-A*0.5)+0.25)))(4)#

"""
plt.gca().set_aspect('equal')
plt.xlim((-1.01,1.01))
plt.ylim((-1.01,1.01))
plt.plot(RD*np.cos(np.linspace(0,2*np.pi,100)),RD*np.sin(np.linspace(0,2*np.pi,100)),'black',lw=2)
#plt.plot(2**0.5*RD*np.cos(np.linspace(0,2*np.pi,100)),2**0.5*RD*np.sin(np.linspace(0,2*np.pi,100)))
plt.scatter(distributor[0],distributor[1],marker='+')  
plt.axis('off')
plt.show()
plt.close()   
"""
for z,alpha,ntp in[(0.02,2.5,20),(0.02,1.5,20)]:
  for i,offset in enumerate(np.array([[0,0],[0.0125,0]])):
    plt.close()
    fig = plt.figure(1,(A4[0]/0.0254,A4[1]/0.0254))
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
      flowdistribution=np.array([[np.pi*RR/total_flow(points)*sample(x,y,z,all_points) if (x**2+y**2)<(R+rs)**2 else 1.0  for x in x_sample] for y in y_sample])
      t2=time.time()
      
      #Adding an infinite packing around the mirrored area with an initial uniform distribution 
      #counteracts 'leakage' through the wall at large values of 'z', and ensures
      #correct results for z->inf: 
      flowdistribution += (1-ideal_distribution(x_sample,y_sample,z,RD=R*(1+total_flow(mirrored_points)/total_flow(points))**0.5,max_R_sample=R+rs))
      
      t3=time.time()
      print(f'execution time for "flowdistribution": {t2-t1:.3f}s, "ideal_distribution": {t3-t2:.3f}s')
      pl1=fig.add_subplot(4,3,j+1,adjustable='box', aspect='equal')
      for r in np.arange(R,1.5*R,0.01):
        pl1.plot(r*np.cos(np.linspace(0,2*np.pi,100)),r*np.sin(np.linspace(0,2*np.pi,100)),'white',lw=2)
        pass
      pl1.set_title(f'$D={2*R*1000:.0f}\\mathrm{{mm}},\\ \\Delta x={(offset[0]**2+offset[1]**2)**0.5*1000:.1f}\\mathrm{{mm}}$')
      pl1.plot(R*np.cos(np.linspace(0,2*np.pi,100)),R*np.sin(np.linspace(0,2*np.pi,100)),'black',lw=2)
      pl1.contourf(x_sample, y_sample, flowdistribution, np.arange(0.7,1.3,0.01),cmap='jet')
      pl1.set_xlim((-R*1.02,R*1.02))
      pl1.set_ylim((-R*1.02,R*1.02))
      pl1.axis('off')
      areas=gridcellareas(x_sample,y_sample,R,Δxs,Δys)
      flowdistribution=np.concatenate((flowdistribution[:,:,np.newaxis],np.ones((*flowdistribution.shape,1)),areas[:,:,np.newaxis]),axis=2)
      
      flow_spectrum=np.array(sorted([(l/g,l,g,a,l*a,g*a) for row in flowdistribution for l,g,a in row  if a>0]))
      
      cummulative_area=np.cumsum(flow_spectrum[:,3])
      cummulative_L=np.cumsum(flow_spectrum[:,4])/cummulative_area[-1]
      cummulative_G=np.cumsum(flow_spectrum[:,5])/cummulative_area[-1]
      sampled_area=cummulative_area[-1]
      print(f'average flow:{cummulative_L[-1]}, sampled Diameter:{(4/np.pi*sampled_area)**0.5}, total flow:{cummulative_L[-1]*cummulative_area[-1]}')
      pl2=fig.add_subplot(4,3,j+3+1)
      pl2.set_title(f'$z = {z}$')
      pl2.contourf([i/len(flow_spectrum) for i in range(len(flow_spectrum))], [0.7,1.3], np.vstack((flow_spectrum[:,0],flow_spectrum[:,0])), np.arange(0.7,1.3,0.01),cmap='jet')
      LG_bins=np.linspace(flow_spectrum[:,0].min()-0.001,flow_spectrum[:,0].max()+0.001,11)
      A_bins=np.interp(LG_bins,flow_spectrum[:,0],cummulative_area,left=0)
      # L_G_A has a row for each sub-column: [Liquid, Gas, Area, x-coordinate for spectrum plot)]
      L_G_A=np.array([[L1-L0,G1-G0,A1-A0,0.5*(A0+A1)/cummulative_area[-1]] for L0,L1,G0,G1,A0,A1 in 
        zip(
        np.interp(A_bins[:-1],cummulative_area,cummulative_L,left=0),
        np.interp(A_bins[1:],cummulative_area,cummulative_L,left=0),
        np.interp(A_bins[:-1],cummulative_area,cummulative_G,left=0),
        np.interp(A_bins[1:],cummulative_area,cummulative_G,left=0),        
        A_bins[:-1],
        A_bins[1:]) 
        if A0!=A1])
      cmap = matplotlib.cm.get_cmap('jet')
      for Li,Gi,Ai,A_i in L_G_A:
        LGi=Li/Gi
        pl2.plot(A_i,LGi,color='black',markerfacecolor=cmap((LGi-0.7)/(1.3-0.7)),marker='o',zorder=10)
      for LGi,Ai in zip(LG_bins,A_bins):
        Ai=Ai/cummulative_area[-1]
        pl2.plot([0,Ai],[LGi,LGi],'black',ls='-',lw=1)
        pl2.plot([Ai,Ai],[0.7,LGi],'black',ls='-',lw=1)
      pl2.plot(np.linspace(0,1,50),np.interp(np.linspace(0,cummulative_area[-1],50),cummulative_area,flow_spectrum[:,0]), 'black',lw=3)
      pl2.set_xlim((0,1))
      pl2.set_ylim((0.7,1.3))
      if j!=0: 
        pl2.yaxis.set_visible(False)
      else:
        pl2.set(ylabel='rel. liquid load')
      if j==1:
        pl2.set(xlabel='fraction of column cross-section')
        
      pl3=fig.add_subplot(4,3,j+6+1,adjustable='box', aspect='equal')
      pl4=fig.add_subplot(4,3,j+9+1,adjustable='box', aspect='equal')
      x=np.linspace(0,1,30)
      for ax in [pl3,pl4]:
        ax.plot(x,y_eq(x,alpha),'black',lw=1)
        ax.set_xlim((0,1))
        ax.set_ylim((0,1))
        if j!=0: 
          ax.yaxis.set_visible(False)
      x_y=(1-alpha**0.5)/(1-alpha)
      yde=y_eq(x_y,alpha)
      for alpha,xb,yb,xd,yd,ntp,ax in [
        (alpha,0,0,x_y,x_y - (2*x_y - 1)/1.25,ntp,pl3),
        (alpha,1-x_y + (2*x_y - 1)/1.25,1-x_y,1,1,ntp,pl3),
        (alpha,0,0,x_y,yde,ntp,pl4),
        (alpha,1-(yde),1-x_y,1,1,ntp,pl4) ]:
          
        lg=(yd-yb)/(xd-xb)
        xb,yb,xd,yd=xb_yd(yb,xd,lg,alpha,ntp)
        ax.plot([xb,xd],[yb,yd],color='black',ls=':',zorder=10)
        cmap = matplotlib.cm.get_cmap('jet')
        Ltot=L_G_A[:,0].sum()
        Gtot=L_G_A[:,1].sum()
        yd_m=0
        xb_m=0
        for Li,Gi,Ai,_ in L_G_A:
          Li=Li/Ltot*lg
          Gi=Gi/Gtot
          LGi=Li/Gi
          cl=cmap((LGi/lg-0.7)/(1.3-0.7))
          xb_,yb_,xd_,yd_=xb_yd(yb,xd,LGi,alpha,ntp)
          ax.plot([xb_,xd_],[yb_,yd_],color=cl)
          yd_m+=yd_*Gi
          xb_m+=xb_*Li
        xb_m=xb_m/lg
        ax.plot([xb_m,xd],[yb,yd_m],color='black',marker='+',zorder=10)
        ax.text((xd+xb)/2,1-x_y,f'{ntp_a(alpha,xb_m,yb,xd,yd_m):.2f}',verticalalignment='center', horizontalalignment='center')
        if j==1 and lg<1:
          r=1/(1/lg-1)
          r_min=1/(1/x_y-2)
          ax.set(xlabel=f'$r={r/r_min:.2f}r_\\mathrm{{min}},\\ \\alpha={alpha:.2f},\\ ntp={ntp:.1f}$')
      
    fig.tight_layout(pad=0.3)
    plt.show()
#    plt.savefig(f"Maldist.pdf", papertype = 'a4', orientation = 'portrait', format = 'pdf')
    plt.close()
    
