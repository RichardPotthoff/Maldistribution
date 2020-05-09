#Methods for the calculation of the intersection area between a circle and an arbitrary
#polyline.
#The method uses the planimeter method, with the center of the circle as pole.
#Counter-clockwise areas are positive, clockwise areas are negative.
#
#The method is simple: The line segments that intersect the circle are split at 
#the intersection points, and then the triangular areas between the center of the circle
#and each line segment are calculated. If the line segment lies outside the circle,
#the area of the corresponding circle segment is calculated and used instead.
#The total area is the sum of the areas calculated for each line (sub-) segment.
#
import numpy as np
from matplotlib import pyplot as plt
from math import asin
import time
def line_circle_intersection_area(p1,p2,R):
  RR=R*R
  x1,y1,x2,y2=*p1,*p2
  D=x1*y2-x2*y1
  if D==0:# the line is aligned with the pole -> area=0 
    return 0.0,[] #return the result immediately to prevent division by zero later on
  r1=(x1**2+y1**2)**0.5
  r2=(x2**2+y2**2)**0.5
  dx,dy=p2[0]-p1[0],p2[1]-p1[1]
  drr=(dx**2+dy**2)
  Delta=RR*drr-D**2
  result=[[x1,y1,r1,0,0]]
  if Delta>0:
    sqrtDelta=Delta**0.5
    for i in [1,-1]:
      x=(D*dy+i*(-1 if dy<0 else 1)*dx*sqrtDelta)/drr
      y=(-D*dx+i*abs(dy)*sqrtDelta)/drr
      D_=x1*y-x*y1
      a=D_/D
      if a>0 and a<1: #check if point is between p1, and p2
        result.append([x,y,R,a,D_])#we save some intermediate results for the area calculation
    if len(result)==3 and result[1][3]>result[2][3]:#sort the results if there are 2 intersections
      p=result[1]
      result[1]=result[2]
      result[2]=p
  result.append([x2,y2,r2,1,D])
  A=0
  for i in range(1,len(result)):
    Aline2=(result[i][4]-result[i-1][4])
    if result[i][2]>R or result[i-1][2]>R: #if one of the point lies outside the circle
      sin_theta=Aline2/(result[i][2]*result[i-1][2])
      Acircle=asin(sin_theta)*RR/2
      A+= Acircle 
    else:
      A+=Aline2/2
  return A,[c[:2] for c in result[1:-1]]#return area and intersection point coordinates

def gridcellareas(x_samples=None,y_samples=None,R=None,dx=None,dy=None,cellcenters=None,cellcorners=None):
  RR=R*R
  if cellcorners==None:
    dx=(x_samples[-1]-x_samples[0])/(len(x_samples)-1) if dx==None else dx
    dy=(y_samples[-1]-y_samples[0])/(len(y_samples)-1) if dy==None else dy
    dr=0.5*(dx**2+dy**2)**0.5
    cellcorners=np.array([[-dx,-dy],[dx,-dy],[dx,dy],[-dx,dy]])*0.5
  else:
    cellcorners=np.array(cellcorners)
    dr=max(cellcorners[:,0]**2+cellcorners[:,1]**2)**0.5
  cellcorners=np.concatenate((cellcorners,cellcorners[:1]))
  cellarea=0.5*sum([p1[0]*p2[1]-p2[0]*p1[1] for p1,p2 in zip(cellcorners[:-1],cellcorners[1:])])
  if cellcenters==None:
    cellcenters=np.array([[(x,y) for x in x_samples] for y in y_samples])
  else:
    cellcenters=np.array(cellcenters)
  shape=cellcenters.shape
  cellcenters=cellcenters.reshape((-1,shape[-1]))#flatten array, except for last dimension
  result=np.ones(len(cellcenters))*cellarea#initialize the result (all cells are set to full area)
  rij=(cellcenters[:,0]**2+cellcenters[:,1]**2)**0.5#distance of cellcenter to center of circle
  result[rij>R]= 0#set all cells with ||cellcenter|| > R to zero
  
  borderix=np.where(np.abs(rij-R)<dr) #indices of the cells that may be cut by the circumference 
  #calculate the areas off all cells near the circumference exactly:
  for i in borderix[0]:
    corners=cellcorners+cellcenters[i,:2]
    result[i]=max(0,sum([line_circle_intersection_area(p1,p2,R)[0] for p1,p2 in 
    zip(corners[:-1],corners[1:])]))
  return result.reshape(shape[:-1]) #return array has the same shape as 'cellcenters', minus the last dimension: [[ [x,y], ... ], ...] -> [[ a, ... ], ...]
  
if __name__ =='__main__':
  R=1.6  
  #line_circle_intersection_area(*[[-2,-2],[0,-2]],1.5)
  poly=[
    np.array([[-2,1],[1,0.3],[1.5,1.3],[1.1,1.6]]),
    np.array([[-1,-1],[1,-1],[1,1],[-1,1]])*0.5+(0.5,0.1),
    np.array(list(reversed([[-1,-1],[1,-1],[1,1],[-1,1]])))*0.5+(0.5,0.1),
    np.array([[-1,-1],[1,-1],[1,1],[-1,1]])*2.0+(0.0,0.0),
    np.array([[-1,-1],[1,-1],[1,1],[-1,1]])*1.0+(0.5,0.1)
    ]
  for pl in poly:
    lines=np.vstack((pl,pl[0,:]))
    plt.plot(lines[:,0],lines[:,1])
    
    intersections=[]
    A=0
    for p1,p2 in zip(lines[:-1,:],lines[1:,:]):
      plt.plot(*np.vstack((p1,p2)).transpose(),lw=2)
      plt.plot(*np.vstack(([0,0],p1)).transpose())
      area,xpoints=line_circle_intersection_area(p1,p2,R)
      print(p1,p2,area)
      A+=area
      intersections+=xpoints
    print(f'polygon area inside the circle = {A}')
    plt.title(f'polygon area inside the circle = {A:0.3f}')
    if len(intersections)>0:
      intersections=np.array(intersections)
      plt.scatter(intersections[:,0],intersections[:,1])
    for p in intersections:
      plt.plot(*np.vstack(([0,0],p)).transpose())
    plt.plot(*[R*f(np.linspace(0,2*np.pi,200)) for f in [np.cos,np.sin]],lw=2)
    plt.gca().set_aspect('equal')
    plt.show()
    plt.close()
  
  x_samples=np.linspace(-2,2,21)
  y_samples=np.linspace(-2,2,21)
  dx=(x_samples[-1]-x_samples[0])/(len(x_samples)-1)
  dy=(y_samples[-1]-y_samples[0])/(len(y_samples)-1)
  
  areas=np.ones((len(x_samples),len(y_samples)))
  rij=(x_samples**2+y_samples[:,np.newaxis]**2)**0.5
  areas[rij>R]=0
  plt.title(f'error for "areas[rij>R]=0" : {(sum(sum(areas))*dx*dy/np.pi)**0.5/R-1:0.3g}')
  plt.contourf(x_samples,y_samples, areas,np.linspace(0,2,20))
  plt.plot(*[R*f(np.linspace(0,2*np.pi,200)) for f in [np.cos,np.sin]],'white',lw=1)
  plt.gca().set_aspect('equal')
  plt.show()
  plt.close()
  print(f'error: {(sum(sum(areas))*dx*dy/np.pi)**0.5/R-1}')
  
  t1=time.time()
  areas=gridcellareas(x_samples,y_samples,R,dx=dx,dy=dy)/(dx*dy)
  t2=time.time()
  plt.title(f'error for "gridcellareas(...)" : {(sum(sum(areas))*dx*dy/np.pi)**0.5/R-1:0.3g}')
  plt.contourf(x_samples,y_samples, areas,np.linspace(0,2,20))
  plt.plot(*[R*f(np.linspace(0,2*np.pi,200)) for f in [np.cos,np.sin]],'white',lw=1)
  plt.gca().set_aspect('equal')
  plt.show()
  plt.close()
  print(f'execution time for gridcellareas: {t2-t1}')
  print(f'error: {(sum(sum(areas))*dx*dy/np.pi)**0.5/R-1}')
  
  R1=R
  R2=0.8*R 
  x_samples=np.linspace(-2,2,81)
  y_samples=np.linspace(-2,2,81)
  dx=(x_samples[-1]-x_samples[0])/(len(x_samples)-1)
  dy=(y_samples[-1]-y_samples[0])/(len(y_samples)-1)
  t1=time.time()
  areas=1/(dx*dy)*(gridcellareas(R=R1,cellcenters=[[(x,y) for x in x_samples] for y in y_samples],cellcorners=np.array([[-dx,-dy],[dx,-dy],[dx,dy],[-dx,dy]])*0.5)-gridcellareas(x_samples,y_samples,R2,dx=dx,dy=dy))
  areas[areas<0]=0.0#rounding errors screw up contour plot: force areas<0 to 0
  t2=time.time()
  plt.title(f'error for "gridcellareas(...)" : {sum(sum(areas))*dx*dy/((R1**2-R2**2)*np.pi)-1}')
  plt.contourf(x_samples,y_samples, areas,np.linspace(0,2,20))
  for R in [R1,R2]:
    plt.plot(*[R*f(np.linspace(0,2*np.pi,200)) for f in [np.cos,np.sin]],'white',lw=1)
  plt.gca().set_aspect('equal')
  plt.show()
  plt.close()
  print(f'execution time for gridcellareas: {t2-t1}')
  print(f'error: {sum(sum(areas))*dx*dy/((R1**2-R2**2)*np.pi)-1}')
