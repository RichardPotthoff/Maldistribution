#Methods for the calculation of the intersection area between a circle and an arbitrary
#polyline.
#The method uses the polarimeter method, with the center of the circle as pole.
#Counter-clockwise areas are positive, clockwise areas are negative.
#
#The method is simple: The line segments that intersect the circle are split at 
#the intersection points, and then the triangular areas between the center of the circle
#and each line segment are calculated. If the line segment lies outside the circle,
#the area of the corresponding circle segment is calculated and used instead.
#The total area is the sum of the areas calculated for each line (sub-) segment.
#
#*** work in progress, method not yet tested for correctness in all cases
import numpy as np
from matplotlib import pyplot as plt
from math import asin
import time
def line_circle_intersection_area(p1,p2,R):
  RR=R*R
  x1,y1,x2,y2=*p1,*p2
  r1=(x1**2+y1**2)**0.5
  r2=(x2**2+y2**2)**0.5
  dx,dy=p2[0]-p1[0],p2[1]-p1[1]
  drr=(dx**2+dy**2)
  D=x1*y2-x2*y1
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
    if len(result)==3 and result[1][3]>result[2][3]:
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

def gridcellareas(x_samples,y_samples,R):
  RR=R*R
  dx=(x_samples[-1]-x_samples[0])/(len(x_samples)-1)
  dy=(y_samples[-1]-y_samples[0])/(len(y_samples)-1)
  dr=(dx**2+dy**2)**0.5
  result=np.ones((len(x_samples),len(y_samples)))#everything is set to one
  ri=(x_samples**2+y_samples[:,np.newaxis]**2)**0.5
  result[ri>R]=0#everything outside the circle is set to zero
  borderix=np.where(np.abs(ri-R)<dr/2) #cells that are cut by the circumference 
  cellcoords=np.array([[-dx,-dy],[dx,-dy],[dx,dy],[-dx,dy],[-dx,-dx]])*0.5
  for i,j in zip(*borderix):
    x=x_samples[i]
    y=y_samples[j]
    corners=cellcoords+[x,y]
    result[i,j]=1/(dx*dy)*sum([line_circle_intersection_area(p1,p2,R)[0] for p1,p2 in 
    zip(corners[:-1,:],corners[1:,:])])
  return result
  
if __name__ =='__main__':
  R=1.6  
  x_samples=np.linspace(-2,2,81)
  y_samples=np.linspace(-2,2,81)
  dx=(x_samples[-1]-x_samples[0])/(len(x_samples)-1)
  dy=(y_samples[-1]-y_samples[0])/(len(y_samples)-1)
  t1=time.time()
  areas=gridcellareas(x_samples,y_samples,R)
  t2=time.time()
  
  #line_circle_intersection_area(*[[-2,-2],[0,-2]],1.5)
  poly1=np.array([[-2,1],[1,0.3],[1.5,1.3],[1.1,1.6]])
  #poly1=np.array([[ 0.5, -0.5],[ 1.5, -0.5],[ 1.5,  0.5],[ 0.5,  0.5]])
  #poly1=np.array([[-2,-2],[2,-2],[2,2],[-2,2]])
  #poly1=np.array(list(([[-2,-2],[2,-2],[2,2],[-2,2]])))
  lines=np.vstack((poly1,poly1[0,:]))
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
  if len(intersections)>0:
    intersections=np.array(intersections)
    plt.scatter(intersections[:,0],intersections[:,1])
  for p in intersections:
    plt.plot(*np.vstack(([0,0],p)).transpose())
  plt.plot(*[R*f(np.linspace(0,2*np.pi,200)) for f in [np.cos,np.sin]],lw=2)
  plt.gca().set_aspect('equal')
  plt.show()
  plt.close()
  
  plt.contourf(x_samples,y_samples, areas,np.linspace(0,2,20))
  plt.plot(*[R*f(np.linspace(0,2*np.pi,200)) for f in [np.cos,np.sin]],'black',lw=1)
  plt.gca().set_aspect('equal')
  plt.show()
  plt.close()
  print(f'time for gridcellareas: {t2-t1}')
  print(f'error: {(sum(sum(areas))*dx*dy/np.pi)**0.5/R-1}')
