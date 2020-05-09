import numpy as np
from matplotlib import pyplot as plt

def y_eq(x,alpha):
  return x*alpha/(x*(alpha-1)+1)
  
def dy_eq_dx(x,alpha):
  return alpha/(x*(alpha - 1) + 1)**2
  
def x_eq(y,alpha):
  return 1-y_eq(1-y,alpha)

def y_ol(x,x0,y0,lg):
  return y0+lg*(x-x0)
      
def integrate(f,x0,x1,n):
  n=2*((n+1)//2)#make sure n is even
  x=np.linspace(x0,x1,n+1)
  h=x[1]-x[0]
  y=f(x)
  return h/3*(y[0]+4*np.sum(y[1:-1:2])+2*np.sum(y[2:-2:2])+y[-1])#simpson's formula

def xb_yd(yb,xd,lg,alpha,ntp,test=False):
  yd_min=yb
  yed=y_eq(xd,alpha)
  yd_pinch=yed#pinch at xd
  xeb=x_eq(yb,alpha)
  yd_pinch_=yb+lg*(xd-xeb)#pinch at xb
  if (yd_pinch_<yd_pinch)!=(yd_pinch_<yd_min):
    yd_pinch=yd_pinch_
  xp=((alpha/lg)**0.5-1)/(alpha-1)#m=l/g 
  if (xp<xeb)!=(xp<xd):
    yep=y_eq(xp,alpha)
    yd_pinch_=yep+lg*(xd-xp)#pinch at m=l/g
    if (yd_pinch_<yd_pinch)!=(yd_pinch_<yd_min):
      yd_pinch=yd_pinch_
  def xb_yb_xd_yd(t):#return operating line with yd between yd_min and yd_pinch for t between -inf to inf
    yd=yd_min+(yd_pinch-yd_min)*1/(1+np.exp(-t))#use logistic function 
    xb=xd-(yd-yb)/lg
    return (xb,yb,xd,yd)
#  return xb_yb_xd_yd(newton(lambda t:ntp-ntp_a(alpha,*xb_yb_xd_yd(t)),0))
  if test: return xb_yb_xd_yd(ntp)
  return xb_yb_xd_yd(root(lambda t:ntp-ntp_a(alpha,*xb_yb_xd_yd(t)),-10,10))  
  
def newton(f,x,df=None,eps=1e-6):
  y=2*eps
  it=0
  while abs(y)>eps:
    it+=1
    y=f(x)
    x=x-(y/df(x) if df else y/((f(x+eps)-y)/eps))
  print(f'newton steps:{it}')
  return x
  
def root(f,x1,x2,eps=1e-6,maxit=100):
  y1=f(x1)
  y2=f(x2)
  if (y1>0)==(y2>0):
    raise Exception('root: no zero in interval')
  it=0
  while(abs(x1-x2)>eps):
      it+=1
      x=(x1+x2)/2
      y=f(x)
      if (y<0)==(y1<0):
        x1=x
        y1=y
      else:
        x2=x
        y2=y
  return x1 if abs(y1)<abs(y2) else x2
  
def ntp_a(alpha,xb,yb,xd,yd):
  lg=(yd-yb)/(xd-xb)
  xz=zeros(alpha,xb,yb,lg)
  yz=y_eq(xz,alpha)
  b=alpha**-0.5
  scale=1/b-b
  b1=b+xz[0]*scale
  b2=b+(1-yz[1])*scale
  alphaz=1/(b1*b2)
  x_scale=(1/b2-b1)
  y_scale=(1/b1-b2)
  xb_=(b-b1+xb*scale)/x_scale 
  yd_=1-(b-b2+(1-yd)*scale)/y_scale
#  xb_=(xb-xz[0])/(xz[1]-xz[0])
#  yd_=yd_=1-(yz[1]-yd)/(yz[1]-yz[0])
  return ((np.log(yd_/(1-yd_))-np.log(xb_/(1-xb_)))/np.log(alphaz)).real
  
def ntp_s(alpha,xb,yb,xd,yd):
  lg=(yd-yb)/(xd-xb)
  if (y_eq(xb,alpha)>y_ol(xb,xb,yb,lg))==(xd>xb):
    sign=1
    x=xb 
    y=yb
  else:
    sign=-1
    x=xd
    y=yd
  ntp=0
  while True:
    ye=y_eq(x,alpha)
    x_=xb+(ye-yb)/lg
    if x_==xd or x_==xb:
      ntp+=1
      break
    if (x_>xd)==(xd>xb):
      yde=y_eq(xd,alpha)
      m=(yde-ye)/(xd-x)
      ntp+=(xd-x)/(x_-x) if m/lg==1 else np.log((yde-yd)/(ye-y))/np.log(m/lg)
      break
    if (x_<xb)==(xd>xb):
      ybe=y_eq(xb,alpha)
      m=(ybe-ye)/(xb-x)
      ntp+=(xb-x)/(x_-x) if m/lg==1 else np.log((ybe-yb)/(ye-y))/np.log(m/lg)
      break 
    ntp+=1
    x=x_
    y=ye
  return sign*ntp
         
def ntp_i(alpha,xb,yb,xd,yd,intervals=21):
  lg=(yd-yb)/(xd-xb)
  def dntp(x): 
    y=yb+(x-xb)*lg
    ye=y_eq(x,alpha)
    m=dy_eq_dx(x,alpha)
    return (m-lg)*((np.log(m)-np.log(lg)) * (ye-y))**-1
    
  return integrate(dntp,xb,xd,intervals)
  
def zeros(alpha,x0,y0,lg):
  #calculate the intersection points between operating line and equilibrium line
  b=((alpha-1)*(x0*lg+(1-y0)) + 1 - lg )
  d=complex((b+2.0*lg)**2 -4.0*alpha*lg)**0.5
  return np.array([b - d, b + d])/(2.0*lg*(alpha-1))
  
def plotMcCabe(alpha,xb,yb,xd,yd,ax=None,limits=((0,1),(0,1))):
  (xmin,xmax),(ymin,ymax)=limits
  lg=(yd-yb)/(xd-xb)
  xz=zeros(alpha,xb,yb,lg)
  yz=y_eq(xz,alpha)
  def steps():
    x=xb if (y_eq(xb,alpha)>y_ol(xb,xb,yb,lg))==(xd>xb) else xd
    while True:
      y=y_ol(x,xb,yb,lg)
      yield(x,y)
      y=y_eq(x,alpha)
      yield(x,y)
      x=xb+(y-yb)/lg
      if (x>xd)==(xd>xb):
        yield(xd,y)
        break
      if (x<xb)==(xd>xb):
        yield(xb,y)
        break
  if ax==None:
    ax=plt.axes()
  x=np.linspace(xmin,xmax,51)
  ax.plot(x,y_eq(x,alpha), 'red',lw=2)
  ax.plot([xb,xd],[yb,yd],'black',marker='+')
  xp=((alpha/lg)**0.5-1)/(alpha-1)
  dy=y_eq(xp,alpha)-(yb+lg*(xp-xb))
  ax.plot([xb,xd],[yb+dy,yd+dy],'green')
  ax.plot([xp],[y_eq(xp,alpha)],'green',marker='o')
  st=np.array(list(steps()))
  ax.plot(xz,yz,'blue',ls=':',marker='o')
  ax.plot(st[:,0],st[:,1],'blue',marker='+')  
  ax.set_xlim(limits[0])
  ax.set_ylim(limits[1])

  
    
if __name__ == '__main__':
  A4=(lambda A:(2**((-A*0.5)-0.25),2**((-A*0.5)+0.25)))(4)#
  fig=plt.figure(figsize=(A4[1]/0.0254,A4[0]/0.0254))
  
  for i,(alpha,xb,yb,xd,yd) in enumerate([[3,0.5,0.5,0.9,0.9],[3,0.4,0.7,0.1,0.55],[1/2.5,0.4,0.15,0.7,0.4]]):
    ax1=fig.add_subplot(2,3,i+1,adjustable='box', aspect='equal')  
    ax2=fig.add_subplot(2,3,i+1+3,adjustable='box', aspect='equal')  
    ax1.set(ylabel='y')
    ax1.set(xlabel='x')
  #    ax2.set(ylabel='y')
    ax2.set(xlabel='x')
    plotMcCabe(alpha,xb,yb,xd,yd,ax=ax1)
    ax1.set_title(f'ntp={ntp_a(alpha,xb,yb,xd,yd).real:.5f}(a),{ntp_s(alpha,xb,yb,xd,yd):.5f}(s)')
    lg=(yd-yb)/(xd-xb)
    xz=zeros(alpha,xb,yb,lg)
    yz=y_eq(xz,alpha)
  #  x=np.linspace(0,1,50)
    b_=alpha**-0.5
    scale=1/b_-b_
    b1=b_+xz[0]*scale
    b2=b_+(1-yz[1])*scale
    b=(b1*b2)**0.5
    alphaz=1/b**2
    xb_=(xb-xz[0])/(xz[1]-xz[0])
    yb_=(yb-yz[0])/(yz[1]-yz[0])
    xd_=1-(xz[1]-xd)/(xz[1]-xz[0])
    yd_=1-(yz[1]-yd)/(yz[1]-yz[0])
    xb,yb,xd,yd=xb_yd(yb,xd,lg,alpha,10)
    ax2.set_title(f'ntp={ntp_a(alpha,xb,yb,xd,yd).real:.5f}(a),{ntp_s(alpha,xb,yb,xd,yd):.5f}(s)')
    plotMcCabe(alpha,xb,yb,xd,yd,ax=ax2)
  
  plt.show()
  plt.close()
