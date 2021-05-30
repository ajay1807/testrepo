#%%
# part 1(v)
import numpy as np
import matplotlib as plt
#import csv
from random import shuffle
import math
from math import exp
from mpl_toolkits.mplot3d import Axes3D
rows = []
train=[]
valid=[]
test=[]
train_class0=[]
train_class1=[]
train_class2=[]
a=0
b=0
c=0
N=0#num of rows
#rows = np.genfromtxt ("Dataset_1_Team_3.csv", delimiter=",")
rows=np.genfromtxt("Dataset_2_Team_3.csv", delimiter=",")
#rows=np.concatenate((rows,rows2))
N=rows.shape
#for row in rows[:10]:
   # for col in row: 
    #   print("%3s"%col,end =" "),
   # print('\n')
for row in rows:
    #print(row[2])
    if row[2]==0:
        a=a+1
    elif row[2]==1:
        b=b+1
    elif row[2]==2:
        c=c+1
p0=round(a/(a+b+c),2)
p1=round(b/(a+b+c),2)
p2=round(c/(a+b+c),2)
#print(p0,p1,p2)
#assuming  normal distr
#shuffle(rows)
#using 70 percent after shuffling
for row in rows[:int(0.7*(N[0]))]:
    train.append(row)
    if row[2]==0:
        train_class0.append(row)
    elif row[2]==1:
        train_class1.append(row)
    elif row[2]==2:
        train_class2.append(row)
for row in rows[int(0.7*(N[0]))+1:int(0.85*(N[0]))]:
    valid.append(row)
for row in rows[int(0.85*(N[0]))+1:]:
    test.append(row)
train_class0= np.array(train_class0)
train_class1 = np.array(train_class1)
train_class2= np.array(train_class2)
train=np.array(train)
valid=np.array(valid)
test=np.array(test)
mean01=0
mean02=0
mean11=0
mean12=0
mean21=0
mean22=0
a=0
b=0
c=0
#estimate class conditional densities:
for row in train:
    if(row[2]==0):
        mean01=float(row[0])+mean01
        mean02=float(row[1])+mean02
        a=a+1
    elif(row[2]==1):
        mean11=float(row[0])+mean11
        mean12=float(row[1])+mean12
        b=b+1
    elif(row[2]==2):
        mean21=float(row[0])+mean21
        mean22=float(row[1])+mean22
        c=c+1
mean01=round(mean01/a,7)
mean02=round(mean02/a,7)
mean11=round(mean11/b,7)
mean12=round(mean12/b,7)
mean21=round(mean21/c,7)
mean22=round(mean22/c,7)

mean0=np.array([mean01,mean02])
mean1=np.array([mean11,mean12])
mean2=np.array([mean21,mean22])
#print(mean0)
s0xx,s0xy,s0yy,s1xx,s1xy,s1yy,s2xx,s2xy,s2yy=0,0,0,0,0,0,0,0,0
for row in train:
    if row[2]==0:
        s0xy+=(float(row[0])-mean01)*(float(row[1])-mean02)
        s0xx+=(float(row[0])-mean01)**2
        s0yy+=(float(row[1])-mean02)**2
    elif row[2]==1:
        s1xy+=(float(row[0])-mean11)*(float(row[1])-mean12)
        s1xx+=(float(row[0])-mean11)**2
        s1yy+=(float(row[1])-mean12)**2
    elif row[2]==2:
        s2xy+=(float(row[0])-mean21)*(float(row[1])-mean22)
        s2xx+=(float(row[0])-mean21)**2
        s2yy+=(float(row[1])-mean22)**2
        
s0xy=s0xy/a
s0xx=s0xx/a
s0yy=s0yy/a
s1xy=s1xy/b
s1xx=s1xx/b
s1yy=s1yy/b
s2xy=s2xy/c
s2xx=s2xx/c
s2yy=s2yy/c
#print(s0xx,s0xy,s0yy)
#print(s1xx,s1xy,s1yy)
#print(s2xx,s2xy,s2yy)
#conditional density calculation
cov_0=np.matrix([[s0xx,s0xy],[s0xy,s0yy]])
cov_1=np.matrix([[s1xx,s1xy],[s1xy,s1yy]])
cov_2=np.matrix([[s2xx,s2xy],[s2xy,s2yy]])

#cov_0=np.cov(train_class0[:,0:2],rowvar=False,bias=True)
#cov_1=np.cov(train_class1[:,0:2],rowvar=False,bias=True)
#cov_2=np.cov(train_class2[:,0:2],rowvar=False,bias=True)
#print(cov_0,cov_1,cov_2)
#%%
def class_conditional_density(input_x,mean,covariance):
    # a is the inverse of the covariance matrix
    #print(input_x," ",input_x.shape)
    a=0
    #print(input_x.shape)
    #print(mean.shape)
    a=np.linalg.inv(covariance)
    # b is the determinant of the covariance matrix
    b=0
    b=np.linalg.det(covariance)
    #print(b)
    b=(b**0.5)
    c=np.transpose(input_x-mean)
    #print(c)
    d=np.dot(c,a)
    #print(d)
    e=d.dot(input_x-mean)
    #print(e)
    f= (math.exp(-0.5*e))/(2*(math.pi)*b)
    return f

def class_conditional_density1(input_x,mean,covariance):
    b=0
    b=np.linalg.det(covariance)
    b=(np.log(b))*(-0.5)
    a=0
    a=np.linalg.inv(covariance)
    c=np.transpose(input_x-mean)
    d=c.dot(a)
    e=d.dot(input_x-mean)
    return -0.5*e+b
    
    
#ccd0= class_conditional_density(input_x,mean0,cov_0)
#ccd1= class_conditional_density(input_x,mean1,cov_1)
#ccd2= class_conditional_density(input_x,mean2,cov_2)

# l is loss matrix given in the problem
 #%%
l=[[0, 1, 2],[1, 0, 1],[2, 1, 0]]
l=np.array(l)
accuracy_5=0
for row in valid:
    s=0
    q0=class_conditional_density(row[0:2],mean0,cov_0)
    q1=class_conditional_density(row[0:2],mean1,cov_1)
    q2=class_conditional_density(row[0:2],mean2,cov_2)
    s=(q0+q1+q2)
    q0=q0/s
    q1=q1/s
    q2=q2/s
    #print(q0,q1,q2)
    R=l.dot([q0,q1,q2])
   # print(R)
    index=np.argmin(R)
   # print(index,row[2])
    if index-row[2]==0:
        accuracy_5+=1

#print(accuracy_5)    
accuracy_5=accuracy_5/valid.shape[0]
eigenvalues0,eigenvectors0=np.linalg.eig(cov_0)
eigenvalues1,eigenvectors1=np.linalg.eig(cov_1)
eigenvalues2,eigenvectors2=np.linalg.eig(cov_2)

#print(accuracy_5)
# above calculated accuracy is for bayes for all different covariance matrices

#%%
# part1(iv)
common_cov=(cov_0+cov_1+cov_2)/3
accuracy_4=0

for row in valid:
    s=0
    q0=class_conditional_density(row[0:2],mean0,common_cov)
    q1=class_conditional_density(row[0:2],mean1,common_cov)
    q2=class_conditional_density(row[0:2],mean2,common_cov)
    s=(q0+q1+q2)
    q0=q0/s
    q1=q1/s
    q2=q2/s
    #print(q0,q1,q2)
    R=l.dot([q0,q1,q2])
    #print(R)
    index=np.argmin(R)
    if index-row[2]==0:
        accuracy_4+=1
        
#print(accuracy_4)   
accuracy_4=accuracy_4/valid.shape[0]
#print(accuracy_4)

#%%
# part 1(iii)
u=cov_0[1,0]
v=cov_1[1,0]
w=cov_2[1,0]
cov_0[1,0]=cov_0[0,1]=0
cov_1[1,0]=cov_1[0,1]=0
cov_2[1,0]=cov_2[0,1]=0
accuracy_3=0
for row in valid:
    s=0
    q0=class_conditional_density(row[0:2],mean0,cov_0)
    q1=class_conditional_density(row[0:2],mean1,cov_1)
    q2=class_conditional_density(row[0:2],mean2,cov_2)
    s=(q0+q1+q2)
    q0=q0/s
    q1=q1/s
    q2=q2/s
    #print(q0,q1,q2)
    R=l.dot([q0,q1,q2])
   # print(R)
    index=np.argmin(R)
    if index-row[2]==0:
        accuracy_3+=1
#print(accuracy_4)
accuracy_3=accuracy_3/valid.shape[0]
#print(accuracy_3)
#%%
#part 1(ii)
common_cov[0,1]=common_cov[1,0]=0
accuracy_2=0

for row in valid:
    s=0
    q0=class_conditional_density(row[0:2],mean0,common_cov)
    q1=class_conditional_density(row[0:2],mean1,common_cov)
    q2=class_conditional_density(row[0:2],mean2,common_cov)
    s=(q0+q1+q2)
    q0=q0/s
    q1=q1/s
    q2=q2/s
    #print(q0,q1,q2)
    #print(l)
    #print(q0,q1,q2)
    R=l.dot([q0,q1,q2])
    #print(R)
    index=np.argmin(R)
    #print(index,row[2])
    if index-row[2]==0:
        accuracy_2+=1

#print(accuracy_2)    
accuracy_2=accuracy_2/valid.shape[0]
#print(accuracy_2)
#%%
accuracy_1=0
canc=0#to capture noise points
I=[[1,0],[0,1]]
table=np.zeros((4,4))
for row in test:
    s=0
    q0=class_conditional_density(row[0:2],mean0,I)
    q1=class_conditional_density(row[0:2],mean1,I)
    q2=class_conditional_density(row[0:2],mean2,I)
   # print(q0,q1,q2)
    s=(q0+q1+q2)
    #print(s)
    if(s!=0):
      q0=q0/s
      q1=q1/s
      q2=q2/s
      R=l.dot([q0,q1,q2])
      #print(R)
      index=np.argmin(R)
      if (index-row[2]==0):
          accuracy_1+=1
          table[index][index]+=1
      else:
          table[index][int(row[2])]+=1
              
accuracy_1=accuracy_1/(test.shape[0])
#print(accuracy_1)

#%%
# gi(x)
def g(input_x, mean , covariance,prior):
    a=np.linalg.inv(covariance)
    mean=np.transpose(np.matrix(mean))
    #print(a)
    b=np.linalg.det(covariance)
    #print(b)
    c=np.transpose(input_x).dot(-0.5*a)
    #print(c)
    c=c.dot(input_x)
    #print(c)
    #print(mean)
    d=(np.transpose(a*mean)).dot(input_x)
    #print(d.shape)
    e=(-0.5*((np.transpose(mean)).dot(a)).dot(mean))+(-0.5*math.log(b))+(math.log(prior))
    return (c+d+e)
#%%
# plotting pdf
#-500 to 700 dataset 1
    #-10 to 15 dataset 2 x
    #-40 to 20 dataset 2 y
#Create grid and multivariate normal
cov_0[1,0]=cov_0[0,1]=u
cov_1[1,0]=cov_1[0,1]=v
cov_2[1,0]=cov_2[0,1]=w
t=100
x = np.linspace(-10,15,t)#-10 to 15 
y = np.linspace(-40,20,t)
X, Y = np.meshgrid(x,y)
Z_0=np.zeros((t,t))
Z_1=np.zeros((t,t))
Z_2=np.zeros((t,t))
g_0=np.zeros((t,t))
g_1=np.zeros((t,t))
g_2=np.zeros((t,t))
G_1_2=np.zeros((t,t))
G_1_3=np.zeros((t,t))
G_2_3=np.zeros((t,t))
for i in range(t):
    inp=[]
    for j in range(t):
        inp=[X[i,j],Y[i,j]]
        inp=np.transpose(inp)
        Z_0[i,j]=class_conditional_density(inp,mean0,cov_0)
        Z_1[i,j]=class_conditional_density(inp,mean1,cov_1)
        Z_2[i,j]=class_conditional_density(inp,mean2,cov_2)
        g_0[i,j]=g(inp,mean0,cov_0,p0)
        g_1[i,j]=g(inp,mean1,cov_1,p1)
        g_2[i,j]=g(inp,mean2,cov_2,p2)
        
#%%

#Make a 3D plot
fig1 = plt.pyplot.figure()
fig2 = plt.pyplot.figure()

ax1 = fig1.gca(projection='3d')
#ax3 = fig3.gca(projection='3d')
ax1.plot_surface(X, Y, Z_0,cmap='Blues')
ax1.plot_surface(X, Y, Z_1,cmap='Greens')
ax1.plot_surface(X, Y, Z_2,cmap='Oranges')
#ax1.legend(loc='best')
ax1.set_xlabel('X1')
ax1.set_ylabel('X2')
ax1.set_zlabel('PDF')
ax1.view_init(5,85)
ax1.set_title('PDF (gaussians) for different classes')

fig=plt.pyplot.figure()
plt.pyplot.plot([0],[0],'r',label="Class 0")
plt.pyplot.plot([0],[0],'b',label="Class 1")
plt.pyplot.plot([0],[0],'g',label="Class 2")
plt.pyplot.legend(loc='best')
cp = plt.pyplot.contour(X, Y, Z_0,colors='red')
cp = plt.pyplot.contour(X, Y, Z_1,colors='blue')
cp = plt.pyplot.contour(X, Y, Z_2,colors='green')
#fig.suptitle('Contour Plot')
plt.pyplot.xlabel('X1')
plt.pyplot.ylabel('X2')
#cp=plt.pyplot.plot(test[:,0],test[:,1],'o')
#plt.colorbar(cp)

#plt.pyplot.clabel(cp, inline=True, fontsize=10)
plt.pyplot.title('Contour Plot')
plt.pyplot.show()
#%%
for i in range(t):
    for j in range(t):
        if abs(g_0[i,j]-g_1[i,j])<0.01 and g_2[i,j]<g_1[i,j] and g_2[i,j]<g_0[i,j]:
            G_1_2[i,j]=1
        if abs(g_0[i,j]-g_2[i,j])<0.01 and g_1[i,j]<g_0[i,j] and g_1[i,j]<g_2[i,j]:
            G_1_3[i,j]=1
        if abs(g_1[i,j]-g_2[i,j])<0.01 and g_0[i,j]<g_2[i,j] and g_0[i,j]<g_1[i,j]:
            G_2_3[i,j]=1
fig3 = plt.pyplot.figure()
#ax3 = fig3.gca(projection='3d')
#ax3.plot_surface(X,Y,G_1_2,cmap='Oranges')
plt.pyplot.plot([0],[0],'r',label="Class 0")
plt.pyplot.plot([0],[0],'b',label="Class 1")
plt.pyplot.plot([0],[0],'g',label="Class 2")
cp = plt.pyplot.contour(X, Y, G_1_2,colors='black')
cp = plt.pyplot.contour(X, Y, G_1_3,colors='black')
cp = plt.pyplot.contour(X, Y, G_2_3,colors='black')
cp = plt.pyplot.contour(X, Y, Z_0,colors='red')
cp = plt.pyplot.contour(X, Y, Z_1,colors='blue')
cp = plt.pyplot.contour(X, Y, Z_2,colors='green')
cp=plt.pyplot.plot(train_class1[:60,0],train_class1[:60,1],'bo',linewidth=0.1)
cp=plt.pyplot.plot(train_class2[:60,0],train_class2[:60,1],'go',linewidth=0.1)
cp=plt.pyplot.plot(train_class0[:60,0],train_class0[:60,1],'ro',linewidth=0.1)
plt.pyplot.xlabel('X1')
plt.pyplot.ylabel('X2')
plt.pyplot.title('Contour Plot, Decision Boundary and Training data')

#cp=plt.pyplot.plot(mean0[0],mean0[1],'ro')
#cp=plt.pyplot.plot(mean1[0],mean1[1],'bo')
#cp=plt.pyplot.plot(mean2[0],mean2[1],'go')
test_class0=[]
test_class1=[]
test_class2=[]
for row in test:
    if row[2]==0:
        test_class0.append(row)
    elif row[2]==1:
        test_class1.append(row)
    elif row[2]==2:
        test_class2.append(row)
test_class0=np.array(test_class0)
test_class1=np.array(test_class1)
test_class2=np.array(test_class2)
#cp=plt.pyplot.plot(train_class0[:,0],train_class0[:,1],'ro')
#cp=plt.pyplot.plot(train_class1[:,0],train_class1[:,1],'bo')
#cp=plt.pyplot.plot(train_class2[:,0],train_class2[:,1],'go')
#plt.pyplot.fill_between(x1,y1, G_1_2,G_1_3)
#cp = plt.pyplot.contour(X, Y, Z_0)
#cp = plt.pyplot.contour(X, Y, Z_1)
#cp = plt.pyplot.contour(X, Y, Z_2)

#%%
# plotting eigenvector and training dataset and contour curves
#500 for data set 1
#10 for dataset 2
#s1=s2=500
s1=10
s2=20
fig4=plt.pyplot.figure()
#cp=plt.pyplot.plot([0,1000*eigenvectors0[0,0]],[0,1000*eigenvectors0[1,0]])
#cp=plt.pyplot.plot([0,1000*eigenvectors0[0,1]],[0,1000*eigenvectors0[1,1]])
#cp=plt.pyplot.plot([0,1000*eigenvectors1[0,0]],[0,1000*eigenvectors1[1,0]])
#cp=plt.pyplot.plot([0,1000*eigenvectors1[0,1]],[0,1000*eigenvectors1[1,1]])
#cp=plt.pyplot.plot([0,1000*eigenvectors2[0,0]],[0,1000*eigenvectors2[1,0]])
#cp=plt.pyplot.plot([0,1000*eigenvectors2[0,1]],[0,1000*eigenvectors2[1,1]]
plt.pyplot.plot([0],[0],'r',label="Class 0")
plt.pyplot.plot([0],[0],'b',label="Class 1")
plt.pyplot.plot([0],[0],'g',label="Class 2")
plt.pyplot.legend(loc='best')
cp=plt.pyplot.plot([mean0[0],mean0[0]+s1*eigenvectors0[0,0]],[mean0[1],mean0[1]+s2*eigenvectors0[1,0]],'r',linewidth=4)
cp=plt.pyplot.plot([mean0[0],mean0[0]+s1*eigenvectors0[0,1]],[mean0[1],mean0[1]+s2*eigenvectors0[1,1]],'r',linewidth=4)
cp=plt.pyplot.plot([mean1[0],mean1[0]+s1*eigenvectors1[0,0]],[mean1[1],mean1[1]+s2*eigenvectors1[1,0]],'b',linewidth=4)
cp=plt.pyplot.plot([mean1[0],mean1[0]+s1*eigenvectors1[0,1]],[mean1[1],mean1[1]+s2*eigenvectors1[1,1]],'b',linewidth=4)
cp=plt.pyplot.plot([mean2[0],mean2[0]+s1*eigenvectors2[0,0]],[mean2[1],mean2[1]+s2*eigenvectors2[1,0]],'g',linewidth=4)
cp=plt.pyplot.plot([mean2[0],mean2[0]+s1*eigenvectors2[0,1]],[mean2[1],mean2[1]+s2*eigenvectors2[1,1]],'g',linewidth=4)

cp = plt.pyplot.contour(X, Y, Z_0,colors='red')
cp = plt.pyplot.contour(X, Y, Z_1,colors='blue')
cp = plt.pyplot.contour(X, Y, Z_2,colors='green')
plt.pyplot.legend(loc='best')
cp=plt.pyplot.plot(train_class1[:150,0],train_class1[:150,1],'bo',linewidth=0.1)
cp=plt.pyplot.plot(train_class2[:150,0],train_class2[:150,1],'go',linewidth=0.1)
cp=plt.pyplot.plot(train_class0[:150,0],train_class0[:150,1],'ro',linewidth=0.1)
plt.pyplot.xlabel('X1')
plt.pyplot.ylabel('X2')
plt.pyplot.title('Contour Plot, Eigenvectors and Training data')


































    

