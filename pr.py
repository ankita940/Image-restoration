import sys
import random
import math
import operator
import matplotlib.pyplot as plt
import numpy
import cv2
import networkx as nx
import itertools
from numpy import array
import pylab as pl

pixelr=[]
pixelg=[]
pixelb=[]
R=5
imggread=cv2.imread("4.jpg")
h,w,c=imggread.shape

#print imggread
#print "\n\nsdsdfsf\n"
row=h/8
col=w/8

h=(row*8)
w=(col*8)
window=[[[]for j in range(col)]for i in range(row) ]
nearneighbour=[[]for i in range(h*w)]
maincount=0



print h,w
#print imggread[0][0]
#print imgread

imgread=[[[0.0,0.0,0.0]for i in range(w)]for i in range(h)]

ccc=256.0
#print 255/256.0
for x in range(h):
	for y in range(w):
		for z in range(3):
			#imgread[x][y][z]=float(float(imgread[x][y][z])/float(ccc))
			imgread[x][y][z]=(imggread[x][y][z])/256.0

#print imggread
for x in range(h):
	for y in range(w):
		j=-1*R
		k=-1*R
		temp1=0.0
		temp2=0.0
		temp3=0.0
		while(j<=R):
			while(k<=R):
				if(x+j>=0 and y+k>=0 and x+j<=h-1 and y+k<=w-1):					
					temp1+=round(float(pow((imgread[x][y][0]-imgread[x+j][y+k][0]),2))/float(2*R+1),3)
				if(x+j>=0 and y+k>=0 and x+j<=h-1 and y+k<=w-1):					
					temp2+=round(float(pow((imgread[x][y][1]-imgread[x+j][y+k][1]),2))/float(2*R+1),3)
				if(x+j>=0 and y+k>=0 and x+j<=h-1 and y+k<=w-1):					
					temp3+=round(float(pow((imgread[x][y][2]-imgread[x+j][y+k][2]),2))/float(2*R+1),3)

				k+=1
			j+=1

		pixelb.append(temp1)			
		pixelg.append(temp2)
		pixelr.append(temp3)
		
		#window creation
		
		window[x/8][y/8].append(imgread[x][y])
		
		n=-1		#for x
					#for y
		while(n<=1):
			m=-1			
			while(m<=1):
				if(x+n>=0 and x+n<h and y+m>=0 and y+m<w ):
					if(n==0 and m==0):
						ccc=0
					else:
						nearneighbour[maincount].append(((x+n)*w)+((y+m)))
				m+=1
			n+=1
		#print maincount		
		#print nearneighbour[maincount]
		maincount+=1
		
#print nearneighbour[0]
#print nearneighbour[97]
"""
H=[i for i in range(h*w)]
G=nx.Graph()
G.add_nodes_from(H)
for i in range(h*w):
	for j in range(len(nearneighbour[i])):
		G.add_edge(i,nearneighbour[i][j])

cliques = nx.find_cliques(G)
cliques3 = [clq for clq in cliques ]

print len(cliques3)
"""

cliquesum=[0 for i in range(h*w)]
for i in range(h*w):
	G=nx.Graph()
	H=[]
	H.append(i)	
	G.add_node(i)	
	for j in nearneighbour[i]:
		if j not in H:
			H.append(j)
			G.add_node(j)
			G.add_edge(i,j)		
		for k in nearneighbour[j]:
			if k not in H:
				H.append(k)
				G.add_node(k)
				G.add_edge(j,k)
	cliques = nx.find_cliques(G)
	cliques3 = [clq for clq in cliques ]
	cliquesum[i]=len(cliques3)
	#print cliquesum[i],
xl=[]
yl=[]
for i in range(64):
	xl.append(random.randint(0,h-1))
	yl.append(random.randint(0,w-1))
windowl=[]
for i in range(64):
	windowl.append(imgread[xl[i]][yl[i]])

#print windowl

#finding mean
meanl =[0.0,0.0,0.0]
mean =[[[0.0,0.0,0.0] for i in range(col)]for i in range(row) ]
#print window[0][0][0][0][0]
for x in range(64):
		meanl[0]+=windowl[x][0]
		meanl[1]+=windowl[x][1]
		meanl[2]+=windowl[x][2]

meanl[0]/=float(64)
meanl[1]/=float(64)
meanl[2]/=float(64)	

for i in range(row):
	for j in range(col):
		for x in range(64):
				mean[i][j][0]+=window[i][j][x][0]
				mean[i][j][1]+=window[i][j][x][1]
				mean[i][j][2]+=window[i][j][x][2]	

for i in range(row):
	for j in range(col):
		mean[i][j][0]/=64
		mean[i][j][1]/=64
		mean[i][j][2]/=64

	

#print mean[0][0]
#print mean[1][3]
#print meanl

#global  feature extraction

E=[0.0,0.0,0.0]
final=999999231.0
finalmean=[]
for i in range(row):
	for j in range(col):
		En=[0.0,0.0,0.0]
		Ed1=[0.0,0.0,0.0]
		Ed2=[0.0,0.0,0.0]
		energy=[0.0,0.0,0.0]		
		temp=0.0		
		for k in range(64):
			for z in range(3):
				En[z]+=((windowl[k][z]-meanl[z])*(window[i][j][k][z]-mean[i][j][z]))
				Ed1[z]+=(pow((windowl[k][z]-meanl[z]),2))
				Ed2[z]+=(pow((window[i][j][k][z]-mean[i][j][z]),2))
		for z in range(3):
			if(En[z]<0):
				En[z]=-1*En[z]
			
			energy[z]=En[z]/(math.sqrt(Ed1[z])*Ed2[z])
			temp+=energy[z]
		#print energy		
		if(temp<final):
			final=temp
			finalwindow=window[i][j]
			finalmean=mean[i][j]

"""
print "final"
print "finalwindow",finalwindow
print "finalmean",finalmean
"""

global_pixelb=[]
global_pixelg=[]
global_pixelr=[]

dl=[]
for x in range(64):
	l=[]
	for z in range(3):
		l.append(meanl[z]-windowl[x][z])
	dl.append(l)
	

for i in range(row):
	for y in range(8):
		for j in range(col):
			for x in range(8):
				for z in range(3):
					temp=round((finalmean[z]-window[i][j][(y*8)+x][z])*(dl[(y*8)+x][z]),3)	
					if(temp<0):
						temp=temp*-1
					if(z==0):
						global_pixelb.append(temp)
					elif(z==1):
						global_pixelg.append(temp)
					else:
						global_pixelr.append(temp)

#print pixelb
"""

for i in range(h):
	for j in range(w):
		for z in range(3):
			window[i/8][j/8]
"""
energyb=[]
energyg=[]
energyr=[]
partitionb=0.0
partitiong=0.0
partitionr=0.0

probb=[]
probg=[]
probr=[]

for i in range(h*w):
	t1=round(cliquesum[i]*(pixelb[i]+global_pixelb[i]),8)
	t2=round(cliquesum[i]*(pixelg[i]+global_pixelg[i]),8)
	t3=round(cliquesum[i]*(pixelr[i]+global_pixelr[i]),8)
	#print t1
	"""
	energyb.append(math.exp(round((t1/-2),150)))
	energyg.append(math.exp(round((t2/-2),50)))
	energyr.append(math.exp(round((t3/-2),50)))
	"""	

	t4=math.exp((t1/-2))
	t5=math.exp((t2/-2))
	t6=math.exp((t3/-2))
	energyb.append(t4)
	energyg.append(t5)
	energyr.append(t6)
	"""
	partitionb+=t1
	partitiong+=t2
	partitionr+=t3
	"""
	probb.append(float(energyb[i]/(t4+t5+t6)))
	probg.append(float(energyg[i]/(t4+t5+t6)))
	probr.append(float(energyr[i]/(t4+t5+t6)))
	
	"""
	probb.append(float(energyb[i]))
	probg.append(float(energyg[i]))
	probr.append(float(energyr[i]))
	"""

#print energyb
#print partitionb

"""
for i in range(h*w):
	probb.append(float(energyb[i]/partitionb))
	probg.append(energyg[i]/partitiong)
	probr.append(energyr[i]/partitionr)

"""

showimg=[[[0.0 for i in range(3)]for i in range(w)]for i in range(h)]
for i in range(h*w):
	x=i/w
	y=i%w
	"""
	showimg[x][y][0]=int(probb[i]*256)
	showimg[x][y][1]=int(probg[i]*256)
	showimg[x][y][2]=int(probr[i]*256)
	"""


	showimg[x][y][0]=probb[i]
	showimg[x][y][1]=probg[i]
	showimg[x][y][2]=probr[i]
	
	"""
	showimg[x][y][0]=probb[i]*256
	showimg[x][y][1]=probg[i]*256
	showimg[x][y][2]=probr[i]*256
	"""
ss=array(showimg)
#print len(ss)
#print global_pixelr
#print len(probg)
#cv2.imshow('image',ss)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

def MRF(I,J,eta=2.0,zeta=1.5):
    ind =numpy.arange(numpy.shape(I)[0])
    numpy.random.shuffle(ind)
    orderx = ind.copy()
    numpy.random.shuffle(ind)

    for i in orderx:
        for j in ind:
            oldJ = J[i,j]
            J[i,j]=1
            patch = 0
            for k in range(-1,1):
                for l in range(-1,1):
                    patch += J[i,j] * J[i+k,j+l]
            energya = -eta*numpy.sum(I*J) - zeta*patch
            J[i,j]=-1
            patch = 0
            for k in range(-1,1):
                for l in range(-1,1):
                    patch += J[i,j] * J[i+k,j+l]
            energyb = -eta*numpy.sum(I*J) - zeta*patch
            if energya<energyb:
                J[i,j] = 1
            else:
                J[i,j] = -1
    #J=rgb2gray(J)
    print J
    return J

def rgb2gray(rgb):

	r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    	gray = (0.2989* r) + (0.5870 * g) + (0.1140 * b)

    	return gray
   


import Image

#image = Image.open('inumpyut1.jpg')
#image.show()

I=rgb2gray(cv2.imread("1.jpeg"))
#pl.imshow(I,cmap='gray')
#cv2.waitKey(0)

#c = pl.imread('abid.jpg')
#print c
#pl.imshow(c)
#pl.title('jio')
#I=rgb2gray(c)
#print I
#pl.figure()
#pl.imshow(I)
#pl.title('ji')


N = numpy.shape(I)[0]
print N
#I = I[:,:,0]
#I = numpy.where(I<0.1,-1,1)
#pl.figure()
#cv2.imshow('original',I)
#cv2.waitKey(0)
#pl.title('Original Image')

noise = numpy.random.rand(N,N)
J = I.copy()
ind = numpy.where(noise<0.1)
J[ind] = -J[ind]
#pl.figure()
pl.imshow(J,cmap='gray')
#cv2.waitKey(0)
pl.title('Noisy image')
newJ = J.copy()
newJ = MRF(I,newJ)
newJ=newJ*I
pl.figure()
pl.imshow(newJ,cmap='gray')
#cv2.waitKey(0)
pl.title('Denoised version')
print numpy.sum(I-J), numpy.sum(I-newJ)
pl.show()

