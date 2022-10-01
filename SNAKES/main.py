import numpy as np 
import cv2 
import matplotlib.pyplot as plt 
from scipy import sparse as sp

Image= cv2.imread('cpe.png',0) #ouverture de l'image
#Image = rgb2gray(Image) ??
#Gradient
Gx, Gy= np.gradient(Image) #Gx= Gradient selon x et Gy, gradient selon y
#Nx=np.linalg.norm(Gx)  #Norme de x
#Ny=np.linalg.norm(Gy)  #Norme de y

G=[Gx,Gy]               #Gradient
N=np.linalg.norm(G)     #Norme du gradient
#print(N)


#Algorythme snake


#Paramètres: 
alpha= 1   #si alpha grand, miniminsation energie elastique
beta= 2    #si beta grand, miniminsation energie courbure
gamma= 4   #pondere attirance des contours
K= 1000

#Initialisation 

v=np.linspace(0,2*np.pi,200) #vecteur de 200 valeurs allant de 0 a pi 
x=250*np.cos(v) +300    #coord x du snake (initiale)
y=250*np.sin(v) +250    #coord y du snake (initiale)
coords=np.array([x,y]).T   #stockage des coord du snake 


#Affichage du cercle de base + image
plt.ion()
fig=plt.figure(3)
ax=fig.add_subplot(111)
line,=ax.plot(x,y,'r',linewidth=2)
plotfig=plt.imshow(Image,'gray')
plt.savefig("plotfig.png")

#taille de x
taillex=len(x)


#matrices
D2=sp.diags([1,1,-2,1,1], [-taillex+1,-1,0,1,taillex-1],shape=(taillex,taillex)).toarray()
D4= sp.diags([-4,1,1,-4,6,-4,1,1,-4], [-taillex+1,-taillex+2,-2,-1,0,1,2,taillex-2,taillex-1],shape=(taillex,taillex)).toarray()

I=np.eye(taillex, taillex)
D=alpha*D2 -beta*D4
A= np.linalg.inv(I-D)

new_x=np.zeros(K)
new_y=np.zeros(K)


for i in range(K):
    new_x = np.dot(A, x+gamma*Gx[y.astype(int),x.astype(int)]) 
    new_y = np.dot(A, y+gamma*Gy[y.astype(int),x.astype(int)])
    plt.ion()
    fig=plt.figure(3)
    line=plt(x,y,'r',linewidth=2)
    plotfig=plt.imshow(Image,'gray')
    plt.savefig("plotfig.png")


#Affichage (Enregistrement)

plt.figure() 
#Image Normale
plt.imsave('Image-N&B.png', Image , cmap=plt.cm.gray)

plt.imshow(Image, 'binary')

#Gradient selon x
plt.imsave('Gx.png', Gx , cmap=plt.cm.gray)
plt.show()


#Verification 
print("Le programme est éxécuté")