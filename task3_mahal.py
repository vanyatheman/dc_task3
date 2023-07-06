import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

im1 = np.array(Image.open('hol_sklad_big.png'), dtype=np.float64)
im1 = im1[:,:,:3]
# Выбор региона с кирпичами для выявления паттерна который нам нужен
Isub = im1[2000:2257, 2075:2286]

im1 = im1 / 255.
Isub = Isub / 255.

plt.figure()
plt.subplot(121)
plt.imshow(im1)
plt.title("origin")
plt.subplot(122)
plt.imshow(Isub)
plt.title("subregion")
plt.show()

# Расчет среднего значения по цветам у кучи кирпичей которые нам нужны
ar = np.mean(Isub[:,:,0])
ag = np.mean(Isub[:,:,1])
ab = np.mean(Isub[:,:,2])
a = np.array([ar,ag,ab])

# Составление ковариационной матрицы
x,y,z = np.shape(Isub)
Ivec = Isub.reshape(x*y, z)
a = np.mean(Ivec, axis = 0)
C = np.cov(Ivec.T)

m, n, l = np.shape(im1)
D = np.zeros((m, n))
C_inv = np.linalg.inv(C)

# Вычисление расстояния Махаланобиса
for i,j in itertools.product(range(m), range(n)):
    z = im1[i, j, :]
    D[i, j] = np.sqrt((z-a).T.dot(C_inv).dot(z-a))

# Трешхолд для выялвения регионов
D0_values = [1, 1.5, 1.8]
plt.figure(figsize = (12,10))
plt.subplot(221)
plt.imshow(im1)
plt.title('Original')

i = 2
for D0 in D0_values:
    mask = np.zeros((m,n), dtype = int)
    mask[D <= D0] = 1
    mask = np.dstack((mask, mask, mask))
    
    Iseg = im1 * mask
    plt.subplot(2,2,i)
    plt.imshow(Iseg)
    plt.title('Segmented, $D_0$ = '+str(D0))
    
    i += 1
 