import cv2 as cv
import numpy as np
import pathlib
import random
from matplotlib import pyplot as plt

def filtroBilateral():
    repeticoes = 8 #numero varia de 8 a 30
    max_repeticoes = 30

    img = cv.imread('images/tec9.jpg')

    temp_img = img.copy()

    b, g, r = cv.split(temp_img)

    ksize = 7
    sigmaColor = 7
    sigmaSpace = 7

    # imagem = cv.merge((tempB,tempG,tempR))
    images = []

    while repeticoes <= max_repeticoes:
        tempB = cv.bilateralFilter(b, ksize, sigmaColor, sigmaSpace)
        tempG = cv.bilateralFilter(g, ksize, sigmaColor, sigmaSpace)
        tempR = cv.bilateralFilter(r, ksize, sigmaColor, sigmaSpace)
        for i in range(0, repeticoes):
            tempB = cv.bilateralFilter(tempB, ksize, sigmaColor, sigmaSpace)
            tempG = cv.bilateralFilter(tempG, ksize, sigmaColor, sigmaSpace)
            tempR = cv.bilateralFilter(tempR, ksize, sigmaColor, sigmaSpace)
    
        imagem = cv.merge((tempB,tempG,tempR))
        images.append(imagem)
        
        repeticoes = repeticoes + 4

    return images

def sobel(img, kernel):
    sobelX = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=kernel)
    sobelY = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=kernel)
    sobelX = np.uint8(np.absolute(sobelX))
    sobelY = np.uint8(np.absolute(sobelY))
    sobel = cv.bitwise_or(sobelX, sobelY)

    return sobel

def retornaPixelsMeio(img):
    width, heigth = img.shape
    meio = []

    for i in range(1, heigth):
        temp = img[int(width/2),i]
        meio.append(temp)

    return meio

def saltandpepper(img):
    rows, columns, ch = img.shape
    percent = 0.05
    output = np.zeros(img.shape, np.uint8)

    for i in range(rows):
        for j in range(columns):
            r = random.random()
            if r < percent/2:
                output[i][j] = [0, 0, 0]
            elif r < percent:
                output[i][j] = [255,255,255]
            else:
                output[i][j] = img[i][j]
    
    return output


bilateral_images = filtroBilateral()
repeticoes = 8
path = pathlib.Path().absolute()
path_out = path.as_posix() + "/out"
for i in range(0, len(bilateral_images)):
    cv.imwrite(path_out +"/bilateral" + str(repeticoes) + ".jpg", bilateral_images[i])
    repeticoes += 4
    cv.waitKey(0)


img = cv.imread('images/Fig0107(b)(kidney-original).tif')
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imwrite(path_out+"/kidneyoriginal.jpg", img)
antes = retornaPixelsMeio(img)
img_sobel3 = sobel(img, 3)
depois_sobel = retornaPixelsMeio(img_sobel3)
img_sobel5 = sobel(img, 5)
img_sobel7 = sobel(img, 7)
img_laplace3 = cv.Laplacian(img,cv.CV_64F, ksize=3)
depois_laplace = retornaPixelsMeio(img_laplace3)
img_laplace5 = cv.Laplacian(img,cv.CV_64F, ksize=5)
img_laplace7 = cv.Laplacian(img,cv.CV_64F, ksize=7)

cv.imwrite(path_out + "/sobel3.jpg", img_sobel3)
cv.imwrite(path_out + "/sobel5.jpg", img_sobel5)
cv.imwrite(path_out + "/sobel7.jpg", img_sobel7)
cv.imwrite(path_out + "/laplacian3.jpg", img_laplace3)
cv.imwrite(path_out + "/laplacian5.jpg", img_laplace5)
cv.imwrite(path_out + "/laplacian7.jpg", img_laplace7)

plt.plot(antes)
plt.savefig(path_out +"/antessobel.png")
plt.plot(depois_sobel)
plt.savefig(path_out +"/depoissobel.png")
plt.plot(depois_laplace)
plt.savefig(path_out +"/depoislaplace.png")

barbara_saltpepper = cv.imread("images/Fig0417(a)(barbara).tif")
barbara_saltpepper = cv.cvtColor(barbara_saltpepper, cv.COLOR_BGR2RGB)
saltpepperbarbara = saltandpepper(barbara_saltpepper)
cv.imwrite(path_out + "/barbara_saltandpepper.jpg", saltpepperbarbara)

#aplica filtro media
barbaramedia3 = cv.blur(saltpepperbarbara, (3,3))
barbaramedia5 = cv.blur(saltpepperbarbara, (5,5))
barbaramedia7 = cv.blur(saltpepperbarbara, (7,7))

#aplica filtro gaussiano
barbaragaus3 = cv.GaussianBlur(saltpepperbarbara, (3, 3), 0)
barbaragaus5 = cv.GaussianBlur(saltpepperbarbara, (5, 5), 0)
barbaragaus7 = cv.GaussianBlur(saltpepperbarbara, (7, 7), 0)

#aplica mediana
barbaramediana3 = cv.medianBlur(saltpepperbarbara, 3)
barbaramediana5 = cv.medianBlur(saltpepperbarbara, 5)
barbaramediana7 = cv.medianBlur(saltpepperbarbara, 7)

cv.imwrite(path_out + "/barbara_media3.jpg", barbaramedia3)
cv.imwrite(path_out + "/barbara_media5.jpg", barbaramedia5)
cv.imwrite(path_out + "/barbara_media7.jpg", barbaramedia7)
cv.imwrite(path_out + "/barbara_gaus3.jpg", barbaragaus3)
cv.imwrite(path_out + "/barbara_gaus5.jpg", barbaragaus5)
cv.imwrite(path_out + "/barbara_gaus7.jpg", barbaramedia7)
cv.imwrite(path_out + "/barbara_mediana3.jpg", barbaramediana3)
cv.imwrite(path_out + "/barbara_mediana5.jpg", barbaramediana5)
cv.imwrite(path_out + "/barbara_mediana7.jpg", barbaramediana7)
print("Imagens prontas!")