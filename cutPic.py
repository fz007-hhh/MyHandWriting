from PIL import Image


def cut(img_file, dx, dy):
    img = Image.open(img_file)
    n = 1
    x1 = 0
    y1 = 0
    x2 = dx
    y2 = dy
    while x2 <= img.size[1]:#纵向
        while y2 <= img.size[0]:#横向
            new_pic = 'pic/pic' + str(n) + '.jpg'
            #print('n=', n, ' x1=', x1, ' y1=', y1, ' x2=', x2, ' y2=', y2)
            img2 = img.crop((y1, x1, y2, x2))
            img2.save(new_pic)
            y1 += dy
            y2 = y1 + dy
            n += 1

        x1 = x1 + dx
        x2 = x1 + dx
        y1 = 0
        y2 = dy
        #print('------------------------------------------------------------')
    return n - 1


def main(img):
    #img = 'testPic.jpg'
    n = cut(img, 28, 28)
    return n


if __name__ == '__main__':
    img = 'testPic.jpg'
    main(img)
