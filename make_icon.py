from PIL import Image, ImageDraw

size = 256
img = Image.new('RGB', (size, size), (20, 30, 55))
d = ImageDraw.Draw(img)
for y in range(size):
    t = y / size
    r = int(20 + (108 - 20) * t)
    g = int(30 + (92 - 30) * t)
    b = int(55 + (231 - 55) * t)
    d.line([(0, y), (size, y)], fill=(r, g, b))
d.rounded_rectangle([46, 96, 210, 200], radius=14, fill=(15, 20, 38))
d.polygon([(46, 96), (70, 72), (96, 96)], fill=(255, 255, 255))
d.polygon([(96, 96), (120, 72), (146, 96)], fill=(255, 255, 255))
d.polygon([(146, 96), (170, 72), (196, 96)], fill=(255, 255, 255))
d.polygon([(108, 128), (108, 176), (150, 152)], fill=(0, 210, 255))
d.ellipse([170, 40, 196, 66], fill=(255, 107, 157))
img.save('app/static/icon.png')
img.save('app/static/icon.ico',
         sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
print('icon created')
