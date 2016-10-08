
from PIL import Image

all_images = ["image/class1-energy-dB.png","image/class-1-energy-dB.png",
              "image/class1-energy-db-hist.png","image/class-1-energy-db-hist.png",
              "image/class1-energy-slope.png","image/class-1-energy-slope.png",
              "image/class1-energy-slope-hist.png","image/class-1-energy-slope-hist.png",
              "image/class1-frequency.png","image/class-1-frequency.png",
              "image/class1-frequency-hist.png","image/class-1-frequency-hist.png",
              "image/class1-NAQ.png","image/class-1-NAQ.png",
              "image/class1-NAQ-hist.png","image/class-1-NAQ-hist.png",
              "image/class1-PeakSlope.png","image/class-1-PeakSlope.png",
              "image/class1-PeakSlope-hist.png","image/class-1-PeakSlope-hist.png",
              "image/class1-stationarity-hist.png","image/class-1-stationarity-hist.png",
              "image/class1-Voiced-Unvoiced-hist.png","image/class-1-Voiced-Unvoiced-hist.png"]

for i in range(0,len(all_images),2):
    
    images = map(Image.open, [all_images[i],all_images[i+1]])
    widths, heights = zip(*(i.size for i in images))
    
    total_width = sum(widths)
    max_height = max(heights)
    
    new_im = Image.new('RGB', (total_width, max_height))
    
    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0]
    
    new_im.save(all_images[i][6:]+"-"+all_images[i+1][6:])